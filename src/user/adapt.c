/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED

/* Level-1 cache for temporary asynchronous status of all user processes in the
 * world, using local memory.
 * It provides latest asynchronous status of any user process with very low
 * query overhead, but the value is not accurate. It is synchronized with level-2
 * cache at set interval.*/
CSP_async_stat *ra_gsync_local_cache = NULL;

double ra_gsync_interavl_sta = 0;

/* Level-2 cache for temporary asynchronous status of all user processes in the
 * world, using memory allocated on ghost process.
 * Note that it should only be accessed via RMA operations from the user side
 * to guarantee atomicity. Thus this level is not for direct user query, and the
 * synchronization between level-1 and level-2 caches only happens at set interval. */
static CSP_local_shm_region shm_global_stats_region;

/* The displacement of my status on the shared window. */
static MPI_Aint my_stat_local_index = 0;
static MPI_Aint my_stat_shm_disp = 0;


/* Update local asynchronous status to the global synchronization cache on
 * ghost process (blocking call).
 * On return, the data must have been written to the memory on ghost process. */
int CSP_ra_gsync_update(CSP_async_stat my_stat)
{
    int mpi_errno = MPI_SUCCESS;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    /* write to local cache. */
    ra_gsync_local_cache[my_stat_local_index] = my_stat;

    /* per-integer atomic write. */
    mpi_errno = PMPI_Accumulate(&ra_gsync_local_cache[my_stat_local_index], 1, MPI_INT,
                                CSP_RA_GSYNC_GHOST_LOCAL_RANK, my_stat_shm_disp, 1,
                                MPI_INT, MPI_REPLACE, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Win_flush(CSP_RA_GSYNC_GHOST_LOCAL_RANK, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    CSP_DBG_PRINT(">>> ra_gsync_update: update cache[%ld]=%d\n", my_stat_local_index, my_stat);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Reload temporary asynchronous status for all user processes in the world from
 * the global synchronization cache located on ghost process (blocking call).
 * On return, the data must have been copied to the local memory.*/
int CSP_ra_gsync_refresh(void)
{
    int mpi_errno = MPI_SUCCESS;
    int user_nprocs = 0;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    PMPI_Comm_size(CSP_COMM_USER_WORLD, &user_nprocs);

    /* per-integer atomic read. */
    mpi_errno = PMPI_Get_accumulate(NULL, 0, MPI_INT, ra_gsync_local_cache, user_nprocs, MPI_INT,
                                    CSP_RA_GSYNC_GHOST_LOCAL_RANK, 0, user_nprocs, MPI_INT,
                                    MPI_NO_OP, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Win_flush(CSP_RA_GSYNC_GHOST_LOCAL_RANK, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    CSP_DBG_PRINT(">>> ra_gsync_refresh: done\n");

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

void CSP_ra_finalize(void)
{
    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        return;

    if (shm_global_stats_region.win && shm_global_stats_region.win != MPI_WIN_NULL) {
        PMPI_Win_unlock(CSP_RA_GSYNC_GHOST_LOCAL_RANK, shm_global_stats_region.win);
        PMPI_Win_free(&shm_global_stats_region.win);
    }

    shm_global_stats_region.win = MPI_WIN_NULL;
    shm_global_stats_region.base = NULL;
    shm_global_stats_region.size = 0;

    if (ra_gsync_local_cache) {
        free(ra_gsync_local_cache);
        ra_gsync_local_cache = NULL;
    }
}

int CSP_ra_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint region_size = 0, r_size = 0;
    void *local_base = NULL;
    int user_rank = 0, user_nprocs;
    int r_disp_unit = 0, i;
    MPI_Comm tmp_gsync_comm = MPI_COMM_NULL;
    CSP_async_stat init_async_stat = CSP_ASYNC_NONE;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    PMPI_Comm_rank(CSP_COMM_USER_WORLD, &user_rank);
    PMPI_Comm_size(CSP_COMM_USER_WORLD, &user_nprocs);

    /* help ghost create gsync communicator */
    mpi_errno = PMPI_Comm_split(MPI_COMM_WORLD, 0, 1, &tmp_gsync_comm);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* translate to basic datatype for atomic access */
    CSP_assert(sizeof(int) >= sizeof(CSP_async_stat));

    region_size = sizeof(int) * user_nprocs;
    shm_global_stats_region.win = MPI_WIN_NULL;
    shm_global_stats_region.base = NULL;
    shm_global_stats_region.size = region_size;

    mpi_errno = PMPI_Win_allocate_shared(0, 1, MPI_INFO_NULL, CSP_COMM_LOCAL,
                                         &local_base, &shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* do not need exclusive access, only for later load/store. */
    mpi_errno = PMPI_Win_lock(MPI_LOCK_SHARED, CSP_RA_GSYNC_GHOST_LOCAL_RANK,
                              MPI_MODE_NOCHECK, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* get shared region's local address. */
    mpi_errno = PMPI_Win_shared_query(shm_global_stats_region.win, CSP_RA_GSYNC_GHOST_LOCAL_RANK,
                                      &r_size, &r_disp_unit, &shm_global_stats_region.base);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* allocate local cache. */
    ra_gsync_local_cache = CSP_calloc(user_nprocs, sizeof(CSP_async_stat));

    my_stat_local_index = user_rank;
    my_stat_shm_disp = sizeof(int) * user_rank;
    CSP_DBG_PRINT(" ra_init: allocated shm_reg %p, size %ld, my_disp=%lx\n",
                  shm_global_stats_region.base, region_size, my_stat_shm_disp);

    /* send my user rank to the gsync ghost */
    mpi_errno = PMPI_Send(&user_rank, 1, MPI_INT, CSP_RA_GSYNC_GHOST_LOCAL_RANK, 0, CSP_COMM_LOCAL);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* initialize local state and local cache */
    CSP_ra_update_async_stat(CSP_ENV.async_config);
    init_async_stat = CSP_ra_get_async_stat();
    for (i = 0; i < user_nprocs; i++)
        ra_gsync_local_cache[i] = init_async_stat;

    ra_gsync_interavl_sta = MPI_Wtime();

  fn_exit:
    if (tmp_gsync_comm != MPI_COMM_NULL)
        PMPI_Comm_free(&tmp_gsync_comm);
    return mpi_errno;

  fn_fail:
    CSP_ra_finalize();
    goto fn_exit;
}
#endif
