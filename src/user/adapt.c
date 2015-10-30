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


/* Update asynchronous status in local cache and in the global synchronization cache
 * on ghost process (blocking call). It is called when local state is updated, or
 * receive other processes' state in win-collective calls.
 * The caller can set remote_flag to enable or disable remote update to the cache on
 * ghost process. Usually when local state is updated at set interval, the remote
 * update is always required; when updating mutiple states, only the local root
 * process needs to update ghost cache. */
int CSP_ra_gsync_update(int count, int *user_world_ranks, CSP_async_stat * stats, int remote_flag)
{
    int mpi_errno = MPI_SUCCESS;
    int i = 0, rank = 0;
    MPI_Aint target_disp = 0;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    /* write to local cache. */
    for (i = 0; i < count; i++) {
        rank = user_world_ranks[i];
        ra_gsync_local_cache[rank] = stats[i];
    }

    if (remote_flag == 1) {
        /* per-integer atomic write. */
        for (i = 0; i < count; i++) {
            rank = user_world_ranks[i];
            target_disp = sizeof(int) * rank;

            mpi_errno = PMPI_Accumulate(&ra_gsync_local_cache[rank], 1, MPI_INT,
                                        CSP_RA_GSYNC_GHOST_LOCAL_RANK, target_disp,
                                        1, MPI_INT, MPI_REPLACE, shm_global_stats_region.win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }

        mpi_errno = PMPI_Win_flush(CSP_RA_GSYNC_GHOST_LOCAL_RANK, shm_global_stats_region.win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

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
    int user_nprocs = 0, user_rank = 0;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    PMPI_Comm_size(CSP_COMM_USER_WORLD, &user_nprocs);
    PMPI_Comm_rank(CSP_COMM_USER_WORLD, &user_rank);

    /* per-integer atomic read. */
    mpi_errno = PMPI_Get_accumulate(NULL, 0, MPI_INT, ra_gsync_local_cache, user_nprocs, MPI_INT,
                                    CSP_RA_GSYNC_GHOST_LOCAL_RANK, 0, user_nprocs, MPI_INT,
                                    MPI_NO_OP, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Win_flush(CSP_RA_GSYNC_GHOST_LOCAL_RANK, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;


    CSP_ADAPT_DBG_PRINT(">>> ra_gsync_refresh: done\n");

    if (CSP_ENV.verbose > 2 && user_rank == 0) {
        int async_on_cnt = 0, async_off_cnt = 0, i = 0;
        for (i = 0; i < user_nprocs; i++) {
            if (ra_gsync_local_cache[i] == CSP_ASYNC_ON) {
                async_on_cnt++;
            }
            else {
                async_off_cnt++;
            }
        }
        CSP_INFO_PRINT(3, "GSYNC local cache: on %d; off %d\n", async_on_cnt, async_off_cnt);
    }

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

    /* initialize local state */
    CSP_ra_update_async_stat(CSP_ENV.async_config);

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
    CSP_ADAPT_DBG_PRINT(" ra_init: allocated shm_reg %p, local cache=%p, size %ld\n",
                        shm_global_stats_region.base, ra_gsync_local_cache, region_size);

    /* send my user rank to the gsync ghost */
    mpi_errno = PMPI_Send(&user_rank, 1, MPI_INT, CSP_RA_GSYNC_GHOST_LOCAL_RANK, 0, CSP_COMM_LOCAL);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* initialize local cache */
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
