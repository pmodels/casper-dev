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

/* Level-2 cache for temporary asynchronous status of all user processes in the
 * world, using memory allocated on ghost process.
 * Note that it should only be accessed via RMA operations from the user side
 * to guarantee atomicity. Thus this level is not for direct user query, and the
 * synchronization between level-1 and level-2 caches only happens at set interval. */
static CSP_local_shm_region shm_global_stats_region;

/* communicator consisting all local user processes and the root ghost process */
static MPI_Comm CSP_RA_GNTF_COMM = MPI_COMM_NULL;
static int RA_LNTF_GHOST_RANK = 0;

/* parameters for local reset notification */
static CSP_ra_reset_lnotify_pkt_t reset_pkt = { CSP_RA_LNOTIFY_NONE };

static MPI_Request reset_req = MPI_REQUEST_NULL;
static int issued_reset_cnt = 0;

/* parameters for local dirty notification */
static CSP_ra_dirty_lnotify_pkt_t dirty_pkt = { CSP_RA_LNOTIFY_NONE };

static MPI_Request dirty_req = MPI_REQUEST_NULL;
static int recvd_dirty_cnt = 0;
static int local_dirty_flag = 0;
static int dirty_notify_end_flag = 0;

static inline int ra_lnotify_progress(void)
{
    int mpi_errno = MPI_SUCCESS;

    if (dirty_req != MPI_REQUEST_NULL) {
        int flag = 0;

        mpi_errno = PMPI_Test(&dirty_req, &flag, MPI_STATUS_IGNORE);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;

        if (flag) {
            recvd_dirty_cnt++;

            if (dirty_pkt.type == CSP_RA_LNOTIFY_DIRTY)
                local_dirty_flag = 1;

            if (dirty_pkt.type == CSP_RA_LNOTIFY_END)
                dirty_notify_end_flag = 1;

            CSP_ADAPT_DBG_PRINT(">>> ra_lnotify_progress: recv from ghost %s%s, recvd=%d\n",
                                (dirty_pkt.type == CSP_RA_LNOTIFY_DIRTY ? "(dirty)" : ""),
                                (dirty_pkt.type == CSP_RA_LNOTIFY_END ? "(end)" : ""),
                                recvd_dirty_cnt);
        }
    }

    /* reissue broadcast for next dirty notification */
    if (dirty_req == MPI_REQUEST_NULL && !dirty_notify_end_flag) {
        dirty_pkt.type = CSP_RA_LNOTIFY_NONE;
        mpi_errno = PMPI_Ibcast(&dirty_pkt, sizeof(CSP_ra_dirty_lnotify_pkt_t),
                                MPI_CHAR, RA_LNTF_GHOST_RANK, CSP_RA_GNTF_COMM, &dirty_req);
    }

    return mpi_errno;
}

static inline int ra_lnotify_issue_reset(void)
{
    int mpi_errno = MPI_SUCCESS;
    int flag = 0;

    mpi_errno = PMPI_Test(&reset_req, &flag, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    /* skip notify if an outstanding one exists */
    if (flag) {
        reset_pkt.type = CSP_RA_LNOTIFY_RESET;
        mpi_errno = PMPI_Isend(&reset_pkt, sizeof(CSP_ra_reset_lnotify_pkt_t),
                               MPI_CHAR, RA_LNTF_GHOST_RANK, CSP_RA_LNOTIFY_RESET_TAG,
                               CSP_RA_GNTF_COMM, &reset_req);
        issued_reset_cnt++;
        CSP_ADAPT_DBG_PRINT(">>> ra_lnotify_issue_reset: issued_reset_cnt=%d\n", issued_reset_cnt);
    }
    else {
        CSP_ADAPT_DBG_PRINT(">>> ra_lnotify_issue_reset (skipped)\n");
    }

    return MPI_SUCCESS;
}

static int ra_lnotify_complete(void)
{
    int mpi_errno = MPI_SUCCESS;

    /* finish reset notify */
    mpi_errno = PMPI_Wait(&reset_req, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;
    CSP_ADAPT_DBG_PRINT(">>> >> ra_lnotify_complete: wait previous reset\n");

    reset_pkt.type = CSP_RA_LNOTIFY_END;
    mpi_errno = PMPI_Send(&reset_pkt, sizeof(CSP_ra_reset_lnotify_pkt_t),
                          MPI_CHAR, RA_LNTF_GHOST_RANK, CSP_RA_LNOTIFY_RESET_TAG, CSP_RA_GNTF_COMM);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;
    CSP_ADAPT_DBG_PRINT(">>> >> ra_lnotify_complete: reset done\n");

    /* finish dirty notify */
    while (!dirty_notify_end_flag) {
        mpi_errno = ra_lnotify_progress();
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;
    }
    CSP_ADAPT_DBG_PRINT(">>> >> ra_lnotify_complete: dirty done\n");

    CSP_ADAPT_DBG_PRINT(">>> ra_lnotify_complete: done\n");
    return mpi_errno;
}

static inline int ra_lnotify_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    mpi_errno = PMPI_Ibcast(&dirty_pkt, sizeof(CSP_ra_dirty_lnotify_pkt_t), MPI_CHAR,
                            RA_LNTF_GHOST_RANK, CSP_RA_GNTF_COMM, &dirty_req);
    return mpi_errno;
}

/* Update asynchronous status in local cache and in the global synchronization cache
 * on ghost process (blocking call). It is called when local state is updated, or
 * receive other processes' state in win-collective calls.
 * The caller can set flag to enable or disable update to the ghost cache, and to
 * control the global synchronization on ghost processes.
 * Usually when local state is updated at set interval, the remote update and global
 * synchronization is always required; when updating local states after user-level
 * remote exchange (i.e., in win-coll calls), only the local root process needs to
 * update ghost cache, and global synchronization should be skipped. */
int CSP_ra_gsync_update(int count, int *user_world_ranks, CSP_async_stat * stats,
                        CSP_gsync_update_flag flag)
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

    /* update ghost cache */
    if (flag > CSP_GSYNC_UPDATE_LOCAL) {
        mpi_errno = PMPI_Win_lock(MPI_LOCK_EXCLUSIVE, CSP_RA_GSYNC_GHOST_LOCAL_RANK,
                                  0, shm_global_stats_region.win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

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

        mpi_errno = PMPI_Win_unlock(CSP_RA_GSYNC_GHOST_LOCAL_RANK, shm_global_stats_region.win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        CSP_ADAPT_DBG_PRINT(">>> ra_gsync_update count=%d (remote)\n", count);

        if (flag == CSP_GSYNC_UPDATE_GHOST_SYNCED) {
            mpi_errno = ra_lnotify_issue_reset();
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
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

    mpi_errno = ra_lnotify_progress();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* check if ghost cache is dirty, if not return immediately. */
    if (local_dirty_flag == 0)
        goto fn_exit;

    PMPI_Comm_size(CSP_COMM_USER_WORLD, &user_nprocs);
    PMPI_Comm_rank(CSP_COMM_USER_WORLD, &user_rank);

    /* per-integer atomic read. */
    mpi_errno = PMPI_Win_lock(MPI_LOCK_SHARED, CSP_RA_GSYNC_GHOST_LOCAL_RANK,
                              0, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Get_accumulate(NULL, 0, MPI_INT, ra_gsync_local_cache, user_nprocs, MPI_INT,
                                    CSP_RA_GSYNC_GHOST_LOCAL_RANK, 0, user_nprocs, MPI_INT,
                                    MPI_NO_OP, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Win_unlock(CSP_RA_GSYNC_GHOST_LOCAL_RANK, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    local_dirty_flag = 0;

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

    ra_lnotify_complete();

    if (CSP_RA_GNTF_COMM && CSP_RA_GNTF_COMM != MPI_COMM_NULL) {
        PMPI_Comm_free(&CSP_RA_GNTF_COMM);
        CSP_RA_GNTF_COMM = MPI_COMM_NULL;
    }
    if (shm_global_stats_region.win && shm_global_stats_region.win != MPI_WIN_NULL) {
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

static int ra_comm_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Comm tmp_gsync_comm = MPI_COMM_NULL;
    MPI_Group local_ug_group = MPI_GROUP_NULL;
    int *excl_ranks = NULL, i, idx = 0;
    int ra_local_ghost_rank = CSP_RA_GSYNC_GHOST_LOCAL_RANK;
    int ra_lntf_rank = 0, ra_lntf_nprocs = 0;

    /* help ghost create gsync communicator */
    mpi_errno = PMPI_Comm_split(MPI_COMM_WORLD, 0, 1, &tmp_gsync_comm);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* create user-root ghost communicator */
    excl_ranks = CSP_calloc(CSP_ENV.num_g, sizeof(int));        /* at least 1 */
    for (idx = 0, i = 0; i < CSP_ENV.num_g; i++) {
        if (i == CSP_RA_GSYNC_GHOST_LOCAL_RANK)
            continue;
        excl_ranks[idx++] = i;
    }
    mpi_errno = PMPI_Group_excl(CSP_GROUP_LOCAL, CSP_ENV.num_g - 1, excl_ranks, &local_ug_group);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Comm_create_group(CSP_COMM_LOCAL, local_ug_group, 0, &CSP_RA_GNTF_COMM);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Group_translate_ranks(CSP_GROUP_LOCAL, 1, &ra_local_ghost_rank,
                                           local_ug_group, &RA_LNTF_GHOST_RANK);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    PMPI_Comm_rank(CSP_RA_GNTF_COMM, &ra_lntf_rank);
    PMPI_Comm_size(CSP_RA_GNTF_COMM, &ra_lntf_nprocs);
    CSP_ADAPT_DBG_PRINT(" ra_init: create ra_lntf_comm, I am %d/%d, ghost %d\n", ra_lntf_rank,
                        ra_lntf_nprocs, RA_LNTF_GHOST_RANK);
  fn_exit:
    if (excl_ranks)
        free(excl_ranks);
    if (local_ug_group != MPI_GROUP_NULL)
        PMPI_Group_free(&local_ug_group);
    if (tmp_gsync_comm != MPI_COMM_NULL)
        PMPI_Comm_free(&tmp_gsync_comm);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int CSP_ra_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint region_size = 0, r_size = 0;
    void *local_base = NULL;
    int user_rank = 0, user_nprocs;
    int r_disp_unit = 0, i;
    CSP_async_stat init_async_stat = CSP_ASYNC_NONE;

    /* initialize local state */
    CSP_ra_update_async_stat(CSP_ENV.async_config);

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    PMPI_Comm_rank(CSP_COMM_USER_WORLD, &user_rank);
    PMPI_Comm_size(CSP_COMM_USER_WORLD, &user_nprocs);

    mpi_errno = ra_comm_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* translate to basic datatype for atomic access */
    CSP_assert(sizeof(int) >= sizeof(CSP_async_stat));

    region_size = sizeof(int) * user_nprocs;
    shm_global_stats_region.win = MPI_WIN_NULL;
    shm_global_stats_region.base = NULL;
    shm_global_stats_region.size = region_size;

    mpi_errno = PMPI_Win_allocate_shared(0, 1, MPI_INFO_NULL, CSP_RA_GNTF_COMM,
                                         &local_base, &shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* get shared region's local address. */
    mpi_errno = PMPI_Win_shared_query(shm_global_stats_region.win,
                                      RA_LNTF_GHOST_RANK,
                                      &r_size, &r_disp_unit, &shm_global_stats_region.base);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* allocate local cache. */
    ra_gsync_local_cache = CSP_calloc(user_nprocs, sizeof(CSP_async_stat));
    CSP_ADAPT_DBG_PRINT(" ra_init: allocated shm_reg %p, local cache=%p, size %ld\n",
                        shm_global_stats_region.base, ra_gsync_local_cache, region_size);

    /* send my user rank to the gsync ghost */
    mpi_errno = PMPI_Send(&user_rank, 1, MPI_INT, RA_LNTF_GHOST_RANK, 0, CSP_RA_GNTF_COMM);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* initialize local cache */
    init_async_stat = CSP_ra_get_async_stat();
    for (i = 0; i < user_nprocs; i++)
        ra_gsync_local_cache[i] = init_async_stat;

    mpi_errno = ra_lnotify_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

  fn_exit:
    return mpi_errno;

  fn_fail:
    CSP_ra_finalize();
    goto fn_exit;
}
#endif
