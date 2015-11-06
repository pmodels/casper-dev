/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED

/* ========================================================================
 * Ghost-offload Asynchronous Adaptation (GADPT) Routines (User Side)
 * ======================================================================== */

/* Level-1 cache for temporary asynchronous status of all user processes in the
 * world, using local memory.
 * It provides latest asynchronous status of any user process with very low
 * query overhead, but the value is not accurate. It is synchronized with level-2
 * cache once level-2 cache has been changed (dirty).*/
CSP_async_stat *gadpt_local_cache = NULL;

/* Level-2 cache for temporary asynchronous status of all user processes in the
 * world, using memory allocated on ghost process.
 * Note that it should only be accessed via RMA operations from the user side
 * to guarantee atomicity. Thus this level is not for direct user query, and the
 * synchronization between level-1 and level-2 caches only happens when either
 * side is updated. */
static CSP_local_shm_region gadpt_l2_cache_region;

static MPI_Comm GADPT_LNTF_COMM = MPI_COMM_NULL;        /* local notify communicator.
                                                         * consist of all local users and
                                                         * root ghost process */
static int GADPT_LNTF_GHOST_RANK = 0;

/* parameters for local reset notification */
static CSP_gadpt_reset_lnotify_pkt_t reset_pkt = { CSP_GADPT_LNOTIFY_NONE };

static MPI_Request reset_req = MPI_REQUEST_NULL;
static int issued_reset_cnt = 0;

/* parameters for local dirty notification */
static CSP_gadpt_dirty_lnotify_pkt_t dirty_pkt = { CSP_GADPT_LNOTIFY_NONE };

static MPI_Request dirty_req = MPI_REQUEST_NULL;
static int recvd_dirty_cnt = 0;
static int local_dirty_flag = 0;
static int dirty_notify_end_flag = 0;

#ifdef CSP_ENABLE_ADAPT_PROF
CSP_adapt_rma_prof ADAPT_RMA_PROF;

static void adapt_prof_reset(void)
{
    ADAPT_RMA_PROF.put.to_ghost = ADAPT_RMA_PROF.put.to_user = 0;
    ADAPT_RMA_PROF.get.to_ghost = ADAPT_RMA_PROF.get.to_user = 0;
    ADAPT_RMA_PROF.acc.to_ghost = ADAPT_RMA_PROF.acc.to_user = 0;
    ADAPT_RMA_PROF.get_acc.to_ghost = ADAPT_RMA_PROF.get_acc.to_user = 0;
    ADAPT_RMA_PROF.fop.to_ghost = ADAPT_RMA_PROF.fop.to_user = 0;
    ADAPT_RMA_PROF.cas.to_ghost = ADAPT_RMA_PROF.cas.to_user = 0;
}

static void adapt_prof_dump(void)
{
    CSP_INFO_PRINT(4, "ADAPT rma profiling (op to user:ghost): ");
    if (ADAPT_RMA_PROF.put.to_ghost > 0 || ADAPT_RMA_PROF.put.to_user > 0) {
        CSP_INFO_PRINT(4, " put %d:%d, ", ADAPT_RMA_PROF.put.to_user, ADAPT_RMA_PROF.put.to_ghost);
    }
    if (ADAPT_RMA_PROF.get.to_ghost > 0 || ADAPT_RMA_PROF.get.to_user > 0) {
        CSP_INFO_PRINT(4, " get %d:%d, ", ADAPT_RMA_PROF.get.to_user, ADAPT_RMA_PROF.get.to_ghost);
    }
    if (ADAPT_RMA_PROF.acc.to_ghost > 0 || ADAPT_RMA_PROF.acc.to_user > 0) {
        CSP_INFO_PRINT(4, " acc %d:%d, ", ADAPT_RMA_PROF.acc.to_user, ADAPT_RMA_PROF.acc.to_ghost);
    }
    if (ADAPT_RMA_PROF.get_acc.to_ghost > 0 || ADAPT_RMA_PROF.get_acc.to_user > 0) {
        CSP_INFO_PRINT(4, " get_acc %d:%d, ", ADAPT_RMA_PROF.get_acc.to_user,
                       ADAPT_RMA_PROF.get_acc.to_ghost);
    }
    if (ADAPT_RMA_PROF.fop.to_ghost > 0 || ADAPT_RMA_PROF.fop.to_user > 0) {
        CSP_INFO_PRINT(4, " fop %d:%d, ", ADAPT_RMA_PROF.fop.to_user, ADAPT_RMA_PROF.fop.to_ghost);
    }
    if (ADAPT_RMA_PROF.cas.to_ghost > 0 || ADAPT_RMA_PROF.cas.to_user > 0) {
        CSP_INFO_PRINT(4, " cas %d:%d, ", ADAPT_RMA_PROF.cas.to_user, ADAPT_RMA_PROF.cas.to_ghost);
    }
    CSP_INFO_PRINT(4, "\n");

    adapt_prof_reset();
}
#endif

static inline int gadpt_lnotify_progress(void)
{
    int mpi_errno = MPI_SUCCESS;

    if (dirty_req != MPI_REQUEST_NULL) {
        int flag = 0;

        mpi_errno = PMPI_Test(&dirty_req, &flag, MPI_STATUS_IGNORE);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;

        if (flag) {
            recvd_dirty_cnt++;

            if (dirty_pkt.type == CSP_GADPT_LNOTIFY_DIRTY)
                local_dirty_flag = 1;

            if (dirty_pkt.type == CSP_GADPT_LNOTIFY_END)
                dirty_notify_end_flag = 1;

            CSP_ADAPT_DBG_PRINT(">>> gadpt_lnotify_progress: recv from ghost %s%s, recvd=%d\n",
                                (dirty_pkt.type == CSP_GADPT_LNOTIFY_DIRTY ? "(dirty)" : ""),
                                (dirty_pkt.type == CSP_GADPT_LNOTIFY_END ? "(end)" : ""),
                                recvd_dirty_cnt);
        }
    }

    /* reissue broadcast for next dirty notification */
    if (dirty_req == MPI_REQUEST_NULL && !dirty_notify_end_flag) {
        dirty_pkt.type = CSP_GADPT_LNOTIFY_NONE;
        mpi_errno = PMPI_Ibcast(&dirty_pkt, sizeof(CSP_gadpt_dirty_lnotify_pkt_t),
                                MPI_CHAR, GADPT_LNTF_GHOST_RANK, GADPT_LNTF_COMM, &dirty_req);
    }

    return mpi_errno;
}

static inline int gadpt_lnotify_issue_reset(void)
{
    int mpi_errno = MPI_SUCCESS;
    int flag = 0;

    mpi_errno = PMPI_Test(&reset_req, &flag, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    /* skip notify if an outstanding one exists */
    if (flag) {
        reset_pkt.type = CSP_GADPT_LNOTIFY_RESET;
        mpi_errno = PMPI_Isend(&reset_pkt, sizeof(CSP_gadpt_reset_lnotify_pkt_t),
                               MPI_CHAR, GADPT_LNTF_GHOST_RANK, CSP_GADPT_LNOTIFY_RESET_TAG,
                               GADPT_LNTF_COMM, &reset_req);
        issued_reset_cnt++;
        CSP_ADAPT_DBG_PRINT(">>> gadpt_lnotify_issue_reset: issued_reset_cnt=%d\n",
                            issued_reset_cnt);
    }
    else {
        CSP_ADAPT_DBG_PRINT(">>> gadpt_lnotify_issue_reset (skipped)\n");
    }

    return MPI_SUCCESS;
}

static int gadpt_lnotify_complete(void)
{
    int mpi_errno = MPI_SUCCESS;

    /* finish reset notify */
    mpi_errno = PMPI_Wait(&reset_req, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;
    CSP_ADAPT_DBG_PRINT(">>> >> gadpt_lnotify_complete: wait previous reset\n");

    reset_pkt.type = CSP_GADPT_LNOTIFY_END;
    mpi_errno = PMPI_Send(&reset_pkt, sizeof(CSP_gadpt_reset_lnotify_pkt_t),
                          MPI_CHAR, GADPT_LNTF_GHOST_RANK, CSP_GADPT_LNOTIFY_RESET_TAG,
                          GADPT_LNTF_COMM);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;
    CSP_ADAPT_DBG_PRINT(">>> >> gadpt_lnotify_complete: reset done\n");

    /* finish dirty notify */
    while (!dirty_notify_end_flag) {
        mpi_errno = gadpt_lnotify_progress();
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;
    }
    CSP_ADAPT_DBG_PRINT(">>> >> gadpt_lnotify_complete: dirty done\n");

    CSP_ADAPT_DBG_PRINT(">>> gadpt_lnotify_complete: done\n");
    return mpi_errno;
}

static inline int gadpt_lnotify_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    mpi_errno = PMPI_Ibcast(&dirty_pkt, sizeof(CSP_gadpt_dirty_lnotify_pkt_t), MPI_CHAR,
                            GADPT_LNTF_GHOST_RANK, GADPT_LNTF_COMM, &dirty_req);
    return mpi_errno;
}

static int gadpt_comm_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Comm tmp_gsync_comm = MPI_COMM_NULL;
    MPI_Group lntf_group = MPI_GROUP_NULL;
    int *excl_ranks = NULL, i, idx = 0;
    int local_ghost_rank = CSP_RA_GSYNC_GHOST_LOCAL_RANK;
    int lntf_rank = 0, lntf_nprocs = 0;

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
    mpi_errno = PMPI_Group_excl(CSP_GROUP_LOCAL, CSP_ENV.num_g - 1, excl_ranks, &lntf_group);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Comm_create_group(CSP_COMM_LOCAL, lntf_group, 0, &GADPT_LNTF_COMM);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Group_translate_ranks(CSP_GROUP_LOCAL, 1, &local_ghost_rank,
                                           lntf_group, &GADPT_LNTF_GHOST_RANK);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    PMPI_Comm_rank(GADPT_LNTF_COMM, &lntf_rank);
    PMPI_Comm_size(GADPT_LNTF_COMM, &lntf_nprocs);
    CSP_ADAPT_DBG_PRINT(" gadpt_init: create lntf_comm, I am %d/%d, ghost %d\n", lntf_rank,
                        lntf_nprocs, GADPT_LNTF_GHOST_RANK);
  fn_exit:
    if (excl_ranks)
        free(excl_ranks);
    if (lntf_group != MPI_GROUP_NULL)
        PMPI_Group_free(&lntf_group);
    if (tmp_gsync_comm != MPI_COMM_NULL)
        PMPI_Comm_free(&tmp_gsync_comm);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

/* Update asynchronous status in GADPT caches.
 * It is called when local state is updated, or receive other processes' state in
 * win-collective calls. The caller can set flag to for three level updates:
 * - only update local cache
 * - update both local and level-2 cache
 * - update both local and level-2 cache synchronously (which means all ghosts are
 *   changed to the same value, thus next ghost synchronization can be skipped).
 *   For example, win-collective update can set this flag to avoid unnecessary
 *   synchronization. */
int CSP_gadpt_update(int count, int *user_world_ranks, CSP_async_stat * stats,
                     CSP_gadpt_update_flag flag)
{
    int mpi_errno = MPI_SUCCESS;
    int i = 0, rank = 0;
    MPI_Aint target_disp = 0;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    /* write to local cache. */
    for (i = 0; i < count; i++) {
        rank = user_world_ranks[i];
        gadpt_local_cache[rank] = stats[i];
    }

    /* update ghost cache */
    if (flag > CSP_GADPT_UPDATE_LOCAL) {
        mpi_errno = PMPI_Win_lock(MPI_LOCK_EXCLUSIVE, GADPT_LNTF_GHOST_RANK,
                                  0, gadpt_l2_cache_region.win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        /* per-integer atomic write. */
        for (i = 0; i < count; i++) {
            rank = user_world_ranks[i];
            target_disp = sizeof(int) * rank;

            mpi_errno = PMPI_Accumulate(&gadpt_local_cache[rank], 1, MPI_INT,
                                        GADPT_LNTF_GHOST_RANK, target_disp,
                                        1, MPI_INT, MPI_REPLACE, gadpt_l2_cache_region.win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }

        mpi_errno = PMPI_Win_unlock(GADPT_LNTF_GHOST_RANK, gadpt_l2_cache_region.win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        CSP_ADAPT_DBG_PRINT(">>> gadpt_update count=%d (remote)\n", count);

        if (flag == CSP_GADPT_UPDATE_GHOST_SYNCED) {
            mpi_errno = gadpt_lnotify_issue_reset();
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
int CSP_gadpt_refresh(void)
{
    int mpi_errno = MPI_SUCCESS;
    int user_nprocs = 0, user_rank = 0;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    mpi_errno = gadpt_lnotify_progress();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* check if ghost cache is dirty, if not return immediately. */
    if (local_dirty_flag == 0)
        goto fn_exit;

    PMPI_Comm_size(CSP_COMM_USER_WORLD, &user_nprocs);
    PMPI_Comm_rank(CSP_COMM_USER_WORLD, &user_rank);

    /* per-integer atomic read. */
    mpi_errno = PMPI_Win_lock(MPI_LOCK_SHARED, GADPT_LNTF_GHOST_RANK, 0, gadpt_l2_cache_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Get_accumulate(NULL, 0, MPI_INT, gadpt_local_cache, user_nprocs, MPI_INT,
                                    GADPT_LNTF_GHOST_RANK, 0, user_nprocs, MPI_INT,
                                    MPI_NO_OP, gadpt_l2_cache_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Win_unlock(GADPT_LNTF_GHOST_RANK, gadpt_l2_cache_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    local_dirty_flag = 0;

    CSP_ADAPT_DBG_PRINT(">>> gadpt_refresh: done\n");

    if (CSP_ENV.verbose > 2 && user_rank == 0) {
        int async_on_cnt = 0, async_off_cnt = 0, i = 0;
        for (i = 0; i < user_nprocs; i++) {
            if (gadpt_local_cache[i] == CSP_ASYNC_ON) {
                async_on_cnt++;
            }
            else {
                async_off_cnt++;
            }
        }
        CSP_INFO_PRINT(3, "GADPT local cache refresh: on %d; off %d\n",
                       async_on_cnt, async_off_cnt);

#ifdef CSP_ENABLE_ADAPT_PROF
        adapt_prof_dump();
#endif
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static void gadpt_finalize(void)
{
    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        return;

    gadpt_lnotify_complete();

    if (GADPT_LNTF_COMM && GADPT_LNTF_COMM != MPI_COMM_NULL) {
        PMPI_Comm_free(&GADPT_LNTF_COMM);
        GADPT_LNTF_COMM = MPI_COMM_NULL;
    }
    if (gadpt_l2_cache_region.win && gadpt_l2_cache_region.win != MPI_WIN_NULL) {
        PMPI_Win_free(&gadpt_l2_cache_region.win);
    }

    gadpt_l2_cache_region.win = MPI_WIN_NULL;
    gadpt_l2_cache_region.base = NULL;
    gadpt_l2_cache_region.size = 0;

    if (gadpt_local_cache) {
        free(gadpt_local_cache);
        gadpt_local_cache = NULL;
    }
}

static int gadpt_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint region_size = 0, r_size = 0;
    void *local_base = NULL;
    int user_rank = 0, user_nprocs;
    int r_disp_unit = 0, i;
    CSP_async_stat init_async_stat = CSP_ASYNC_NONE;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;


    PMPI_Comm_rank(CSP_COMM_USER_WORLD, &user_rank);
    PMPI_Comm_size(CSP_COMM_USER_WORLD, &user_nprocs);

    mpi_errno = gadpt_comm_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* translate to basic datatype for atomic access */
    CSP_assert(sizeof(int) >= sizeof(CSP_async_stat));

    region_size = sizeof(int) * user_nprocs;
    gadpt_l2_cache_region.win = MPI_WIN_NULL;
    gadpt_l2_cache_region.base = NULL;
    gadpt_l2_cache_region.size = region_size;

    mpi_errno = PMPI_Win_allocate_shared(0, 1, MPI_INFO_NULL, GADPT_LNTF_COMM,
                                         &local_base, &gadpt_l2_cache_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* get shared region's local address. */
    mpi_errno = PMPI_Win_shared_query(gadpt_l2_cache_region.win,
                                      GADPT_LNTF_GHOST_RANK,
                                      &r_size, &r_disp_unit, &gadpt_l2_cache_region.base);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* allocate local cache. */
    gadpt_local_cache = CSP_calloc(user_nprocs, sizeof(CSP_async_stat));
    CSP_ADAPT_DBG_PRINT(" gadpt_init: allocated shm_reg %p, local cache=%p, size %ld\n",
                        gadpt_l2_cache_region.base, gadpt_local_cache, region_size);

    /* send my user rank to the gsync ghost */
    mpi_errno = PMPI_Send(&user_rank, 1, MPI_INT, GADPT_LNTF_GHOST_RANK, 0, GADPT_LNTF_COMM);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* initialize local cache */
    init_async_stat = CSP_adpt_get_async_stat();
    for (i = 0; i < user_nprocs; i++)
        gadpt_local_cache[i] = init_async_stat;

    mpi_errno = gadpt_lnotify_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

  fn_exit:
    return mpi_errno;

  fn_fail:
    gadpt_finalize();
    goto fn_exit;
}

void CSP_adpt_finalize(void)
{
    return gadpt_finalize();
}

int CSP_adpt_init(void)
{
    int mpi_errno = MPI_SUCCESS;

    /* initialize local state */
    CSP_adpt_update_async_stat(CSP_ENV.async_config);

    mpi_errno = gadpt_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

#ifdef CSP_ENABLE_ADAPT_PROF
    adapt_prof_reset();
#endif

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
#endif
