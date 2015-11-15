/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "cspg.h"
#include "sbcast.h"

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED


/* =================================================================
 * Ghost-offload Asynchronous Synchronization Routines (Ghost Side)
 * ================================================================= */

/* The ghost processes and the local user processes share a memory region (level-2
 * cache), and use it to store the asynchronous status of all user processes in the
 * world. The synchronization can be summarized in following steps:
 *
 * - The local user process updates its status on the shared region once its local
 *   status is changed.
 * - [GSYNC] Every node has a single ghost process, called gsync ghost, handling the
 *   background global synchronization, it exchanges all the status with other gsync
 *   ghosts at set interval (gadpt_gsync_interval).
 * - [DIRTY] Once a ghost process finished gsync, it updates the shared region and
 *   notifies all local users that the level-2 cache has been updated (dirty).
 * - When the user processes received the dirty notify, they will refresh all status
 *   from the shared region to its local cache.
 * - [RESET] Local user process can notify ghost to reset the interval timer for GSYNC
 *   when it has been synchronized globally in win-collective calls.
 *
 * Note that the GADPT routine should only effect the PUT/GET operations on local
 * user processes. ACC-like operations should only use the status synchronized
 * through win-collective calls. This is because it requires ordering and atomicity,
 * which means two operations, which are issued from the same origin or from two
 * different origin processes but to the same target, must be always redirected
 * to the same process (either user process or ghost process) before their remote
 * completion. */

static CSP_local_shm_region gadpt_l2_cache_region;      /* asynchronous status of all user processes.
                                                         * one copy per node. */
static CSP_async_stat *shm_atomic_access_buf = NULL;
static int num_local_stats = 0, num_all_stats = 0;
static int symmetric_num_stats = 0;

static MPI_Comm GADPT_LNTF_COMM = MPI_COMM_NULL;        /* all local users and the root ghost process */
static MPI_Comm GADPT_GSYNC_COMM = MPI_COMM_NULL;       /* all gsync ghost processes. */
static int GADPT_LNTF_GHOST_RANK = 0;

static double gadpt_gsync_interval_sta = 0.0;

/* parameters for local reset notification */
static CSP_gadpt_reset_lnotify_pkt_t reset_pkt = { CSP_GADPT_LNOTIFY_NONE };

static MPI_Request reset_req = MPI_REQUEST_NULL;
static int recvd_reset_cnt = 0, done_reset_cnt = 0;

/* parameters for local dirty notification */
static CSP_gadpt_dirty_lnotify_pkt_t dirty_pkt = { CSP_GADPT_LNOTIFY_NONE };

static MPI_Request dirty_req = MPI_REQUEST_NULL;
static int issued_dirty_cnt = 0;

/* parameters for global synchronization */
static MPI_Request gsync_req = MPI_REQUEST_NULL;
static int *gsync_user_ranks = NULL;
static int *gsync_user_stats = NULL;
static int *stats_cnts = NULL, *stats_disps = NULL;     /*in integer */
static int issued_gsync_cnt = 0;

/* Per-element atomic load shared level-2 cache to shm_atomic_access_buf.
 * Caller should only call the load function and then access shm_atomic_access_buf,
 * but not direct access to window region. */
static inline int gadpt_atomic_load_l2_cache(void)
{
    int mpi_errno = MPI_SUCCESS;

    /* per-element atomic read */
    mpi_errno = PMPI_Get_accumulate(NULL, 0, MPI_INT, shm_atomic_access_buf, num_all_stats,
                                    MPI_INT, GADPT_LNTF_GHOST_RANK, 0, num_all_stats,
                                    MPI_INT, MPI_NO_OP, gadpt_l2_cache_region.win);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    mpi_errno = PMPI_Win_flush(GADPT_LNTF_GHOST_RANK, gadpt_l2_cache_region.win);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    return mpi_errno;
}

/* Per-element atomic store shared level-2 cache from shm_atomic_access_buf.
 * Caller should only update shm_atomic_access_buf and call the store function,
 * but not direct access to window region. */
static inline int gadpt_atomic_store_l2_cache(void)
{
    int mpi_errno = MPI_SUCCESS;

    /* per-element atomic write */
    mpi_errno = PMPI_Accumulate(shm_atomic_access_buf, num_all_stats, MPI_INT,
                                GADPT_LNTF_GHOST_RANK, 0, num_all_stats, MPI_INT,
                                MPI_REPLACE, gadpt_l2_cache_region.win);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    mpi_errno = PMPI_Win_flush(GADPT_LNTF_GHOST_RANK, gadpt_l2_cache_region.win);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    return mpi_errno;
}

static inline void gadpt_gsync_reset(void)
{
    double now = 0.0;

    now = PMPI_Wtime();
    CSPG_ADAPT_DBG_PRINT(">>> CSPG_gadpt_gsync_reset: %.4lf(s)\n", now - gadpt_gsync_interval_sta);

    gadpt_gsync_interval_sta = now;
}

static inline int gadpt_gsync_iallgatherv(void)
{
    int mpi_errno = MPI_SUCCESS;
    if (symmetric_num_stats == 1) {
        mpi_errno = PMPI_Iallgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                    gsync_user_stats, num_local_stats, MPI_INT,
                                    GADPT_GSYNC_COMM, &gsync_req);
    }
    else {
        mpi_errno = PMPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                     gsync_user_stats, stats_cnts, stats_disps,
                                     MPI_INT, GADPT_GSYNC_COMM, &gsync_req);
    }

    return mpi_errno;
}

static inline int gadpt_lnotify_issue_dirty(void)
{
    int mpi_errno = MPI_SUCCESS;
    int flag = 0;

    /* test on the outstanding issuing notification */
    mpi_errno = PMPI_Test(&dirty_req, &flag, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    /* issue notification to users when previous one is done, otherwise skip. */
    if (flag) {
        dirty_pkt.type = CSP_GADPT_LNOTIFY_DIRTY;
        mpi_errno = PMPI_Ibcast(&dirty_pkt, sizeof(CSP_gadpt_dirty_lnotify_pkt_t),
                                MPI_CHAR, GADPT_LNTF_GHOST_RANK, GADPT_LNTF_COMM, &dirty_req);
        issued_dirty_cnt++;
        CSPG_ADAPT_DBG_PRINT(">>> gadpt_lnotify_issue_dirty, issued_dirty_cnt=%d\n",
                             issued_dirty_cnt);
    }
    else {
        CSPG_ADAPT_DBG_PRINT(">>> gadpt_lnotify_issue_dirty (skipped)\n");
    }

    return mpi_errno;
}

static inline int gadpt_lnotify_progress(void)
{
    int mpi_errno = MPI_SUCCESS;

    /* receiving progress for reset notify */
    if (reset_req != MPI_REQUEST_NULL) {
        MPI_Status stat;
        int flag = 0;

        mpi_errno = PMPI_Test(&reset_req, &flag, &stat);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;

        if (flag) {
            recvd_reset_cnt++;

            if (reset_pkt.type == CSP_GADPT_LNOTIFY_RESET)
                gadpt_gsync_reset();

            if (reset_pkt.type == CSP_GADPT_LNOTIFY_END)
                done_reset_cnt++;

            CSPG_ADAPT_DBG_PRINT(">>> gadpt_lnotify_progress: recv from %d, %s%s, recvd=%d,"
                                 "done=%d/%d\n", stat.MPI_SOURCE,
                                 (reset_pkt.type == CSP_GADPT_LNOTIFY_RESET ? "(reset)" : ""),
                                 (reset_pkt.type == CSP_GADPT_LNOTIFY_END ? "(end)" : ""),
                                 recvd_reset_cnt, done_reset_cnt, num_local_stats);
        }
    }

    /* always need post a receive till all users are done. */
    if (reset_req == MPI_REQUEST_NULL && done_reset_cnt < num_local_stats) {
        mpi_errno = PMPI_Irecv(&reset_pkt, sizeof(CSP_gadpt_reset_lnotify_pkt_t), MPI_CHAR,
                               MPI_ANY_SOURCE, CSP_GADPT_LNOTIFY_RESET_TAG, GADPT_LNTF_COMM,
                               &reset_req);
    }

    return mpi_errno;
}

static int gadpt_lnotify_complete(void)
{
    int mpi_errno = MPI_SUCCESS;

    /* finish reset notify */
    while (done_reset_cnt < num_local_stats) {
        mpi_errno = gadpt_lnotify_progress();
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;
    }

    /* finish the previous broadcast */
    mpi_errno = PMPI_Wait(&dirty_req, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;
    CSPG_ADAPT_DBG_PRINT(">>> gadpt_lnotify_complete: previous ibcast done\n");

    /* send last broadcast for dirty flag */
    dirty_pkt.type = CSP_GADPT_LNOTIFY_END;
    mpi_errno = PMPI_Ibcast(&dirty_pkt, sizeof(CSP_gadpt_dirty_lnotify_pkt_t),
                            MPI_CHAR, GADPT_LNTF_GHOST_RANK, GADPT_LNTF_COMM, &dirty_req);
    CSPG_ADAPT_DBG_PRINT(">>> gadpt_lnotify_complete: issue last ibcast\n");

    mpi_errno = PMPI_Wait(&dirty_req, MPI_STATUS_IGNORE);
    CSPG_ADAPT_DBG_PRINT(">>> gadpt_lnotify_complete: done\n");

    return mpi_errno;
}

static inline int gadpt_lnotify_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    mpi_errno = PMPI_Irecv(&reset_pkt, sizeof(CSP_gadpt_reset_lnotify_pkt_t), MPI_CHAR,
                           MPI_ANY_SOURCE, CSP_GADPT_LNOTIFY_RESET_TAG, GADPT_LNTF_COMM,
                           &reset_req);
    return mpi_errno;
}

static inline int gadpt_gsync_ready(void)
{
    return (gsync_req == MPI_REQUEST_NULL);
}

/* Issue global synchronization among ghost roots. */
static int gadpt_gsync_issue(void)
{
    int mpi_errno = MPI_SUCCESS;
    int my_stats_disp = 0;
    int i, gsync_rank = 0, gsync_nprocs = 0;

    PMPI_Comm_rank(GADPT_GSYNC_COMM, &gsync_rank);
    PMPI_Comm_size(GADPT_GSYNC_COMM, &gsync_nprocs);
    my_stats_disp = stats_disps[gsync_rank];

    memset(gsync_user_stats, 0, num_all_stats * sizeof(int));

    /* read state.
     * Note that we store states by rank in cache (rank can be non-contiguous in odd_even),
     * but in exchange buffer, we put the states by node. */
    mpi_errno = gadpt_atomic_load_l2_cache();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    for (i = 0; i < num_local_stats; i++) {
        int user_rank = gsync_user_ranks[my_stats_disp + i];
        gsync_user_stats[my_stats_disp + i] = shm_atomic_access_buf[user_rank];
    }

    /* exchange */
    mpi_errno = gadpt_gsync_iallgatherv();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    issued_gsync_cnt++;
    CSPG_ADAPT_DBG_PRINT(">>> gadpt_gsyc_issue, issued_gsync_cnt=%d\n", issued_gsync_cnt);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Make progress on the current outstanding global synchronization.*/
static int gadpt_gsync_progress(void)
{
    int mpi_errno = MPI_SUCCESS;
    int test_flag = 0;
    int i, gsync_rank = 0, gsync_nprocs = 0;

    PMPI_Comm_rank(GADPT_GSYNC_COMM, &gsync_rank);
    PMPI_Comm_size(GADPT_GSYNC_COMM, &gsync_nprocs);

    if (gsync_req != MPI_REQUEST_NULL) {
        mpi_errno = PMPI_Test(&gsync_req, &test_flag, MPI_STATUS_IGNORE);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;

        if (test_flag) {
            /* write cache of local status
             * Note that we store states by rank in cache (rank can be non-contiguous in odd_even),
             * but in exchange buffer, we put the states by node. */
            for (i = 0; i < gsync_nprocs; i++) {        /*per node */
                int disp = stats_disps[i], j;
                for (j = 0; j < stats_cnts[i]; j++) {   /*state on node i */
                    int user_rank = gsync_user_ranks[disp + j];
                    shm_atomic_access_buf[user_rank] = (CSP_async_stat) gsync_user_stats[disp + j];
                }
            }

            mpi_errno = gadpt_atomic_store_l2_cache();
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;

#if defined(CSPG_DEBUG) || defined(CSPG_ADAPT_DEBUG)
            {
                int async_on_cnt = 0, async_off_cnt = 0;
                for (i = 0; i < num_all_stats; i++) {
                    if (gsync_user_stats[i] == CSP_ASYNC_ON) {
                        async_on_cnt++;
                    }
                    else {
                        async_off_cnt++;
                    }
                }
                CSPG_ADAPT_DBG_PRINT(">>> gadpt_gsycn_progress: allgather done, on %d; off %d\n",
                                     async_on_cnt, async_off_cnt);
            }
#endif

            mpi_errno = gadpt_lnotify_issue_dirty();
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int gadpt_gsync_complete(void)
{
    int mpi_errno = MPI_SUCCESS;
    int gsync_nprocs = 0, gsync_rank = 0;
    int max_issued_gsync_cnt = 0;

    PMPI_Comm_size(GADPT_GSYNC_COMM, &gsync_nprocs);
    PMPI_Comm_rank(GADPT_GSYNC_COMM, &gsync_rank);

    /* ensure all ghosts are arrived and gather gsync count */
    mpi_errno = PMPI_Allreduce(&issued_gsync_cnt, &max_issued_gsync_cnt, 1,
                               MPI_INT, MPI_MAX, GADPT_GSYNC_COMM);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;
    CSPG_ADAPT_DBG_PRINT(">>> >> gadpt_gsync_complete: issued_gsync_cnt %d, max %d\n",
                         issued_gsync_cnt, max_issued_gsync_cnt);

    /* there must be at most 1 missing allgather, since have to receive data from every one. */
    CSP_assert((max_issued_gsync_cnt == issued_gsync_cnt + 1) ||
               (max_issued_gsync_cnt == issued_gsync_cnt));

    if (max_issued_gsync_cnt == issued_gsync_cnt + 1) {
        /* finish previous allgather */
        mpi_errno = PMPI_Wait(&gsync_req, MPI_STATUS_IGNORE);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;
        CSPG_ADAPT_DBG_PRINT(">>> >> gadpt_gsync_complete: previous allgather done\n");

        /* issue last allgather */
        mpi_errno = gadpt_gsync_iallgatherv();
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
        CSPG_ADAPT_DBG_PRINT(">>> >> gadpt_gsync_complete: issue last allgather\n");
    }

    /* wait on the last allgather */
    mpi_errno = PMPI_Wait(&gsync_req, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;
    CSPG_ADAPT_DBG_PRINT(">>> gadpt_gsync_complete: done\n");

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static CSP_async_stat adpt_get_init_async_stat(void)
{
    CSP_async_stat init_async_stat = CSP_ASYNC_NONE;

    switch (CSP_ENV.async_config) {
    case CSP_ASYNC_CONFIG_ON:
    case CSP_ASYNC_CONFIG_AUTO:
        init_async_stat = CSP_ASYNC_ON;
        break;
    case CSP_ASYNC_CONFIG_OFF:
        init_async_stat = CSP_ASYNC_OFF;
        break;
    }
    return init_async_stat;
}

static int gadpt_gsync_init()
{
    int mpi_errno = MPI_SUCCESS;
    int i = 0, idx = 0;
    int my_stats_disp = 0;
    int gsync_rank = 0, gsync_nprocs = 0;
    int lnft_nprocs = 0;
    MPI_Request *reqs = NULL;

    PMPI_Comm_rank(GADPT_GSYNC_COMM, &gsync_rank);
    PMPI_Comm_size(GADPT_GSYNC_COMM, &gsync_nprocs);
    PMPI_Comm_size(GADPT_LNTF_COMM, &lnft_nprocs);

    /* initialize global synchronization parameters */
    gsync_user_ranks = CSP_calloc(num_all_stats, sizeof(int));
    gsync_user_stats = CSP_calloc(num_all_stats, sizeof(int));
    stats_cnts = CSP_calloc(gsync_nprocs, sizeof(int));
    stats_disps = CSP_calloc(gsync_nprocs, sizeof(int));
    reqs = CSP_calloc(lnft_nprocs - 1, sizeof(MPI_Request));

    /* gather number of states on every ghost */
    stats_cnts[gsync_rank] = num_local_stats;
    mpi_errno = PMPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                               stats_cnts, 1, MPI_INT, GADPT_GSYNC_COMM);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* translate displacement for allgather */
    stats_disps[0] = 0;
    symmetric_num_stats = 1;
    for (i = 1; i < gsync_nprocs; i++) {
        stats_disps[i] = stats_disps[i - 1] + stats_cnts[i - 1];
        symmetric_num_stats &= (stats_cnts[i] == stats_cnts[0]);
    }
    my_stats_disp = stats_disps[gsync_rank];

#if defined(CSPG_DEBUG) || defined(CSPG_ADAPT_DEBUG)
    CSPG_ADAPT_DBG_PRINT(" gadpt_init: gather num_local_stats, symmetric=%d, node cnt/disps:\n",
                         symmetric_num_stats);
    for (i = 0; i < gsync_nprocs; i++)
        CSPG_ADAPT_DBG_PRINT(" \t[%d] = %d/%d\n", i, stats_cnts[i], stats_disps[i]);
#endif

    /* gather local user ranks in comm_user_world. */
    idx = 0;
    for (i = 0; i < lnft_nprocs; i++) {
        if (i == GADPT_LNTF_GHOST_RANK)
            continue;
        mpi_errno = PMPI_Irecv(&gsync_user_ranks[my_stats_disp + idx], 1, MPI_INT,
                               i, 0, GADPT_LNTF_COMM, &reqs[idx]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
        idx++;
    }
    mpi_errno = PMPI_Waitall(idx, reqs, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* exchange user ranks in user world */
    if (symmetric_num_stats) {
        mpi_errno = PMPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, gsync_user_ranks,
                                   num_local_stats, MPI_INT, GADPT_GSYNC_COMM);
    }
    else {
        mpi_errno = PMPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                    gsync_user_ranks, stats_cnts, stats_disps,
                                    MPI_INT, GADPT_GSYNC_COMM);
    }
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    CSPG_ADAPT_DBG_PRINT(" gadpt_init: gather users ranks in user world:\n");
#if defined(CSPG_DEBUG) || defined(CSPG_ADAPT_DEBUG)
    for (i = 0; i < gsync_nprocs; i++) {
        int disp = stats_disps[i], j;
        CSPG_ADAPT_DBG_PRINT(" \t ghost[%d]\n", i);
        for (j = 0; j < stats_cnts[i]; j++) {
            CSPG_ADAPT_DBG_PRINT(" \t\t user[%d] = %d\n", j, gsync_user_ranks[disp + j]);
        }
    }
#endif

    gadpt_gsync_interval_sta = PMPI_Wtime();

  fn_exit:
    if (reqs)
        free(reqs);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int gadpt_comm_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Group local_ug_group = MPI_GROUP_NULL;
    int i = 0, idx = 0, *excl_ranks = NULL;
    int local_rank = 0;

    PMPI_Comm_rank(CSP_COMM_LOCAL, &local_rank);

    /* note that all ghosts arrive here for communicator creation. */

    /* create a gsync ghost communicator */
    mpi_errno = PMPI_Comm_split(MPI_COMM_WORLD, local_rank == CSP_RA_GSYNC_GHOST_LOCAL_RANK,
                                1, &GADPT_GSYNC_COMM);

    if (local_rank == CSP_RA_GSYNC_GHOST_LOCAL_RANK) {
        int gsync_rank = 0, gsync_nprocs = 0;
        PMPI_Comm_rank(GADPT_GSYNC_COMM, &gsync_rank);
        PMPI_Comm_size(GADPT_GSYNC_COMM, &gsync_nprocs);
        CSPG_ADAPT_DBG_PRINT(" gadpt_init: create gsync_comm, I am %d/%d\n", gsync_rank,
                             gsync_nprocs);
    }

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

    mpi_errno = PMPI_Comm_create_group(CSP_COMM_LOCAL, local_ug_group, 0, &GADPT_LNTF_COMM);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    if (local_rank == CSP_RA_GSYNC_GHOST_LOCAL_RANK) {
        int ra_lntf_rank = 0, ra_lntf_nprocs = 0;
        PMPI_Comm_rank(GADPT_LNTF_COMM, &ra_lntf_rank);
        PMPI_Comm_size(GADPT_LNTF_COMM, &ra_lntf_nprocs);
        CSPG_ADAPT_DBG_PRINT(" gadpt_init: create lntf_comm, I am %d/%d, ghost %d\n", ra_lntf_rank,
                             ra_lntf_nprocs, GADPT_LNTF_GHOST_RANK);
    }

  fn_exit:
    if (excl_ranks)
        free(excl_ranks);
    if (local_ug_group != MPI_GROUP_NULL)
        PMPI_Group_free(&local_ug_group);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

static void gadpt_finalize(void)
{
    int local_rank = 0;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        return;

    PMPI_Comm_rank(CSP_COMM_LOCAL, &local_rank);
    if (local_rank != CSP_RA_GSYNC_GHOST_LOCAL_RANK)
        return;

    /* complete all local notification and global synchronization */
    gadpt_lnotify_complete();
    gadpt_gsync_complete();

    if (gadpt_l2_cache_region.win && gadpt_l2_cache_region.win != MPI_WIN_NULL) {
        PMPI_Win_unlock(GADPT_LNTF_GHOST_RANK, gadpt_l2_cache_region.win);
        PMPI_Win_free(&gadpt_l2_cache_region.win);
    }

    gadpt_l2_cache_region.win = MPI_WIN_NULL;
    gadpt_l2_cache_region.base = NULL;

    if (shm_atomic_access_buf) {
        free(shm_atomic_access_buf);
        shm_atomic_access_buf = NULL;
    }
    if (stats_cnts) {
        free(stats_cnts);
        stats_cnts = NULL;
    }
    if (stats_disps) {
        free(stats_disps);
        stats_disps = NULL;
    }
    if (gsync_user_ranks) {
        free(gsync_user_ranks);
        gsync_user_ranks = NULL;
    }
    if (gsync_user_stats) {
        free(gsync_user_stats);
        gsync_user_stats = NULL;
    }
    if (GADPT_LNTF_COMM && GADPT_LNTF_COMM != MPI_COMM_NULL) {
        PMPI_Comm_free(&GADPT_LNTF_COMM);
        GADPT_LNTF_COMM = MPI_COMM_NULL;
    }
    if (GADPT_GSYNC_COMM && GADPT_GSYNC_COMM != MPI_COMM_NULL) {
        PMPI_Comm_free(&GADPT_GSYNC_COMM);
        GADPT_GSYNC_COMM = MPI_COMM_NULL;
    }
}

static int gadpt_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    int local_rank = 0, local_nprocs = 0, nprocs = 0;
    MPI_Aint region_size = 0;
    CSP_async_stat init_async_stat = CSP_ASYNC_NONE;
    int i;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    PMPI_Comm_rank(CSP_COMM_LOCAL, &local_rank);
    PMPI_Comm_size(CSP_COMM_LOCAL, &local_nprocs);
    PMPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    num_all_stats = nprocs - CSP_ENV.num_g * CSP_NUM_NODES;
    num_local_stats = local_nprocs - CSP_ENV.num_g;     /* number of processes per node can be different,
                                                         * but number of ghosts per node is fixed.*/
    mpi_errno = gadpt_comm_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    if (local_rank != CSP_RA_GSYNC_GHOST_LOCAL_RANK)
        goto fn_exit;

    /* allocate shared region among local node for storing all processes' statuses */
    region_size = sizeof(int) * num_all_stats;
    gadpt_l2_cache_region.win = MPI_WIN_NULL;
    gadpt_l2_cache_region.base = NULL;
    gadpt_l2_cache_region.size = region_size;

    /* only the global synchronizing ghost allocates shared region. */
    mpi_errno = PMPI_Win_allocate_shared(region_size, 1, MPI_INFO_NULL,
                                         GADPT_LNTF_COMM, &gadpt_l2_cache_region.base,
                                         &gadpt_l2_cache_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = PMPI_Win_lock(MPI_LOCK_SHARED, GADPT_LNTF_GHOST_RANK, MPI_MODE_NOCHECK,
                              gadpt_l2_cache_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    CSPG_ADAPT_DBG_PRINT
        (" gadpt_init: allocated shm_reg %p, size %ld, num_local_stats %d, num_all_stats %d\n",
         gadpt_l2_cache_region.base, region_size, num_local_stats, num_all_stats);

    /* temporary buffer for atomic access to shared cache. */
    shm_atomic_access_buf = CSP_calloc(num_all_stats, sizeof(int));

    /* initialize asynchronous state in cache */
    init_async_stat = adpt_get_init_async_stat();
    for (i = 0; i < num_all_stats; i++)
        shm_atomic_access_buf[i] = init_async_stat;

    mpi_errno = gadpt_atomic_store_l2_cache();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = gadpt_lnotify_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* initialize global synchronization */
    mpi_errno = gadpt_gsync_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

  fn_exit:
    return mpi_errno;
  fn_fail:
    gadpt_finalize();
    goto fn_exit;
}

void CSPG_adpt_finalize(void)
{
    return gadpt_finalize();
}

int CSPG_adpt_init(void)
{
    return gadpt_init();
}

/* Globally synchronize the asynchronous status of user processes at set interval. */
int CSPG_gadpt_sync(void)
{
    int mpi_errno = MPI_SUCCESS;
    int local_rank = 0;
    double now = 0.0;

    PMPI_Comm_rank(CSP_COMM_LOCAL, &local_rank);
    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME || local_rank > 0)
        goto fn_exit;

    gadpt_gsync_progress();
    gadpt_lnotify_progress();

    /* only issue synchronization if interval is large enough. */
    now = PMPI_Wtime();
    if ((now - gadpt_gsync_interval_sta - CSP_ENV.gadpt_gsync_interval) < 0)
        goto fn_exit;

    /* if not ready for next issue, return */
    if (!gadpt_gsync_ready())
        goto fn_exit;

    /* TODO: add update rate check. */

    mpi_errno = gadpt_gsync_issue();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = gadpt_gsync_progress();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    gadpt_gsync_interval_sta = now;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#endif
