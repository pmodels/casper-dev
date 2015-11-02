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

/* This file defines the routines for global synchronization of asynchronous
 * status on ghost process side.
 *
 * The ghost processes and the local user processes share a memory region,
 * and use it to store the asynchronous status of all user processes in the world.
 * The synchronization can be summarized in following steps:
 *
 * - The local user process updates its status on the shared region at set intervals.
 * - Every node has a single ghost process, called gsync ghos, handling the background
 *   global synchronization, it frequently checks local updates and then broadcasts any
 *   change to other remote gsync ghost.
 * - Once a gsync ghost process received a message, it updates its local region
 *   according to the received packet.
 * - The local user process could check the latest status of other user processes
 *   via the shared region, and then decide the target process for RMA operations.
 *
 * Note that it should only effect the PUT/GET operations on local user processes.
 * ACC-like operations should only use the status synchronized through collective
 * scheduling routine on user processes. This is because it requires ordering
 * and atomicity, which means two operations, which are issued from the same origin
 * or from two different origin processes but to the same target, must be always
 * redirected to the same process (either user process or ghost process) before
 * their remote completion. */

static CSP_local_shm_region shm_global_stats_region;    /* asynchronous status of all user processes.
                                                         * one copy per node. */
static CSP_async_stat *shm_global_stats_ptr = NULL;     /* point to shm region, do not free */
static int num_local_stats = 0, num_all_stats = 0;
static int symmetric_num_stats = 0;

static MPI_Comm CSPG_RA_LNTF_COMM = MPI_COMM_NULL;      /* all local users and the root ghost process */
static MPI_Comm CSPG_RA_GSYNC_COMM = MPI_COMM_NULL;     /* all gsync ghost processes. */
static int RA_LNTF_GHOST_RANK = 0;

static double ra_gsync_interval_sta = 0.0;

/* parameters for local reset notification */
static CSP_ra_reset_lnotify_pkt_t reset_pkt = { CSP_RA_LNOTIFY_NONE };

static MPI_Request reset_req = MPI_REQUEST_NULL;
static int recvd_reset_cnt = 0, done_reset_cnt = 0;

/* parameters for local dirty notification */
static CSP_ra_dirty_lnotify_pkt_t dirty_pkt = { CSP_RA_LNOTIFY_NONE };

static MPI_Request dirty_req = MPI_REQUEST_NULL;
static int issued_dirty_cnt = 0;

/* parameters for global synchronization */
static MPI_Request gsync_req = MPI_REQUEST_NULL;
static int *gsync_user_ranks = NULL;
static int *gsync_user_stats = NULL;
static int *stats_cnts = NULL, *stats_disps = NULL;     /*in integer */
static int issued_gsync_cnt = 0;


static inline void ra_gsync_reset(void)
{
    double now = 0.0;

    now = PMPI_Wtime();
    CSPG_ADAPT_DBG_PRINT(">>> CSPG_ra_gsync_reset: %.4lf(s)\n", now - ra_gsync_interval_sta);

    ra_gsync_interval_sta = now;
}

static inline int ra_gsync_iallgatherv(void)
{
    int mpi_errno = MPI_SUCCESS;
    if (symmetric_num_stats == 1) {
        mpi_errno = PMPI_Iallgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                    gsync_user_stats, num_local_stats, MPI_INT,
                                    CSPG_RA_GSYNC_COMM, &gsync_req);
    }
    else {
        mpi_errno = PMPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                     gsync_user_stats, stats_cnts, stats_disps,
                                     MPI_INT, CSPG_RA_GSYNC_COMM, &gsync_req);
    }

    return mpi_errno;
}

static inline int ra_lnotify_issue_dirty(void)
{
    int mpi_errno = MPI_SUCCESS;
    int flag = 0;

    /* test on the outstanding issuing notification */
    mpi_errno = PMPI_Test(&dirty_req, &flag, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    /* issue notification to users when previous one is done, otherwise skip. */
    if (flag) {
        dirty_pkt.type = CSP_RA_LNOTIFY_DIRTY;
        mpi_errno = PMPI_Ibcast(&dirty_pkt, sizeof(CSP_ra_dirty_lnotify_pkt_t),
                                MPI_CHAR, RA_LNTF_GHOST_RANK, CSPG_RA_LNTF_COMM, &dirty_req);
        issued_dirty_cnt++;
        CSPG_ADAPT_DBG_PRINT(">>> ra_lnotify_issue_dirty, issued_dirty_cnt=%d\n", issued_dirty_cnt);
    }
    else {
        CSPG_ADAPT_DBG_PRINT(">>> ra_lnotify_issue_dirty (skipped)\n");
    }

    return mpi_errno;
}

static inline int ra_lnotify_progress(void)
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

            if (reset_pkt.type == CSP_RA_LNOTIFY_RESET)
                ra_gsync_reset();

            if (reset_pkt.type == CSP_RA_LNOTIFY_END)
                done_reset_cnt++;

            CSPG_ADAPT_DBG_PRINT
                (">>> ra_lnotify_progress: recv from %d, %s%s, recvd=%d, done=%d/%d\n",
                 stat.MPI_SOURCE, (reset_pkt.type == CSP_RA_LNOTIFY_RESET ? "(reset)" : ""),
                 (reset_pkt.type == CSP_RA_LNOTIFY_END ? "(end)" : ""), recvd_reset_cnt,
                 done_reset_cnt, num_local_stats);
        }
    }

    /* always need post a receive till all users are done. */
    if (reset_req == MPI_REQUEST_NULL && done_reset_cnt < num_local_stats) {
        mpi_errno = PMPI_Irecv(&reset_pkt, sizeof(CSP_ra_reset_lnotify_pkt_t), MPI_CHAR,
                               MPI_ANY_SOURCE, CSP_RA_LNOTIFY_RESET_TAG, CSPG_RA_LNTF_COMM,
                               &reset_req);
    }

    return mpi_errno;
}

static int ra_lnotify_complete(void)
{
    int mpi_errno = MPI_SUCCESS;

    /* finish reset notify */
    while (done_reset_cnt < num_local_stats) {
        mpi_errno = ra_lnotify_progress();
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;
    }

    /* finish the previous broadcast */
    mpi_errno = PMPI_Wait(&dirty_req, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;
    CSPG_ADAPT_DBG_PRINT(">>> ra_lnotify_complete: previous ibcast done\n");

    /* send last broadcast for dirty flag */
    dirty_pkt.type = CSP_RA_LNOTIFY_END;
    mpi_errno = PMPI_Ibcast(&dirty_pkt, sizeof(CSP_ra_dirty_lnotify_pkt_t),
                            MPI_CHAR, RA_LNTF_GHOST_RANK, CSPG_RA_LNTF_COMM, &dirty_req);
    CSPG_ADAPT_DBG_PRINT(">>> ra_lnotify_complete: issue last ibcast\n");

    mpi_errno = PMPI_Wait(&dirty_req, MPI_STATUS_IGNORE);
    CSPG_ADAPT_DBG_PRINT(">>> ra_lnotify_complete: done\n");

    return mpi_errno;
}

static inline int ra_lnotify_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    mpi_errno = PMPI_Irecv(&reset_pkt, sizeof(CSP_ra_reset_lnotify_pkt_t), MPI_CHAR,
                           MPI_ANY_SOURCE, CSP_RA_LNOTIFY_RESET_TAG, CSPG_RA_LNTF_COMM, &reset_req);
    return mpi_errno;
}

static inline int ra_gsync_ready(void)
{
    return (gsync_req == MPI_REQUEST_NULL);
}

/* Issue global synchronization among ghost roots. */
static int ra_gsync_issue(void)
{
    int mpi_errno = MPI_SUCCESS;
    int my_stats_disp = 0;
    int i, gsync_rank = 0, gsync_nprocs = 0;

    PMPI_Comm_rank(CSPG_RA_GSYNC_COMM, &gsync_rank);
    PMPI_Comm_size(CSPG_RA_GSYNC_COMM, &gsync_nprocs);
    my_stats_disp = stats_disps[gsync_rank];

    memset(gsync_user_stats, 0, num_all_stats * sizeof(int));

    /* read state */
    mpi_errno = PMPI_Win_lock(MPI_LOCK_SHARED, RA_LNTF_GHOST_RANK, 0, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    for (i = 0; i < num_local_stats; i++) {
        int user_rank = gsync_user_ranks[my_stats_disp + i];
        gsync_user_stats[my_stats_disp + i] = shm_global_stats_ptr[user_rank];
    }

    mpi_errno = PMPI_Win_unlock(RA_LNTF_GHOST_RANK, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* exchange */
    mpi_errno = ra_gsync_iallgatherv();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    issued_gsync_cnt++;
    CSPG_ADAPT_DBG_PRINT(">>> ra_gsyc_issue, issued_gsync_cnt=%d\n", issued_gsync_cnt);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Make progress on the current outstanding global synchronization.*/
static int ra_gsync_progress(void)
{
    int mpi_errno = MPI_SUCCESS;
    int test_flag = 0;
    int i, gsync_rank = 0, gsync_nprocs = 0;

    PMPI_Comm_rank(CSPG_RA_GSYNC_COMM, &gsync_rank);
    PMPI_Comm_size(CSPG_RA_GSYNC_COMM, &gsync_nprocs);

    if (gsync_req != MPI_REQUEST_NULL) {
        mpi_errno = PMPI_Test(&gsync_req, &test_flag, MPI_STATUS_IGNORE);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;

        if (test_flag) {
            /* write cache of local status */
            mpi_errno = PMPI_Win_lock(MPI_LOCK_EXCLUSIVE, CSP_RA_GSYNC_GHOST_LOCAL_RANK,
                                      0, shm_global_stats_region.win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;

            for (i = 0; i < gsync_nprocs; i++) {
                int disp = stats_disps[i], j;
                for (j = 0; j < stats_cnts[i]; j++) {
                    int user_rank = gsync_user_ranks[disp + j];
                    shm_global_stats_ptr[user_rank] = gsync_user_stats[disp + j];
                }
            }

            mpi_errno = PMPI_Win_unlock(CSP_RA_GSYNC_GHOST_LOCAL_RANK, shm_global_stats_region.win);
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
                CSPG_ADAPT_DBG_PRINT(">>> ra_gsyc_progress: allgather done, on %d; off %d\n",
                                     async_on_cnt, async_off_cnt);
            }
#endif

            mpi_errno = ra_lnotify_issue_dirty();
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int ra_gsync_complete(void)
{
    int mpi_errno = MPI_SUCCESS;
    int gsync_nprocs = 0, gsync_rank = 0;
    int max_issued_gsync_cnt = 0;

    PMPI_Comm_size(CSPG_RA_GSYNC_COMM, &gsync_nprocs);
    PMPI_Comm_rank(CSPG_RA_GSYNC_COMM, &gsync_rank);

    /* ensure all ghosts are arrived and gather gsync count */
    mpi_errno = PMPI_Allreduce(&issued_gsync_cnt, &max_issued_gsync_cnt, 1,
                               MPI_INT, MPI_MAX, CSPG_RA_GSYNC_COMM);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;
    CSPG_ADAPT_DBG_PRINT(">>> >> ra_gsync_complete: issued_gsync_cnt %d, max %d\n",
                         issued_gsync_cnt, max_issued_gsync_cnt);

    /* there must be at most 1 missing allgather, since have to receive data from every one. */
    CSP_assert((max_issued_gsync_cnt == issued_gsync_cnt + 1) ||
               (max_issued_gsync_cnt == issued_gsync_cnt));

    if (max_issued_gsync_cnt == issued_gsync_cnt + 1) {
        /* finish previous allgather */
        mpi_errno = PMPI_Wait(&gsync_req, MPI_STATUS_IGNORE);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;
        CSPG_ADAPT_DBG_PRINT(">>> >> ra_gsync_complete: previous allgather done\n");

        /* issue last allgather */
        mpi_errno = ra_gsync_iallgatherv();
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
        CSPG_ADAPT_DBG_PRINT(">>> >> ra_gsync_complete: issue last allgather\n");
    }

    /* wait on the last allgather */
    mpi_errno = PMPI_Wait(&gsync_req, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;
    CSPG_ADAPT_DBG_PRINT(">>> ra_gsync_complete: done\n");

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static CSP_async_stat ra_get_init_async_stat(void)
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

static int ra_gsync_init()
{
    int mpi_errno = MPI_SUCCESS;
    int i = 0, idx = 0;
    int my_stats_disp = 0;
    int gsync_rank = 0, gsync_nprocs = 0;
    int lnft_nprocs = 0;
    MPI_Request *reqs = NULL;

    PMPI_Comm_rank(CSPG_RA_GSYNC_COMM, &gsync_rank);
    PMPI_Comm_size(CSPG_RA_GSYNC_COMM, &gsync_nprocs);
    PMPI_Comm_size(CSPG_RA_LNTF_COMM, &lnft_nprocs);

    /* initialize global synchronization parameters */
    gsync_user_ranks = CSP_calloc(num_all_stats, sizeof(int));
    gsync_user_stats = CSP_calloc(num_all_stats, sizeof(int));
    stats_cnts = CSP_calloc(gsync_nprocs, sizeof(int));
    stats_disps = CSP_calloc(gsync_nprocs, sizeof(int));
    reqs = CSP_calloc(lnft_nprocs - 1, sizeof(MPI_Request));

    /* gather number of states on every ghost */
    stats_cnts[gsync_rank] = num_local_stats;
    mpi_errno = PMPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                               stats_cnts, 1, MPI_INT, CSPG_RA_GSYNC_COMM);
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
    CSP_ADAPT_DBG_PRINT(" ra_init: gather num_local_stats, symmetric=%d, node cnt/disps:\n",
                        symmetric_num_stats);
    for (i = 0; i < gsync_nprocs; i++)
        CSPG_ADAPT_DBG_PRINT(" \t[%d] = %d/%d\n", i, stats_cnts[i], stats_disps[i]);
#endif

    /* gather local user ranks in comm_user_world. */
    idx = 0;
    for (i = 0; i < lnft_nprocs; i++) {
        if (i == RA_LNTF_GHOST_RANK)
            continue;
        mpi_errno = PMPI_Irecv(&gsync_user_ranks[my_stats_disp + idx], 1, MPI_INT,
                               i, 0, CSPG_RA_LNTF_COMM, &reqs[idx]);
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
                                   num_local_stats, MPI_INT, CSPG_RA_GSYNC_COMM);
    }
    else {
        mpi_errno = PMPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                    gsync_user_ranks, stats_cnts, stats_disps,
                                    MPI_INT, CSPG_RA_GSYNC_COMM);
    }
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    CSP_ADAPT_DBG_PRINT(" ra_init: gather users ranks in user world:\n");
#if defined(CSPG_DEBUG) || defined(CSPG_ADAPT_DEBUG)
    for (i = 0; i < gsync_nprocs; i++) {
        int disp = stats_disps[i], j;
        CSPG_ADAPT_DBG_PRINT(" \t ghost[%d]\n", i);
        for (j = 0; j < stats_cnts[i]; j++) {
            CSPG_ADAPT_DBG_PRINT(" \t\t user[%d] = %d\n", j, gsync_user_ranks[disp + j]);
        }
    }
#endif

    ra_gsync_interval_sta = PMPI_Wtime();

  fn_exit:
    if (reqs)
        free(reqs);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int ra_comm_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Group local_ug_group = MPI_GROUP_NULL;
    int i = 0, idx = 0, *excl_ranks = NULL;
    int local_rank = 0;

    PMPI_Comm_rank(CSP_COMM_LOCAL, &local_rank);

    /* note that all ghosts arrive here for communicator creation. */

    /* create a gsync ghost communicator */
    mpi_errno = PMPI_Comm_split(MPI_COMM_WORLD, local_rank == CSP_RA_GSYNC_GHOST_LOCAL_RANK,
                                1, &CSPG_RA_GSYNC_COMM);

    if (local_rank == CSP_RA_GSYNC_GHOST_LOCAL_RANK) {
        int gsync_rank = 0, gsync_nprocs = 0;
        PMPI_Comm_rank(CSPG_RA_GSYNC_COMM, &gsync_rank);
        PMPI_Comm_size(CSPG_RA_GSYNC_COMM, &gsync_nprocs);
        CSPG_ADAPT_DBG_PRINT(" ra_init: create gsync_comm, I am %d/%d\n", gsync_rank, gsync_nprocs);
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

    mpi_errno = PMPI_Comm_create_group(CSP_COMM_LOCAL, local_ug_group, 0, &CSPG_RA_LNTF_COMM);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    if (local_rank == CSP_RA_GSYNC_GHOST_LOCAL_RANK) {
        int ra_lntf_rank = 0, ra_lntf_nprocs = 0;
        PMPI_Comm_rank(CSPG_RA_LNTF_COMM, &ra_lntf_rank);
        PMPI_Comm_size(CSPG_RA_LNTF_COMM, &ra_lntf_nprocs);
        CSPG_ADAPT_DBG_PRINT(" ra_init: create ra_lntf_comm, I am %d/%d, ghost %d\n", ra_lntf_rank,
                             ra_lntf_nprocs, RA_LNTF_GHOST_RANK);
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

void CSPG_ra_finalize(void)
{
    int local_rank = 0;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        return;

    PMPI_Comm_rank(CSP_COMM_LOCAL, &local_rank);
    if (local_rank != CSP_RA_GSYNC_GHOST_LOCAL_RANK)
        return;

    /* complete all local notification and global synchronization */
    ra_lnotify_complete();
    ra_gsync_complete();

    if (shm_global_stats_region.win && shm_global_stats_region.win != MPI_WIN_NULL) {
        PMPI_Win_free(&shm_global_stats_region.win);
    }

    shm_global_stats_region.win = MPI_WIN_NULL;
    shm_global_stats_region.base = NULL;

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
    if (CSPG_RA_LNTF_COMM && CSPG_RA_LNTF_COMM != MPI_COMM_NULL) {
        PMPI_Comm_free(&CSPG_RA_LNTF_COMM);
        CSPG_RA_LNTF_COMM = MPI_COMM_NULL;
    }
    if (CSPG_RA_GSYNC_COMM && CSPG_RA_GSYNC_COMM != MPI_COMM_NULL) {
        PMPI_Comm_free(&CSPG_RA_GSYNC_COMM);
        CSPG_RA_GSYNC_COMM = MPI_COMM_NULL;
    }
}

int CSPG_ra_init(void)
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
    mpi_errno = ra_comm_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    if (local_rank != CSP_RA_GSYNC_GHOST_LOCAL_RANK)
        goto fn_exit;

    /* allocate shared region among local node for storing all processes' statuses */
    region_size = sizeof(int) * num_all_stats;
    shm_global_stats_region.win = MPI_WIN_NULL;
    shm_global_stats_region.base = NULL;
    shm_global_stats_region.size = region_size;

    /* only the global synchronizing ghost allocates shared region. */
    mpi_errno = PMPI_Win_allocate_shared(region_size, 1, MPI_INFO_NULL,
                                         CSPG_RA_LNTF_COMM, &shm_global_stats_region.base,
                                         &shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    CSPG_ADAPT_DBG_PRINT
        (" ra_init: allocated shm_reg %p, size %ld, num_local_stats %d, num_all_stats %d\n",
         shm_global_stats_region.base, region_size, num_local_stats, num_all_stats);

    /* initialize asynchronous state in cache */
    shm_global_stats_ptr = shm_global_stats_region.base;
    init_async_stat = ra_get_init_async_stat();

    mpi_errno = PMPI_Win_lock(MPI_LOCK_SHARED, RA_LNTF_GHOST_RANK, 0, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    for (i = 0; i < num_local_stats; i++)
        shm_global_stats_ptr[i] = init_async_stat;

    mpi_errno = PMPI_Win_unlock(RA_LNTF_GHOST_RANK, shm_global_stats_region.win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = ra_lnotify_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* initialize global synchronization */
    mpi_errno = ra_gsync_init();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

  fn_exit:
    return mpi_errno;
  fn_fail:
    CSPG_ra_finalize();
    goto fn_exit;
}

/* Globally synchronize the asynchronous status of user processes.
 * If the status of local user processes has been updated (by user itself),
 * it globally broadcasts to all other gsync ghost processes; Meanwhile, it
 * receives messages from any gsync ghost process, and then update its local cache.
 * Note that this function should only be called by the single gsync ghost process
 * on every node. */
int CSPG_ra_gsync(void)
{
    int mpi_errno = MPI_SUCCESS;
    int local_rank = 0;
    double now = 0.0;

    PMPI_Comm_rank(CSP_COMM_LOCAL, &local_rank);
    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME || local_rank > 0)
        goto fn_exit;

    ra_gsync_progress();
    ra_lnotify_progress();

    /* only issue synchronization if interval is large enough. */
    now = PMPI_Wtime();
    if ((now - ra_gsync_interval_sta - CSP_ENV.async_timed_gsync_int) < 0)
        goto fn_exit;

    /* if not ready for next issue, return */
    if (!ra_gsync_ready())
        goto fn_exit;

    /* TODO: add update rate check. */

    mpi_errno = ra_gsync_issue();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    mpi_errno = ra_gsync_progress();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    ra_gsync_interval_sta = now;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#endif
