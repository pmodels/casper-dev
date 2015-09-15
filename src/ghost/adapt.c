/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "cspg.h"

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED

static int *local_u_ranks_in_user_world = NULL;
static CSP_local_shm_region shm_global_stats_region;    /* asynchronous status of all user processes.
                                                         * one copy per node. */
static CSP_async_stat *shm_global_stats_ptr = NULL;     /* point to shm region, do not free */
static CSP_async_stat *prev_local_stats = NULL; /* previous status for each local user. */
static int num_local_stats = 0;

static MPI_Comm CSPG_RA_GSYNC_COMM = MPI_COMM_NULL;     /* all gsync ghost processes. */

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

static CSP_sbcast_root_req ra_sbr_req;
static CSP_sbcast_member_req ra_sbm_req;

static CSPG_ra_gsync_pkt ra_sbr_pkt = { -1, CSP_ASYNC_NONE };
static CSPG_ra_gsync_pkt ra_sbm_pkt = { -1, CSP_ASYNC_NONE };

void CSPG_ra_finalize(void)
{
    int local_rank = 0;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        return;

    PMPI_Comm_rank(CSP_COMM_LOCAL, &local_rank);

    if (shm_global_stats_region.win && shm_global_stats_region.win != MPI_WIN_NULL) {
        if (local_rank == CSP_RA_GSYNC_GHOST_LOCAL_RANK)
            PMPI_Win_unlock(CSP_RA_GSYNC_GHOST_LOCAL_RANK, shm_global_stats_region.win);
        PMPI_Win_free(&shm_global_stats_region.win);
    }

    shm_global_stats_region.win = MPI_WIN_NULL;
    shm_global_stats_region.base = NULL;

    if (local_rank != CSP_RA_GSYNC_GHOST_LOCAL_RANK)
        return;

    /* only gsync ghost needs to release following resources. */

    if (local_u_ranks_in_user_world) {
        free(local_u_ranks_in_user_world);
        local_u_ranks_in_user_world = NULL;
    }

    if (prev_local_stats) {
        free(prev_local_stats);
        prev_local_stats = NULL;
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
    int gsync_rank = 0;
    MPI_Aint region_size = 0;
    MPI_Request *reqs = NULL;
    int i;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    PMPI_Comm_rank(CSP_COMM_LOCAL, &local_rank);
    PMPI_Comm_size(CSP_COMM_LOCAL, &local_nprocs);
    PMPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    num_local_stats = local_nprocs - CSP_ENV.num_g;     /* number of processes per node can be different,
                                                         * but number of ghosts per node is fixed.*/

    /* allocate shared region among local node for storing all processes' statuses */
    region_size = sizeof(CSP_async_stat) * (nprocs - CSP_ENV.num_g * CSP_NUM_NODES);
    shm_global_stats_region.win = MPI_WIN_NULL;
    shm_global_stats_region.base = NULL;
    shm_global_stats_region.size = region_size;

    if (local_rank != CSP_RA_GSYNC_GHOST_LOCAL_RANK) {
        /* other local ghosts just join window allocation and then return. */
        return PMPI_Win_allocate_shared(0, 1, MPI_INFO_NULL,
                                        CSP_COMM_LOCAL, &shm_global_stats_region.base,
                                        &shm_global_stats_region.win);
    }
    else {
        /* only the global synchronizing ghost allocates shared region. */
        mpi_errno = PMPI_Win_allocate_shared(region_size, 1, MPI_INFO_NULL,
                                             CSP_COMM_LOCAL, &shm_global_stats_region.base,
                                             &shm_global_stats_region.win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        /* do not need exclusive access, only for later load/store. */
        mpi_errno = PMPI_Win_lock(MPI_LOCK_SHARED, CSP_RA_GSYNC_GHOST_LOCAL_RANK,
                                  MPI_MODE_NOCHECK, shm_global_stats_region.win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    CSPG_DBG_PRINT(" ra_init: allocated shm_reg %p, size %ld\n",
                   shm_global_stats_region.base, region_size);

    shm_global_stats_ptr = shm_global_stats_region.base;
    prev_local_stats = CSP_calloc(num_local_stats, sizeof(CSP_async_stat));
    for (i = 0; i < num_local_stats; i++)
        prev_local_stats[i] = CSP_ASYNC_NONE;

    /* gsync ghost receives local user processes' rank in comm_user_world. */
    local_u_ranks_in_user_world = CSP_calloc(num_local_stats, sizeof(int));
    reqs = CSP_calloc(num_local_stats, sizeof(MPI_Request));
    for (i = 0; i < num_local_stats; i++) {
        mpi_errno = PMPI_Irecv(&local_u_ranks_in_user_world[i], 1, MPI_INT,
                               i + CSP_ENV.num_g, 0, CSP_COMM_LOCAL, &reqs[i]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    mpi_errno = PMPI_Waitall(num_local_stats, reqs, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    CSPG_DBG_PRINT(" ra_init: local users in user world:\n");
    for (i = 0; i < num_local_stats; i++)
        CSPG_DBG_PRINT(" \t[%d] = %d\n", i, local_u_ranks_in_user_world[i]);

    /* create a gsync ghost communicator */
    mpi_errno = PMPI_Comm_split(MPI_COMM_WORLD, local_rank == CSP_RA_GSYNC_GHOST_LOCAL_RANK,
                                1, &CSPG_RA_GSYNC_COMM);
    PMPI_Comm_rank(CSPG_RA_GSYNC_COMM, &gsync_rank);

    /* Initialize global requests */
    CSP_sbcast_root_req_init(&ra_sbr_req);
    CSP_sbcast_member_req_init(&ra_sbm_req);

    /* Post the first member request */
    ra_gsync_progress();

  fn_exit:
    if (reqs)
        free(reqs);
    return mpi_errno;

  fn_fail:
    CSPG_ra_finalize();
    goto fn_exit;
}

/* Send a <user_rank, stat> packet to the internal broadcast agent.
 * The new packet will be sent only when no outstanding send request,
 * otherwise this packet will be dropped. The caller can check whether the
 * packet has been sent using flag (1:sent|0:dropped). */
static int ra_gsync_issue(int user_rank, CSP_async_stat stat, int *flag)
{
    int mpi_errno = MPI_SUCCESS;
    int test_flag = 0;

    (*flag) = 0;

    /* test on the outstanding issuing request */
    if (!CSP_sbcast_root_is_completed(ra_sbr_req)) {
        mpi_errno = CSP_sbcast_root_test(&ra_sbr_req, &test_flag);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    /* issue the next one only when no outstanding request. */
    if (CSP_sbcast_root_is_completed(ra_sbr_req)) {
        ra_sbr_pkt.user_rank = user_rank;
        ra_sbr_pkt.async_stat = stat;

        mpi_errno = CSP_sbcast_root(&ra_sbr_pkt, sizeof(CSPG_ra_gsync_pkt),
                                    CSPG_RA_GSYNC_COMM, &ra_sbr_req);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
        CSP_DBG_PRINT(" ra_gsyc_issue: sent user %d, stat %d\n",
                      ra_sbr_pkt.user_rank, ra_sbr_pkt.async_stat);

        (*flag) = 1;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Make progress on the current outstanding synchronization.
 * If no member request is active, then it posts a new one; if there is
 * outstanding member request, then test its completion, and update local
 * status cache once it is completed.*/
static int ra_gsync_progress(void)
{
    int mpi_errno = MPI_SUCCESS;
    int test_flag = 0;

    /* post a new member call if no outstanding request. */
    if (CSP_sbcast_member_is_completed(ra_sbm_req)) {
        mpi_errno = CSP_sbcast_member(&ra_sbm_pkt, sizeof(CSPG_ra_gsync_pkt),
                                      CSPG_RA_GSYNC_COMM, &ra_sbm_req);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    /* test on the current outstanding request */
    mpi_errno = CSP_sbcast_member_test(&ra_sbm_req, &test_flag);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    if (test_flag) {
        /* update cache of local status */
        shm_global_stats_ptr[ra_sbm_pkt.user_rank] = ra_sbm_pkt.async_stat;
        CSP_DBG_PRINT(" ra_gsync_progress: received: user %d, stat %d\n",
                      ra_sbm_pkt.user_rank, ra_sbm_pkt.async_stat);

        /* reset temp buffer */
        ra_sbm_pkt.user_rank = -1;
        ra_sbm_pkt.async_stat = CSP_ASYNC_NONE;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int nxt_idx = 0;

/* Globally synchronize the asynchronous status of user processes.
 * If the status of local user processes has been updated (by user itself),
 * it globally broadcasts to all other gsync ghost processes; Meanwhile, it
 * receives messages from any gsync ghost process, and then update its local cache.
 * Note that this function should only be called by the single gsync ghost process
 * on every node. */
int CSPG_ra_gsync(void)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_async_stat *local_async_stats = NULL;
    int local_rank = 0, gsync_rank = 0;
    int i;

    PMPI_Comm_rank(CSP_COMM_LOCAL, &local_rank);
    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME || local_rank > 0)
        goto fn_exit;

    ra_gsync_progress();

    for (i = nxt_idx; i != nxt_idx; i = (i + 1) % num_local_stats) {
        int user_rank = local_u_ranks_in_user_world[i];
        int sent_flag = 0;

        /* issue message if a user has updated its status */
        if (shm_global_stats_ptr[user_rank] != prev_local_stats[i]) {
            mpi_errno = ra_gsync_issue(user_rank, shm_global_stats_ptr[user_rank], &sent_flag);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;

            mpi_errno = ra_gsync_progress();
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;

            if (sent_flag) {
                prev_local_stats[i] = shm_global_stats_ptr[user_rank];
                nxt_idx = (i + 1) % num_local_stats;    /* start from the next one */
                break;
            }
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#endif
