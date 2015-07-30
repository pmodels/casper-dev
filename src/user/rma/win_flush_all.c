/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"
#include "csp_rma_local.h"

static int CSP_win_mixed_flush_all_impl(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int user_rank, user_nprocs;
    int i;

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);

    /* Flush all Ghosts in corresponding ug-window of each target process.. */
#ifdef CSP_ENABLE_SYNC_ALL_OPT

    /* Optimization for MPI implementations that have optimized lock_all.
     * However, user should be noted that, if MPI implementation issues lock messages
     * for every target even if it does not have any operation, this optimization
     * could lose performance and even lose asynchronous! */
    for (i = 0; i < ug_win->num_ug_wins; i++) {
        CSP_DBG_PRINT("[%d]flush_all(ug_win 0x%x)\n", user_rank, ug_win->ug_wins[i]);
        mpi_errno = PMPI_Win_flush_all(ug_win->ug_wins[i]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
#else

    /* TODO: track op issuing, only flush the ghosts which receive ops. */
    for (i = 0; i < user_nprocs; i++) {
#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
        /* flush targets which are in async-off state. */
        if (ug_win->targets[i].async_stat == CSP_TARGET_ASYNC_OFF) {
            CSP_DBG_PRINT("[%d]flush(target(%d), ug_wins 0x%x), instead of "
                          "target rank %d\n", user_rank, ug_win->targets[i].ug_rank,
                          ug_win->targets[i].ug_win, i);
            mpi_errno = PMPI_Win_flush(ug_win->targets[i].ug_rank, ug_win->targets[i].ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
            continue;
        }
#endif

#if !defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
        int j;
        /* RMA operations are only issued to the main ghost, so we only flush it. */
        for (j = 0; j < ug_win->targets[i].num_segs; j++) {
            int main_g_off = ug_win->targets[i].segs[j].main_g_off;
            int target_g_rank_in_ug = ug_win->targets[i].g_ranks_in_ug[main_g_off];
            CSP_DBG_PRINT("[%d]flush(Ghost(%d), ug_wins 0x%x), instead of "
                          "target rank %d seg %d\n", user_rank, target_g_rank_in_ug,
                          ug_win->targets[i].segs[j].ug_win, i, j);

            mpi_errno = PMPI_Win_flush(target_g_rank_in_ug, ug_win->targets[i].segs[j].ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
#else
        int k;

        /* RMA operations may be distributed to all ghosts, so we should
         * flush all ghosts on all windows. See discussion in win_flush. */
        for (k = 0; k < CSP_ENV.num_g; k++) {
            int target_g_rank_in_ug = ug_win->targets[i].g_ranks_in_ug[k];
            CSP_DBG_PRINT("[%d]flush(Ghost(%d), ug_win 0x%x), instead of "
                          "target rank %d\n", user_rank, target_g_rank_in_ug,
                          ug_win->targets[i].ug_win, i);

            mpi_errno = PMPI_Win_flush(target_g_rank_in_ug, ug_win->targets[i].ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
#endif /*end of CSP_ENABLE_RUNTIME_LOAD_OPT */
    }
#endif /*end of CSP_ENABLE_SYNC_ALL_OPT */

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPI_Win_flush_all(MPI_Win win)
{
    CSP_win *ug_win;
    int mpi_errno = MPI_SUCCESS;
    int user_rank, user_nprocs;
    int i;

    CSP_DBG_PRINT_FCNAME();
    CSP_rm_count(CSP_RM_COMM_FREQ);

    CSP_fetch_ug_win_from_cache(win, ug_win);

    if (ug_win == NULL) {
        /* normal window */
        return PMPI_Win_flush_all(win);
    }

    /* casper window starts */

    CSP_assert((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) ||
               (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK_ALL));

#ifdef CSP_ENABLE_EPOCH_STAT_CHECK
    /* Check access epoch status.
     * The current epoch must be lock_all.*/
    if (ug_win->epoch_stat != CSP_WIN_EPOCH_LOCK_ALL) {
        CSP_ERR_PRINT("Wrong synchronization call! "
                      "No opening LOCK_ALL epoch in %s\n", __FUNCTION__);
        mpi_errno = -1;
        goto fn_fail;
    }
    CSP_assert(ug_win->start_counter == 0 && ug_win->lock_counter == 0);
#endif

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);

    if (!(ug_win->info_args.epoch_type & CSP_EPOCH_LOCK)) {
        /* In lock_all only epoch, single window is shared by multiple targets. */

#ifdef CSP_ENABLE_SYNC_ALL_OPT

        /* Optimization for MPI implementations that have optimized lock_all.
         * However, user should be noted that, if MPI implementation issues lock messages
         * for every target even if it does not have any operation, this optimization
         * could lose performance and even lose asynchronous! */
        CSP_DBG_PRINT("[%d]flush_all(ug_win 0x%x)\n", user_rank, ug_win->ug_wins[0]);
        mpi_errno = PMPI_Win_flush_all(ug_win->ug_wins[0]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
#else
        /* Flush every ghost once in the single window.
         * TODO: track op issuing, only flush the ghosts which receive ops. */
        for (i = 0; i < ug_win->num_g_ranks_in_ug; i++) {
            mpi_errno = PMPI_Win_flush(ug_win->g_ranks_in_ug[i], ug_win->ug_wins[0]);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
        if (ug_win->info_args.async_config == CSP_ASYNC_CONFIG_AUTO) {
            /* flush targets which are in async-off state.
             * Note that, for all-async-off case, RMA goes through normal window. */
            for (i = 0; i < user_nprocs; i++) {
                if (ug_win->targets[i].async_stat == CSP_TARGET_ASYNC_OFF) {
                    CSP_DBG_PRINT("[%d]flush(target(%d), ug_wins 0x%x), instead of "
                                  "target rank %d\n", user_rank, ug_win->targets[i].ug_rank,
                                  ug_win->ug_wins[0], i);
                    mpi_errno = PMPI_Win_flush(ug_win->targets[i].ug_rank, ug_win->ug_wins[0]);
                    if (mpi_errno != MPI_SUCCESS)
                        goto fn_fail;
                }
            }
        }
#endif
#endif /* end of CSP_ENABLE_SYNC_ALL_OPT */
    }
    else {

        /* In lock_all/lock mixed epoch, separate windows are bound with each target. */
        mpi_errno = CSP_win_mixed_flush_all_impl(ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

#ifdef CSP_ENABLE_LOCAL_LOCK_OPT
    /* If LOCAL_LOCK_OPT is enabled, PUT/GET may be issued to local
     * target. Thus we need flush the local target as well.
     * Note that ACC operations are always issued to main ghost,
     * since atomicity and ordering issue. */
    mpi_errno = CSP_win_flush_self_impl(ug_win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;
#endif

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    int j;
    for (i = 0; i < user_nprocs; i++) {
        for (j = 0; j < ug_win->targets[i].num_segs; j++) {
            /* Lock of main ghost is granted, we can start load balancing from the next flush/unlock.
             * Note that only target which was issued operations to is guaranteed to be granted. */
            if (ug_win->targets[i].segs[j].main_lock_stat == CSP_MAIN_LOCK_OP_ISSUED) {
                ug_win->targets[i].segs[j].main_lock_stat = CSP_MAIN_LOCK_GRANTED;
                CSP_DBG_PRINT("[%d] main lock (rank %d, seg %d) granted\n", user_rank, i, j);
            }

            CSP_reset_target_opload(i, ug_win);
        }
    }
#endif

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
