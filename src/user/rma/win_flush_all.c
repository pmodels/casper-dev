/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"
#include "csp_rma_local.h"

/* Flushes ghost process for a given target.
 * It can be called by WIN_FLUSH, WIN_COMPLETE or CSP_win_flush_all which is a
 * common routine for WIN_FLUSH_ALL, WIN_FENCE, and WIN_UNLOCK_ALL in lock-all only mode.  */
int CSP_win_target_flush_ghosts(int target_rank, CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_win_target *target = NULL;
    MPI_Win *win_ptr = NULL;
    int user_rank;
    int j;

    target = &(ug_win->targets[target_rank]);
    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

    CSP_target_get_epoch_win(0, target, ug_win, win_ptr);
    CSP_assert(win_ptr != NULL);

#if !defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    /* RMA operations are only issued to the main ghost, so we only flush it. */
    /* TODO: track op issuing, only flush the ghosts which receive ops. */
    for (j = 0; j < target->num_segs; j++) {
        int main_g_off = target->segs[j].main_g_off;
        int target_g_rank_in_ug = target->g_ranks_in_ug[main_g_off];

        CSP_DBG_PRINT("[%d]flush(Ghost(%d), %s 0x%x), instead of "
                      "target rank %d seg %d\n", user_rank, target_g_rank_in_ug,
                      CSP_get_win_type(*win_ptr, ug_win), *win_ptr, target_rank, j);

        mpi_errno = PMPI_Win_flush(target_g_rank_in_ug, *win_ptr);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
#else
    /* RMA operations may be distributed to all ghosts, so we should
     * flush all ghosts on all windows.
     *
     * Note that some flushes could be eliminated before the main lock of a
     * segment granted (see above). However, we have to loop all the segments
     * in order to check each lock status, and we may flush the same ghost
     * on the same window twice if the lock is granted on that segment.
     * i.e., flush (H0, win0) and (H1, win0) twice for seg0 and seg1.
     *
     * Consider flush does nothing if no operations on that target in most
     * MPI implementation, simpler code is better */
    int k;
    for (k = 0; k < CSP_ENV.num_g; k++) {
        int target_g_rank_in_ug = target->g_ranks_in_ug[k];

        CSP_DBG_PRINT("[%d]flush(Ghost(%d), %s 0x%x), instead of "
                      "target rank %d\n", user_rank, target_g_rank_in_ug,
                      CSP_get_win_type(*win_ptr, ug_win), *win_ptr, target_rank);

        mpi_errno = PMPI_Win_flush(target_g_rank_in_ug, *win_ptr);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
#endif /*end of CSP_ENABLE_RUNTIME_LOAD_OPT */

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

/* Flush all targets in a window.
 * It can be called by WIN_FLUSH_ALL, WIN_FENCE, or WIN_UNLOCK_ALL for lock-all only mode.  */
int CSP_win_flush_all(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int user_rank, user_nprocs;
    int is_self_flushed = 0;
    int i CSP_ATTRIBUTE((unused));

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);

#ifdef CSP_ENABLE_SYNC_ALL_OPT

    /* Optimization for MPI implementations that have optimized lock_all.
     * However, user should be noted that, if MPI implementation issues lock messages
     * for every target even if it does not have any operation, this optimization
     * could lose performance and even lose asynchronous! */
    if (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) {
        for (i = 0; i < ug_win->num_ug_wins; i++) {
            CSP_DBG_PRINT("[%d]flush_all(ug_win 0x%x)\n", user_rank, ug_win->ug_wins[i]);
            mpi_errno = PMPI_Win_flush_all(ug_win->ug_wins[i]);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
    }
    else {
        CSP_DBG_PRINT("[%d]flush_all(active_win 0x%x)\n", user_rank, ug_win->active_win);
        mpi_errno = PMPI_Win_flush_all(ug_win->active_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
    is_self_flushed = 1;        /* avoid twice self flush */
#else

    for (i = 0; i < user_nprocs; i++) {
        if (ug_win->targets[i].synced_async_stat == CSP_ASYNC_ON) {
            /* only flush ghosts if async is on */
            mpi_errno = CSP_win_target_flush_ghosts(i, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
        else {
            /* only flush target if async is off. */
            mpi_errno = CSP_win_target_flush_user(i, ug_win, &is_self_flushed);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
    }

#ifdef CSP_ENABLE_LOCAL_LOCK_OPT
    /* If LOCAL_LOCK_OPT is enabled, PUT/GET may be issued to local
     * target. Thus we need flush the local target as well.
     * Note that ACC operations are always issued to main ghost,
     * since atomicity and ordering issue. */
    if (!is_self_flushed) {
        mpi_errno = CSP_win_flush_self_impl(ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
#endif
#endif /* end of CSP_ENABLE_SYNC_ALL_OPT */

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPI_Win_flush_all(MPI_Win win)
{
    CSP_win *ug_win;
    int mpi_errno = MPI_SUCCESS;

    CSP_DBG_PRINT_FCNAME();
    CSP_rm_count_start(CSP_RM_COMM_FREQ);

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

    mpi_errno = CSP_win_flush_all(ug_win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

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
    CSP_rm_count_end(CSP_RM_COMM_FREQ);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
