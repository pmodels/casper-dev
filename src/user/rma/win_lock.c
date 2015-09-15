/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"
#include "csp_rma_local.h"

int MPI_Win_lock(int lock_type, int target_rank, int assert, MPI_Win win)
{
    CSP_win *ug_win;
    CSP_win_target *target;
    int mpi_errno = MPI_SUCCESS;
    int user_rank;
    int is_g_locked = 0;

    CSP_DBG_PRINT_FCNAME();
    CSP_MPI_FUNC_START_ROUTINE();

    CSP_fetch_ug_win_from_cache(win, ug_win);

    if (ug_win == NULL) {
        /* normal window */
        return PMPI_Win_lock(lock_type, target_rank, assert, win);
    }

    /* casper window starts */

    if (target_rank == MPI_PROC_NULL)
        goto fn_exit;

    CSP_assert((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK));

    target = &(ug_win->targets[target_rank]);

#ifdef CSP_ENABLE_EPOCH_STAT_CHECK
    /* Check access epoch status.
     * We do not require closed FENCE epoch, because we don't know whether
     * the previous FENCE is closed or not.*/
    if (ug_win->epoch_stat == CSP_WIN_EPOCH_LOCK_ALL) {
        CSP_ERR_PRINT("Wrong synchronization call! "
                      "Previous LOCK_ALL epoch is still open in %s\n", __FUNCTION__);
        mpi_errno = -1;
        goto fn_fail;
    }

    /* Check per-target access epoch status. */
    if (ug_win->epoch_stat == CSP_WIN_EPOCH_PER_TARGET && target->epoch_stat != CSP_TARGET_NO_EPOCH) {
        CSP_ERR_PRINT("Wrong synchronization call! "
                      "Previous %s epoch on target %d is still open in %s\n",
                      target->epoch_stat == CSP_TARGET_EPOCH_LOCK ? "LOCK" : "PSCW",
                      target_rank, __FUNCTION__);
        mpi_errno = -1;
        goto fn_fail;
    }
#endif

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    target->remote_lock_assert = assert;
    target->remote_lock_type = lock_type;
    CSP_DBG_PRINT("[%d]lock(%d), MPI_MODE_NOCHECK %d(assert %d)\n", user_rank,
                  target_rank, (assert & MPI_MODE_NOCHECK) != 0, assert);

    /* Lock Ghost processes in corresponding ug-window of target process. */
#ifdef CSP_ENABLE_SYNC_ALL_OPT
    /* lock_all cannot handle exclusive locks, thus should use only for shared lock or nocheck. */
    if (target->remote_lock_type == MPI_LOCK_SHARED ||
        target->remote_lock_assert & MPI_MODE_NOCHECK) {
        /* Optimization for MPI implementations that have optimized lock_all.
         * However, user should be noted that, if MPI implementation issues lock messages
         * for every target even if it does not have any operation, this optimization
         * could lose performance and even lose asynchronous! */

        CSP_DBG_PRINT("[%d]lock_all(ug_win 0x%x), instead of target rank %d\n",
                      user_rank, target->ug_win, target_rank);
        mpi_errno = PMPI_Win_lock_all(assert, target->ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        is_g_locked = 1;
        if (user_rank == target_rank)
            ug_win->is_self_locked = 1; /* set flag to avoid twice lock. */
    }
    else
#endif /*end of CSP_ENABLE_SYNC_ALL_OPT */
    {
        if (target->synced_async_stat == CSP_ASYNC_ON ||
            CSP_ENV.async_sched_level == CSP_ASYNC_SCHED_ANYTIME) {
            /* lock all ghosts. */
            mpi_errno = CSP_win_target_lock_ghosts(lock_type, assert, target_rank, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;

            is_g_locked = 1;    /* force lock only when ghosts are locked */
        }
        if (target->synced_async_stat == CSP_ASYNC_OFF ||
            CSP_ENV.async_sched_level == CSP_ASYNC_SCHED_ANYTIME) {
            /* When async is off on that target, we only lock the target process.
             * Static per-coll scheduling must be collective, thus is invalid during a lock epoch. */
            mpi_errno = CSP_win_target_lock_user(lock_type, assert, target_rank, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
    }

    if (user_rank == target_rank) {
        /* If target is itself, we need grant this lock before return.
         * However, the actual locked processes are the Ghosts whose locks may be delayed by
         * most MPI implementation, thus we need a flush to force the lock to be granted.
         *
         * For performance reason, this operation is ignored if meet at least one of following conditions:
         * 1. if user passed information that this process will not do local load/store on this window.
         * 2. if user passed information that there is no concurrent epochs.
         */
        if (is_g_locked && !ug_win->info_args.no_local_load_store &&
            !(target->remote_lock_assert & MPI_MODE_NOCHECK)) {
            mpi_errno = CSP_win_grant_local_lock(user_rank, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }

        /* Lock local rank for memory consistency on local load/store operations.
         * If user passed no_local_load_store, this step can be skipped.*/
        if (!ug_win->info_args.no_local_load_store && !ug_win->is_self_locked /* already locked */) {
            mpi_errno = CSP_win_lock_self_impl(ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
    }

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    int j;
    for (j = 0; j < target->num_segs; j++) {
        target->segs[j].main_lock_stat = CSP_MAIN_LOCK_RESET;
        CSP_reset_target_opload(target_rank, ug_win);
    }
#endif

    /* Indicate epoch status.
     * later operations issued to the target will be redirected to ug_wins.*/
    target->epoch_stat = CSP_TARGET_EPOCH_LOCK;
    ug_win->epoch_stat = CSP_WIN_EPOCH_PER_TARGET;
    ug_win->lock_counter++;

  fn_exit:
    CSP_MPI_FUNC_END_ROUTINE();
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
