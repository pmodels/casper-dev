/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"
#include "csp_rma_local.h"

int MPI_Win_unlock(int target_rank, MPI_Win win)
{
    CSP_win *ug_win;
    CSP_win_target *target;
    int mpi_errno = MPI_SUCCESS;
    int user_rank;

    CSP_DBG_PRINT_FCNAME();
    CSP_rm_count_start(CSP_RM_COMM_FREQ);

    CSP_fetch_ug_win_from_cache(win, ug_win);

    if (ug_win == NULL) {
        /* normal window */
        return PMPI_Win_unlock(target_rank, win);
    }

    /* casper window starts */

    if (target_rank == MPI_PROC_NULL)
        goto fn_exit;

    CSP_assert((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK));

    target = &(ug_win->targets[target_rank]);

#ifdef CSP_ENABLE_EPOCH_STAT_CHECK
    /* Check access epoch status.
     * The current epoch must be lock on target.*/
    if (ug_win->epoch_stat != CSP_WIN_EPOCH_PER_TARGET) {
        CSP_ERR_PRINT("Wrong synchronization call! "
                      "No opening per-target epoch in %s\n", __FUNCTION__);
        mpi_errno = -1;
        goto fn_fail;
    }

    /* Check per-target access epoch status. */
    if (target->epoch_stat != CSP_TARGET_EPOCH_LOCK) {
        CSP_ERR_PRINT("Wrong synchronization call! "
                      "No opening LOCK epoch on target %d in %s\n", target_rank, __FUNCTION__);
        mpi_errno = -1;
        goto fn_fail;
    }
#endif

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

    /* Unlock all ghost processes in every ug-window of target process. */
#ifdef CSP_ENABLE_SYNC_ALL_OPT
    /* lock_all cannot handle exclusive locks, thus should use only for shared lock or nocheck. */
    if (target->remote_lock_type == MPI_LOCK_SHARED ||
        target->remote_lock_assert & MPI_MODE_NOCHECK) {
        /* Optimization for MPI implementations that have optimized lock_all.
         * However, user should be noted that, if MPI implementation issues lock messages
         * for every target even if it does not have any operation, this optimization
         * could lose performance and even lose asynchronous! */

        CSP_DBG_PRINT("[%d]unlock_all(ug_win 0x%x), instead of target rank %d\n",
                      user_rank, target->ug_win, target_rank);
        mpi_errno = PMPI_Win_unlock_all(target->ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        if (user_rank == target_rank)
            ug_win->is_self_locked = 0;
    }
    else
#endif /* end of CSP_ENABLE_SYNC_ALL_OPT */
    {
        if (target->synced_async_stat == CSP_TARGET_ASYNC_ON) {
            /* unlock all ghosts. */
            mpi_errno = CSP_win_target_unlock_ghosts(target_rank, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
        else {
            /* When async is off on that target, we only unlock the target process.
             * Static per-coll scheduling must be collective, thus is invalid during a lock epoch. */
            mpi_errno = CSP_win_target_unlock_user(target_rank, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }

        /* If target is itself, we need also release the lock of local rank  */
        if (user_rank == target_rank && ug_win->is_self_locked /*already unlocked */) {
            mpi_errno = CSP_win_unlock_self_impl(ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
    }

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    int j;
    for (j = 0; j < target->num_segs; j++) {
        target->segs[j].main_lock_stat = CSP_MAIN_LOCK_RESET;
    }
#endif

    /* Reset per-target epoch. */
    target->epoch_stat = CSP_TARGET_NO_EPOCH;
    target->remote_lock_assert = 0;
    target->remote_lock_type = 0;

    /* Reset global epoch status. */
    ug_win->lock_counter--;
    CSP_assert(ug_win->lock_counter >= 0);
    if (ug_win->start_counter == 0 && ug_win->lock_counter == 0) {
        CSP_DBG_PRINT("all per-target epoch are cleared !\n");
        ug_win->epoch_stat = CSP_WIN_NO_EPOCH;
    }

  fn_exit:
    CSP_rm_count_end(CSP_RM_COMM_FREQ);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
