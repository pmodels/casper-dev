/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"
#include "csp_rma_local.h"


int MPI_Win_flush(int target_rank, MPI_Win win)
{
    CSP_win *ug_win;
    CSP_win_target *target;
    int mpi_errno = MPI_SUCCESS;
    int user_rank;
    int is_self_flushed CSP_ATTRIBUTE((unused)) = 0;
    int j CSP_ATTRIBUTE((unused));

    CSP_FUNC_PROFILE_TIMING_START(MPI_WIN_FLUSH);

    CSP_DBG_PRINT_FCNAME();
    CSP_FUNC_PROFILE_TIMING_START(MPI_WIN_FLUSH_GADPT);
    CSP_MPI_FUNC_START_ROUTINE();
    CSP_FUNC_PROFILE_TIMING_END(MPI_WIN_FLUSH_GADPT);
    CSP_FUNC_PROFILE_TIMING_START(MPI_WIN_FLUSH_MAIN);

    CSP_fetch_ug_win_from_cache(win, ug_win);

    if (ug_win == NULL) {
        /* normal window */
        return PMPI_Win_flush(target_rank, win);
    }

    /* casper window starts */

    if (target_rank == MPI_PROC_NULL)
        goto fn_exit;

    CSP_assert((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) ||
               (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK_ALL));

    target = &(ug_win->targets[target_rank]);

#ifdef CSP_ENABLE_EPOCH_STAT_CHECK
    /* Check access epoch status.
     * The current epoch must be lock_all or lock.*/
    if (ug_win->epoch_stat != CSP_WIN_EPOCH_LOCK_ALL &&
        (target->epoch_stat != CSP_TARGET_EPOCH_LOCK)) {
        CSP_ERR_PRINT("Wrong synchronization call! "
                      "No opening LOCK_ALL or LOCK epoch in %s\n", __FUNCTION__);
        mpi_errno = -1;
        goto fn_fail;
    }
#endif

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

#ifdef CSP_ENABLE_SYNC_ALL_OPT
    /* lock_all cannot handle exclusive locks, thus should use only for shared lock or nocheck. */
    if (target->remote_lock_type == MPI_LOCK_SHARED ||
        target->remote_lock_assert & MPI_MODE_NOCHECK) {

        /* Optimization for MPI implementations that have optimized lock_all.
         * However, user should be noted that, if MPI implementation issues lock messages
         * for every target even if it does not have any operation, this optimization
         * could lose performance and even lose asynchronous! */
        if (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) {
            CSP_DBG_PRINT("[%d]flush_all(ug_win 0x%x), instead of target rank %d\n",
                          user_rank, target->ug_win, target_rank);
            mpi_errno = PMPI_Win_flush_all(target->ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
        else {
            CSP_DBG_PRINT("[%d]flush_all(active_win 0x%x), instead of target rank %d\n",
                          user_rank, ug_win->active_win);
            mpi_errno = PMPI_Win_flush_all(ug_win->active_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
    }
    else
#endif /*end of CSP_ENABLE_SYNC_ALL_OPT */
    {
        if (target->synced_async_stat == CSP_ASYNC_ON ||
            CSP_ENV.async_sched_level == CSP_ASYNC_SCHED_ANYTIME) {
            /* flush ghosts if async is on */
            mpi_errno = CSP_win_target_flush_ghosts(target_rank, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }

        if (target->synced_async_stat == CSP_ASYNC_OFF ||
            CSP_ENV.async_sched_level == CSP_ASYNC_SCHED_ANYTIME) {
            /* flush target if async is off */
            mpi_errno = CSP_win_target_flush_user(target_rank, ug_win, &is_self_flushed);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }

#ifdef CSP_ENABLE_LOCAL_LOCK_OPT
        if (user_rank == target_rank && !is_self_flushed) {
            /* If LOCAL_LOCK_OPT is enabled, PUT/GET may be issued to local
             * target. Thus we need flush the local target as well.
             * Note that ACC operations are always issued to main ghost,
             * since atomicity and ordering issue. */
            mpi_errno = CSP_win_flush_self_impl(ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
#endif
    }

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    for (j = 0; j < target->num_segs; j++) {
        /* Lock of main ghost is granted, we can start load balancing from the next flush/unlock.
         * Note that only target which was issued operations to is guaranteed to be granted. */
        if (target->segs[j].main_lock_stat == CSP_MAIN_LOCK_OP_ISSUED) {
            target->segs[j].main_lock_stat = CSP_MAIN_LOCK_GRANTED;
            CSP_DBG_PRINT("[%d] main lock (rank %d, seg %d) granted\n", user_rank, target_rank, j);
        }

        CSP_reset_target_opload(target_rank, ug_win);
    }
#endif

  fn_exit:
    CSP_FUNC_PROFILE_TIMING_END(MPI_WIN_FLUSH_MAIN);
    CSP_MPI_FUNC_END_ROUTINE();

    CSP_FUNC_PROFILE_COUNTER_INC(MPI_WIN_FLUSH);
    CSP_FUNC_PROFILE_TIMING_END(MPI_WIN_FLUSH);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
