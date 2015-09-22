/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"
#include "csp_rma_local.h"

/* Locks ghost processes for a given target.
 * It is called by both WIN_UNLOCK and WIN_UNLOCK_ALL for mixed-lock mode. */
int CSP_win_target_lock_ghosts(int lock_type, int assert, int target_rank, CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_win_target *target = NULL;
    int user_rank;
    int k, j;

    target = &(ug_win->targets[target_rank]);
    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

    /* Lock every ghost on every window for each target.
     * Note that a ghost may be used on any window of this process for runtime
     * load balancing whether it is bound to that segment or not. */
    for (k = 0; k < CSP_ENV.num_g; k++) {
        int target_g_rank_in_ug = 0;
        int g_lock_type = MPI_LOCK_SHARED;
        int g_assert = MPI_MODE_NOCHECK;

        /* Only the main ghosts have permission check, other ghosts should just get
         * shared & nocheck lock. Otherwise deadlock may happen in binding-free stage. */
        for (j = 0; j < target->num_segs; j++) {
            if (target->segs[j].main_g_off == k) {
                g_lock_type = lock_type;
                g_assert = assert;
                break;
            }
        }

        target_g_rank_in_ug = target->g_ranks_in_ug[k];

        CSP_DBG_PRINT("[%d]lock(Ghost(%d), ug_win 0x%x, lock=%s, assert=%s), instead of "
                      "target rank %d\n", user_rank, target_g_rank_in_ug, target->ug_win,
                      CSP_get_lock_type_name(g_lock_type), CSP_get_assert_name(g_assert),
                      target_rank);
        mpi_errno = PMPI_Win_lock(g_lock_type, target_g_rank_in_ug, g_assert, target->ug_win);
        if (mpi_errno != MPI_SUCCESS)
            break;
    }

    return mpi_errno;
}

static int CSP_win_mixed_lock_all_impl(int assert, CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int user_rank, user_nprocs;
    int i;
    int is_local_lock_granted CSP_ATTRIBUTE((unused));
    int is_g_locked = 0;

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);

    ug_win->is_self_locked = 0;

#ifdef CSP_ENABLE_SYNC_ALL_OPT

    /* Optimization for MPI implementations that have optimized lock_all.
     * However, user should be noted that, if MPI implementation issues lock messages
     * for every target even if it does not have any operation, this optimization
     * could lose performance and even lose asynchronous! */
    for (i = 0; i < ug_win->num_ug_wins; i++) {
        CSP_DBG_PRINT("[%d]lock_all(ug_win 0x%x)\n", user_rank, ug_win->ug_wins[i]);
        mpi_errno = PMPI_Win_lock_all(assert, ug_win->ug_wins[i]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
    is_g_locked = 1;
    ug_win->is_self_locked = 1;
#else

    if (CSP_ENV.async_sched_level == CSP_ASYNC_SCHED_PER_WIN) {
        /* Lock either user process or ghost in per-window scheduling. */
        for (i = 0; i < user_nprocs; i++) {
            if (ug_win->targets[i].synced_async_stat == CSP_ASYNC_ON) {
                mpi_errno = CSP_win_target_lock_ghosts(MPI_LOCK_SHARED, assert, i, ug_win);
                if (mpi_errno != MPI_SUCCESS)
                    goto fn_fail;
                if (user_rank == i)
                    is_g_locked = 1;    /* force lock only when ghosts are locked */
            }
            else {
                mpi_errno = CSP_win_target_lock_user(MPI_LOCK_SHARED, assert, i, ug_win);
                if (mpi_errno != MPI_SUCCESS)
                    goto fn_fail;
            }
        }
    }
    else {
        /* Lock both user process and ghost in higher scheduling */
        for (i = 0; i < user_nprocs; i++) {
            mpi_errno = CSP_win_target_lock_ghosts(MPI_LOCK_SHARED, assert, i, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;

            mpi_errno = CSP_win_target_lock_user(MPI_LOCK_SHARED, assert, i, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
        is_g_locked = 1;        /* force lock only when ghosts are locked */
    }
#endif

    if (is_g_locked && !ug_win->info_args.no_local_load_store &&
        !(ug_win->targets[user_rank].remote_lock_assert & MPI_MODE_NOCHECK)) {
        /* We need grant the local lock (self-target) before return.
         * However, the actual locked processes are the Ghosts whose locks may be delayed by
         * most MPI implementation, thus we need a flush to force the lock to be granted on ghost 0
         * who is the one actually controls the locks.
         *
         * For performance reason, this operation is ignored if meet at least one of following conditions:
         * 1. if user passed information that this process will not do local load/store on this window.
         * 2. if user passed information that there is no concurrent epochs.
         */
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

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPI_Win_lock_all(int assert, MPI_Win win)
{
    CSP_win *ug_win;
    int mpi_errno = MPI_SUCCESS;
    int user_rank, user_nprocs;
    int i;

    CSP_DBG_PRINT_FCNAME();
    CSP_MPI_FUNC_START_ROUTINE();

    CSP_fetch_ug_win_from_cache(win, ug_win);

    if (ug_win == NULL) {
        /* normal window */
        return PMPI_Win_lock_all(assert, win);
    }

    /* casper window starts */

    CSP_assert((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) ||
               (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK_ALL));

#ifdef CSP_ENABLE_EPOCH_STAT_CHECK
    /* Check access epoch status.
     * We do not require closed FENCE epoch, because we don't know whether
     * the previous FENCE is closed or not.*/
    if (ug_win->epoch_stat == CSP_WIN_EPOCH_LOCK_ALL
        || ug_win->epoch_stat == CSP_WIN_EPOCH_PER_TARGET) {
        CSP_ERR_PRINT("Wrong synchronization call! "
                      "Previous %s epoch is still open in %s\n",
                      (ug_win->epoch_stat == CSP_WIN_EPOCH_LOCK_ALL) ? "LOCK_ALL" : "PER_TARGET",
                      __FUNCTION__);
        mpi_errno = -1;
        goto fn_fail;
    }
    CSP_assert(ug_win->start_counter == 0 && ug_win->lock_counter == 0);
#endif

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);

    for (i = 0; i < user_nprocs; i++) {
        ug_win->targets[i].remote_lock_assert = assert;
        ug_win->targets[i].remote_lock_type = MPI_LOCK_SHARED;
    }

    CSP_DBG_PRINT("[%d]lock_all, MPI_MODE_NOCHECK %d(assert %d)\n", user_rank,
                  (assert & MPI_MODE_NOCHECK) != 0, assert);

    if (!(ug_win->info_args.epoch_type & CSP_EPOCH_LOCK)) {

        /* In lock_all only epoch, lock_all already issued on global window
         * in win_allocate. */
        CSP_DBG_PRINT("[%d]lock_all(active_win 0x%x) (no actual lock call)\n", user_rank,
                      ug_win->active_win);

        /* Do not need grant lock before lock local target, because only shared lock
         * in current epoch. */
        ug_win->is_self_locked = 1;
    }
    else {

        /* In lock_all/lock mixed epoch, separate windows are bound with each target. */
        mpi_errno = CSP_win_mixed_lock_all_impl(assert, ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    int j;
    for (i = 0; i < user_nprocs; i++) {
        for (j = 0; j < ug_win->targets[i].num_segs; j++) {
            ug_win->targets[i].segs[j].main_lock_stat = CSP_MAIN_LOCK_RESET;

            CSP_reset_target_opload(i, ug_win);
        }
    }
#endif

    /* Indicate epoch status.
     * Later operations will be redirected to single window.*/
    ug_win->epoch_stat = CSP_WIN_EPOCH_LOCK_ALL;

  fn_exit:
    CSP_MPI_FUNC_END_ROUTINE();
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
