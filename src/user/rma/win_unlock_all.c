/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"
#include "csp_rma_local.h"

/* Unlocks ghost processes for a given target.
 * It is called by both WIN_UNLOCK and WIN_UNLOCK_ALL for mixed-lock mode. */
int CSP_win_target_unlock_ghosts(int target_rank, CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_win_target *target = NULL;
    int k;
    int user_rank;

    target = &(ug_win->targets[target_rank]);
    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

    /* Unlock every ghost on every window for each target. */
    for (k = 0; k < CSP_ENV.num_g; k++) {
        int target_g_rank_in_ug = target->g_ranks_in_ug[k];

        CSP_DBG_PRINT("[%d]unlock(Ghost(%d), ug_win 0x%x), instead of "
                      "target rank %d\n", user_rank, target_g_rank_in_ug,
                      target->ug_win, target_rank);
        mpi_errno = PMPI_Win_unlock(target_g_rank_in_ug, target->ug_win);
        if (mpi_errno != MPI_SUCCESS)
            break;
    }

    return mpi_errno;
}

static int CSP_win_mixed_unlock_all_impl(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int user_rank, user_nprocs;
    int i;

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);

#ifdef CSP_ENABLE_SYNC_ALL_OPT

    /* Optimization for MPI implementations that have optimized lock_all.
     * However, user should be noted that, if MPI implementation issues lock messages
     * for every target even if it does not have any operation, this optimization
     * could lose performance and even lose asynchronous! */
    for (i = 0; i < ug_win->num_ug_wins; i++) {
        CSP_DBG_PRINT("[%d]unlock_all(ug_win 0x%x)\n", user_rank, ug_win->ug_wins[i]);
        mpi_errno = PMPI_Win_unlock_all(ug_win->ug_wins[i]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
#else
    if (CSP_ENV.async_sched_level == CSP_ASYNC_SCHED_PER_WIN) {
        /* Unlock either user process or ghost in per-window scheduling. */
        for (i = 0; i < user_nprocs; i++) {
            if (ug_win->targets[i].synced_async_stat == CSP_ASYNC_ON) {
                mpi_errno = CSP_win_target_unlock_ghosts(i, ug_win);
                if (mpi_errno != MPI_SUCCESS)
                    goto fn_fail;
            }
            else {
                mpi_errno = CSP_win_target_unlock_user(i, ug_win);
                if (mpi_errno != MPI_SUCCESS)
                    goto fn_fail;
            }
        }
    }
    else {
        /* Unlock both user process and ghost in higher scheduling */
        for (i = 0; i < user_nprocs; i++) {
            mpi_errno = CSP_win_target_unlock_ghosts(i, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;

            mpi_errno = CSP_win_target_unlock_user(i, ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
    }
#endif

    if (ug_win->is_self_locked) {
        mpi_errno = CSP_win_unlock_self_impl(ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPI_Win_unlock_all(MPI_Win win)
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
        return PMPI_Win_unlock_all(win);
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

        /* In lock_all only epoch, unlock_all will be issued on global window
         * in win_free. We only need flush_all here.*/
        CSP_DBG_PRINT("[%d]unlock_all(active_win 0x%x) (no actual unlock call)\n",
                      user_rank, ug_win->active_win);

        mpi_errno = CSP_win_flush_all(ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        ug_win->is_self_locked = 0;
    }
    else {

        /* In lock_all/lock mixed epoch, separate windows are bound with each target. */
        mpi_errno = CSP_win_mixed_unlock_all_impl(ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    for (i = 0; i < user_nprocs; i++) {
        int j;
        for (j = 0; j < ug_win->targets[i].num_segs; j++) {
            ug_win->targets[i].segs[j].main_lock_stat = CSP_MAIN_LOCK_RESET;
        }
    }
#endif

    /* Reset epoch. */
    for (i = 0; i < user_nprocs; i++) {
        ug_win->targets[i].remote_lock_assert = 0;
        ug_win->targets[i].remote_lock_type = 0;
    }
    ug_win->epoch_stat = CSP_WIN_NO_EPOCH;

  fn_exit:
    CSP_MPI_FUNC_END_ROUTINE();
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
