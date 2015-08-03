/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef CSP_RMA_LOCAL_H_
#define CSP_RMA_LOCAL_H_

/* This header file includes all generic routines used in local RMA communication. */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "csp.h"

#define CSP_get_lock_type_name(lock_type) (lock_type == MPI_LOCK_SHARED ? "SHARED" : "EXCLUSIVE")
#define CSP_get_assert_name(assert) (assert & MPI_MODE_NOCHECK ? "NOCHECK" : (assert == 0 ? "none" : "other"))
#define CSP_get_win_type(win, ug_win) ((win) == ug_win->active_win ? "active_win" : "ug_wins")

/* Get the window for lock/lockall epochs (do not check other epochs).
 * If only lock-all is used, then return the global window, otherwise return
 * per-target window. */
static inline void CSP_win_get_epoch_lock_win(CSP_win * ug_win, CSP_win_target * target,
                                              MPI_Win * win_ptr)
{
    if (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) {
        (*win_ptr) = target->ug_win;
    }
    else {
        (*win_ptr) = ug_win->active_win;
    }
}

/* Lock self process on lock window.
 * It is called by both WIN_LOCK and WIN_LOCK_ALL. */
static inline int CSP_win_lock_self_impl(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_win_target *target;
    int user_rank;
    MPI_Win lock_win = MPI_WIN_NULL;

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    target = &(ug_win->targets[user_rank]);

    CSP_win_get_epoch_lock_win(ug_win, target, &lock_win);

    CSP_DBG_PRINT("[%d]lock self(%d, local win 0x%x)\n", user_rank,
                  ug_win->my_rank_in_ug_comm, lock_win);
    mpi_errno = PMPI_Win_lock(MPI_LOCK_SHARED, ug_win->my_rank_in_ug_comm,
                              MPI_MODE_NOCHECK, lock_win);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    ug_win->is_self_locked = 1;
    return mpi_errno;
}

/* Unlock self process on lock window.
 * It is called by both WIN_UNLOCK and WIN_UNLOCK_ALL. */
static inline int CSP_win_unlock_self_impl(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;

    CSP_win_target *target;
    int user_rank;
    MPI_Win lock_win = MPI_WIN_NULL;

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    target = &(ug_win->targets[user_rank]);

    CSP_win_get_epoch_lock_win(ug_win, target, &lock_win);

    CSP_DBG_PRINT("[%d]unlock self(%d, local win 0x%x)\n", user_rank,
                  ug_win->my_rank_in_ug_comm, lock_win);
    mpi_errno = PMPI_Win_unlock(ug_win->my_rank_in_ug_comm, lock_win);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    ug_win->is_self_locked = 0;
    return mpi_errno;
}

#ifdef CSP_ENABLE_LOCAL_LOCK_OPT
static inline int CSP_win_flush_self_impl(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;

    CSP_win_target *target;
    int user_rank;
    MPI_Win lock_win = MPI_WIN_NULL;

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    target = &(ug_win->targets[user_rank]);

    CSP_win_get_epoch_lock_win(ug_win, target, &lock_win);

    if (ug_win->is_self_locked) {
        CSP_DBG_PRINT("[%d]flush self(%d, local win 0x%x)\n", user_rank,
                      ug_win->my_rank_in_ug_comm, lock_win);
        mpi_errno = PMPI_Win_flush(ug_win->my_rank_in_ug_comm, lock_win);
    }
    return mpi_errno;
}
#endif

/* Lock user process for a target.
 * It is called by both WIN_LOCK, and WIN_LOCK_ALL for mixed-lock mode.*/
static inline int CSP_win_target_lock_user(int lock_type, int assert, int target_rank,
                                           CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int user_rank;
    CSP_win_target *target = NULL;

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    target = &(ug_win->targets[target_rank]);

    CSP_DBG_PRINT("[%d]lock(target(%d), ug_wins 0x%x, lock=%s, assert=%s), instead of "
                  "target rank %d\n", user_rank, target->ug_rank, target->ug_win,
                  CSP_get_lock_type_name(lock_type), CSP_get_assert_name(assert), target_rank);
    mpi_errno = PMPI_Win_lock(lock_type, target->ug_rank, assert, target->ug_win);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    if (user_rank == target_rank)
        ug_win->is_self_locked = 1;

    return mpi_errno;
}

/* Flush user process on the locked window (ug_win or active_win) for a target.
 * It can be called by WIN_FLUSH, WIN_FLUSH_ALL, WIN_FENCE, or WIN_COMPLETE.  */
static inline int CSP_win_target_flush_user(int target_rank, CSP_win * ug_win, int *is_self_flushed)
{
    int mpi_errno = MPI_SUCCESS;
    int user_rank;
    CSP_win_target *target = NULL;
    MPI_Win *win_ptr = NULL;

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    target = &(ug_win->targets[target_rank]);

    CSP_target_get_epoch_win(0, target, ug_win, win_ptr);

    CSP_DBG_PRINT("[%d]flush(target(%d), %s 0x%x), instead of "
                  "target rank %d\n", user_rank, target->ug_rank,
                  CSP_get_win_type(*win_ptr, ug_win), *win_ptr, target_rank);
    mpi_errno = PMPI_Win_flush(target->ug_rank, *win_ptr);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    if (target_rank == user_rank)
        (*is_self_flushed) = 1; /* avoid twice self flush */

    return mpi_errno;
}

/* Unlock user process for a target.
 * It is called by both WIN_UNLOCK, and WIN_UNLOCK_ALL for mixed-lock mode.*/
static inline int CSP_win_target_unlock_user(int target_rank, CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int user_rank;
    CSP_win_target *target = NULL;

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    target = &(ug_win->targets[target_rank]);

    CSP_DBG_PRINT("[%d]unlock(target %d,  0x%x)\n", user_rank, target->ug_rank, target->ug_win);
    mpi_errno = PMPI_Win_unlock(target->ug_rank, target->ug_win);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    if (user_rank == target_rank)
        ug_win->is_self_locked = 0;

    return mpi_errno;
}

extern int CSP_win_target_lock_ghosts(int lock_type, int assert, int target_rank, CSP_win * ug_win);
extern int CSP_win_target_flush_ghosts(int target_rank, CSP_win * ug_win);
extern int CSP_win_target_unlock_ghosts(int target_rank, CSP_win * ug_win);
extern int CSP_win_flush_all(CSP_win * ug_win);

#endif /* CSP_RMA_LOCAL_H_ */
