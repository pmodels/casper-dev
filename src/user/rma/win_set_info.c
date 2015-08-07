/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include "csp.h"

int MPI_Win_set_info(MPI_Win win, MPI_Info info)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_win *ug_win;
    int symmetric_flag = 0;

    CSP_DBG_PRINT_FCNAME();
    CSP_rm_count_start(CSP_RM_COMM_FREQ);

    CSP_fetch_ug_win_from_cache(win, ug_win);
    if (ug_win == NULL) {
        /* normal window */
        mpi_errno = PMPI_Win_set_info(win, info);
        goto fn_exit;
    }

    if (info != MPI_INFO_NULL) {
        int info_flag = 0;
        char info_value[MPI_MAX_INFO_VAL + 1];

        memset(info_value, 0, sizeof(info_value));
        mpi_errno = PMPI_Info_get(info, "symmetric", MPI_MAX_INFO_VAL, info_value, &info_flag);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        if (info_flag == 1) {
            if (!strncmp(info_value, "true", strlen("true"))) {
                symmetric_flag = 1;
            }
        }
    }

    /* This is a symmetric call on all processes with the same info value.
     * It is safe to reschedule asynchronous configuration for this window. */
    if (symmetric_flag && CSP_ENV.async_sched_level >= CSP_ASYNC_SCHED_PER_COLL) {
        mpi_errno = CSP_win_get_async_config_info(info, &ug_win->info_args.async_config, NULL);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;

        /* always reschedule async_config, since no cost for static configuration，
         * and could trigger dynamic adaptation. */
        mpi_errno = CSP_win_sched_async_config(ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

  fn_exit:
    CSP_rm_count_end(CSP_RM_COMM_FREQ);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
