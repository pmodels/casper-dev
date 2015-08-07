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

    if (CSP_ENV.async_sched_level >= CSP_ASYNC_SCHED_PER_COLL) {
        CSP_async_config async_config;
        int async_config_phases = 0;
        int set_flag = 0;
        int user_rank = 0;

        PMPI_Comm_rank(ug_win->user_comm, &user_rank);

        mpi_errno = CSP_win_get_async_config_info(info, &async_config,
                                                  &async_config_phases, &set_flag);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
        /* Locally reschedule my status according to runtime monitored data. */
        /* FIXME: this local update is not related to any window, should not be
         * triggered in win_set_info. */
        if (async_config_phases & CSP_ASYNC_CONFIG_PHASE_LOCAL_UPDATE) {
            CSP_ra_sched_async_stat();

            if (CSP_ENV.verbose >= 2 && user_rank == 0)
                CSP_INFO_PRINT(2, "[local-update] my async_config: %s, freq=%d (%.4f/%.4f)\n",
                               CSP_get_async_config_name(CSP_ra_get_async_stat()),
                               CSP_RM[CSP_RM_COMM_FREQ].last_freq,
                               CSP_RM[CSP_RM_COMM_FREQ].last_time,
                               CSP_RM[CSP_RM_COMM_FREQ].last_interval);
        }
#endif

        /* This is a symmetric call on all processes with the same info value.
         * It is safe to update asynchronous configuration for this window. */
        if (symmetric_flag) {
            /* update window setting */
            if (set_flag)
                ug_win->info_args.async_config = async_config;
            ug_win->info_args.async_config_phases = async_config_phases;

            /* update asynchronous configures */
            mpi_errno = CSP_win_sched_async_config(ug_win);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
    }

  fn_exit:
    CSP_rm_count_end(CSP_RM_COMM_FREQ);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
