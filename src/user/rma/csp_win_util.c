/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csp.h"
#include "casper.h"

void CSP_win_dump_async_config(MPI_Win win, const char *fname)
{
    CSP_win *ug_win = NULL;
    FILE *fp = NULL;
    char ffname[128];
    int mpi_errno CSP_ATTRIBUTE((unused)) = MPI_SUCCESS;

    CSP_DBG_PRINT_FCNAME();

    sprintf(ffname, "async-dump-%s", fname);
    fp = fopen(ffname, "a");
    if (fp != NULL) {
        CSP_fetch_ug_win_from_cache(win, ug_win);
        if (ug_win == NULL) {
            char *name = NULL;
            CSP_fetch_win_name_from_cache(win, name);

            /* all off */
            fprintf(fp, "CASPER Window: %s, async_config=%s\n", name, "off");
        }
        else {
            /* on or auto */
            fprintf(fp, "CASPER Window: %s, async_config=%s\n",
                    ug_win->info_args.win_name,
                    CSP_get_async_config_name(ug_win->info_args.async_config));

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
            if (ug_win->info_args.async_config == CSP_ASYNC_CONFIG_AUTO) {
                int i, user_nprocs = 0;
                int async_on_cnt = 0;
                int async_off_cnt = 0;
                PMPI_Comm_size(ug_win->user_comm, &user_nprocs);
                for (i = 0; i < user_nprocs; i++) {
                    if (ug_win->targets[i].synced_async_stat == CSP_ASYNC_ON) {
                        async_on_cnt++;
                    }
                    else {
                        async_off_cnt++;
                    }
                }
                fprintf(fp, "    Per-target async_config summary: on %d; off %d\n",
                        async_on_cnt, async_off_cnt);
            }
#endif
        }
        fflush(fp);
        fclose(fp);
    }
}

/*
 * Get asynchronous configuration from info parameter.
 * This call is also in by win_allocate.
 */
int CSP_win_get_async_config_info(MPI_Info info, CSP_async_config * async_config,
                                  int *set_config_flag, int *async_config_phases,
                                  int *set_phases_flag)
{
    int mpi_errno = MPI_SUCCESS;
    int tmp_set_config_flag = 0, tmp_set_phases_flag = 0;

    if (info != MPI_INFO_NULL) {
        int info_flag = 0;
        char info_value[MPI_MAX_INFO_VAL + 1];

        memset(info_value, 0, sizeof(info_value));
        mpi_errno = PMPI_Info_get(info, "async_config", MPI_MAX_INFO_VAL, info_value, &info_flag);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;

        if (info_flag == 1) {
            if (!strncmp(info_value, "off", strlen("off"))) {
                (*async_config) = CSP_ASYNC_CONFIG_OFF;
                tmp_set_config_flag = 1;
            }
            else if (!strncmp(info_value, "on", strlen("on"))) {
                (*async_config) = CSP_ASYNC_CONFIG_ON;
                tmp_set_config_flag = 1;
            }
#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
            else if (!strncmp(info_value, "auto", strlen("auto"))) {
                (*async_config) = CSP_ASYNC_CONFIG_AUTO;
                tmp_set_config_flag = 1;
            }
#endif
        }
#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
        memset(info_value, 0, sizeof(info_value));
        mpi_errno = PMPI_Info_get(info, "async_config_phases",
                                  MPI_MAX_INFO_VAL, info_value, &info_flag);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;

        if (info_flag == 1) {
            if (!strncmp(info_value, "local-update", strlen("local-update"))) {
                (*async_config_phases) = CSP_ASYNC_CONFIG_PHASE_LOCAL_UPDATE;
                tmp_set_phases_flag = 1;
            }
            else if (!strncmp(info_value, "remote-exchange", strlen("remote-exchange"))) {
                (*async_config_phases) = CSP_ASYNC_CONFIG_PHASE_REMOTE_EXCHANGE;
                tmp_set_phases_flag = 1;
            }
        }
#endif
    }

    if (set_config_flag != NULL)
        (*set_config_flag) = tmp_set_config_flag;
    if (set_phases_flag != NULL)
        (*set_phases_flag) = tmp_set_phases_flag;

    return mpi_errno;
}

int CSP_win_print_async_config(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int user_nprocs, user_rank;
    double *tmp_async_gather_buf = NULL;

    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);
    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    if (ug_win->info_args.async_config == CSP_ASYNC_CONFIG_AUTO) {
        int i;

        /* Only the root user prints a summary of the async status. */
        if (user_rank == 0) {
            int async_on_cnt = 0, async_off_cnt = 0;
            for (i = 0; i < user_nprocs; i++) {
                if (ug_win->targets[i].synced_async_stat == CSP_ASYNC_ON) {
                    async_on_cnt++;
                }
                else {
                    async_off_cnt++;
                }
            }
            CSP_INFO_PRINT(2, "CASPER Window %s: target async_config: on %d; off %d\n",
                           ug_win->info_args.win_name, async_on_cnt, async_off_cnt);
            CSP_INFO_PRINT_FILE_APPEND(1, "    Per-target async_config summary: on %d; off %d\n",
                                       async_on_cnt, async_off_cnt);
        }

        /* The root user gathers the frequency number on all targets and print in file. */
        if (CSP_ENV.file_verbose >= 1) {
            /* Gather frequency numbers to root user */
            tmp_async_gather_buf = CSP_calloc(3 * user_nprocs, sizeof(double));

            tmp_async_gather_buf[3 * user_rank] = CSP_RM[CSP_RM_COMM_FREQ].last_freq * 1.0;
            tmp_async_gather_buf[3 * user_rank + 1] = CSP_RM[CSP_RM_COMM_FREQ].last_time;
            tmp_async_gather_buf[3 * user_rank + 2] = CSP_RM[CSP_RM_COMM_FREQ].last_interval;
            mpi_errno = PMPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, tmp_async_gather_buf,
                                       3, MPI_DOUBLE, ug_win->user_comm);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;

            if (user_rank == 0) {
                CSP_INFO_PRINT_FILE_APPEND(1, "    Per-target async_config:\n");
                for (i = 0; i < user_nprocs; i++)
                    CSP_INFO_PRINT_FILE_APPEND(1, "    [%d] async stat: freq=%.0f(%.4f/%.4f)\n",
                                               i, tmp_async_gather_buf[3 * i],
                                               tmp_async_gather_buf[3 * i + 1],
                                               tmp_async_gather_buf[3 * i + 2]);
            }
        }
    }
    else
#endif
    {
        if (user_rank == 0) {
            const char *name = CSP_get_async_config_name(ug_win->info_args.async_config);
            CSP_INFO_PRINT(2, "CASPER Window: %s target async_config: all %s\n",
                           ug_win->info_args.win_name, name);
        }
    }

  fn_exit:
    if (tmp_async_gather_buf)
        free(tmp_async_gather_buf);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

/* Schedule asynchronous configuration in collective manner.
 * This routine is safe to be used in window collective calls, such as win_fence
 * and win_set_info with symmetric hint (user has to guarantee collective and remote
 * completion of all issued operations before win_set_info started).
 *
 * Note that win_allocate is also a collective call, but it uses a separate routine
 * since it is not fully controlled by the async_config_phases hint (remote-exchange
 * phase can not be skipped) and we want to merge it into the original allgather
 * operation to hide latency.
 *
 * This routine contains following steps:
 * - Every process updates its local asynchronous status if "local-update" phase is enabled;
 * - Every process collectively exchanges with other processes if "remote-exchange"
 *   phase is enabled.
 */
int CSP_win_coll_sched_async_config(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint *tmp_gather_buf = NULL;
    CSP_async_stat my_async_stat = CSP_ASYNC_NONE;
    int user_nprocs, user_rank, i;

    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);
    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

    if (user_rank == 0)
        CSP_INFO_PRINT_FILE_START(1, ug_win->info_args.win_name);

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    /* If runtime scheduling is enabled for this window, we exchange the
     * asynchronous configure with every target, since its value might be different. */
    if (ug_win->info_args.async_config == CSP_ASYNC_CONFIG_AUTO) {
        if (ug_win->info_args.async_config_phases & CSP_ASYNC_CONFIG_PHASE_LOCAL_UPDATE) {
            /* reschedule my status according to runtime monitored data. */
            CSP_ra_sched_async_stat();
        }

        if (ug_win->info_args.async_config_phases & CSP_ASYNC_CONFIG_PHASE_REMOTE_EXCHANGE) {
            /* read my current status. */
            my_async_stat = CSP_ra_get_async_stat();

            /* exchange with all targets. */
            tmp_gather_buf = calloc(user_nprocs, sizeof(MPI_Aint));
            tmp_gather_buf[user_rank] = (MPI_Aint) my_async_stat;

            mpi_errno = PMPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                       tmp_gather_buf, 1, MPI_AINT, ug_win->user_comm);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;

            for (i = 0; i < user_nprocs; i++) {
                ug_win->targets[i].synced_async_stat = (CSP_async_stat) tmp_gather_buf[i];
                ug_win->targets[i].async_stat = my_async_stat;
            }
        }
    }
    else
#endif
    {
        /* Locally set for all targets since the value is globally consistent. */
        my_async_stat = (ug_win->info_args.async_config == CSP_ASYNC_CONFIG_ON) ?
            CSP_ASYNC_ON : CSP_ASYNC_OFF;
        for (i = 0; i < user_nprocs; i++) {
            ug_win->targets[i].synced_async_stat = my_async_stat;
            ug_win->targets[i].async_stat = my_async_stat;
        }
    }

    if (CSP_ENV.verbose >= 2 && user_rank == 0) {
        if (ug_win->info_args.async_config_phases == CSP_ASYNC_CONFIG_PHASE_REMOTE_EXCHANGE) {
            CSP_INFO_PRINT(2, "[remote-exchange] ");
        }
        else {
            CSP_INFO_PRINT(2, "[update] ");
        }
    }

    if (CSP_ENV.verbose >= 2 || CSP_ENV.file_verbose >= 1)
        CSP_win_print_async_config(ug_win);

  fn_exit:
    if (user_rank == 0)
        CSP_INFO_PRINT_FILE_END(1, ug_win->info_args.win_name);
    if (tmp_gather_buf)
        free(tmp_gather_buf);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
