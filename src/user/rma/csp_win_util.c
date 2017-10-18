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
                                  int *set_config_flag,
                                  int *async_config_phases CSP_ATTRIBUTE((unused)),
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
            CSP_INFO_PRINT(2, "CASPER Window %s-%d: target async_config: on %d; off %d\n",
                           ug_win->info_args.win_name, ug_win->adapt_remote_exed,
                           async_on_cnt, async_off_cnt);
        }
    }
    else
#endif
    {
        if (user_rank == 0) {
            CSP_INFO_PRINT(2, "CASPER Window: %s target async_config: all %s\n",
                           ug_win->info_args.win_name,
                           CSP_get_async_config_name(ug_win->info_args.async_config));
        }
    }

    CSP_INFO_PRINT_FILE_PER_RANK(2, "CASPER Window %s-%d: target async_config\n",
                                 ug_win->info_args.win_name, ug_win->adapt_remote_exed);

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    /* The root user gathers the frequency number on all targets and print in file. */
    if (ug_win->info_args.async_config == CSP_ASYNC_CONFIG_AUTO) {
        CSP_async_stat stat = CSP_adpt_get_async_stat();
        CSP_INFO_PRINT_FILE_PER_RANK(2, "last predict freq: %.1f, %.2lf,%.2lf, %s\n",
                                     CSP_RM[CSP_RM_COMM_FREQ].last_freq * 1.0,
                                     CSP_RM[CSP_RM_COMM_FREQ].last_time,
                                     CSP_RM[CSP_RM_COMM_FREQ].last_interval,
                                     ((stat == CSP_ASYNC_ON) ? "on" : "off"));
    }
#endif

#ifdef CSP_ENABLE_ADAPT_PROF
    CSP_adapt_prof_dump();
#endif

  fn_exit:
    if (tmp_async_gather_buf)
        free(tmp_async_gather_buf);
    return mpi_errno;

  fn_fail:
    CSP_ATTRIBUTE((unused))
        goto fn_exit;
}

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
/* Update global asynchronous synchronization system when the asynchronous states
 * of window processes have been updated (i.e., in every win-collective call).
 * All processes update the local cache, only the root process on each node updates
 * the ghost cache.*/
int CSP_win_gsync_update(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int i, user_nprocs = 0, local_user_rank = 0;
    int *ranks_in_world = NULL;
    CSP_async_stat *async_stats = NULL;
    CSP_gadpt_update_flag flag = CSP_GADPT_UPDATE_LOCAL;

    if (CSP_ENV.async_sched_level < CSP_ASYNC_SCHED_ANYTIME)
        goto fn_exit;

    PMPI_Comm_rank(ug_win->local_user_comm, &local_user_rank);
    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);

    ranks_in_world = CSP_calloc(user_nprocs, sizeof(int));
    async_stats = CSP_calloc(user_nprocs, sizeof(CSP_async_stat));

    for (i = 0; i < user_nprocs; i++) {
        ranks_in_world[i] = ug_win->targets[i].user_world_rank;
        async_stats[i] = ug_win->targets[i].synced_async_stat;
    }

    /* all processes update local cache, only root updates ghost cache. */
    if (local_user_rank == 0)
        flag = CSP_GADPT_UPDATE_GHOST_SYNCED;

    mpi_errno = CSP_gadpt_update(user_nprocs, ranks_in_world, async_stats, flag);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

  fn_exit:
    if (ranks_in_world)
        free(ranks_in_world);
    if (async_stats)
        free(async_stats);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#endif

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

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    /* If runtime scheduling is enabled for this window, we exchange the
     * asynchronous configure with every target, since its value might be different. */
    if (ug_win->info_args.async_config == CSP_ASYNC_CONFIG_AUTO) {
        if (ug_win->info_args.async_config_phases & CSP_ASYNC_CONFIG_PHASE_LOCAL_UPDATE) {
            /* reschedule my status according to runtime monitored data. */
            CSP_adpt_sched_async_stat();
        }

        if (ug_win->info_args.async_config_phases & CSP_ASYNC_CONFIG_PHASE_REMOTE_EXCHANGE) {
            /* read my current status. */
            my_async_stat = CSP_adpt_get_async_stat();

            /* exchange with all targets. */
            tmp_gather_buf = CSP_calloc(user_nprocs, sizeof(MPI_Aint));
            tmp_gather_buf[user_rank] = (MPI_Aint) my_async_stat;

            mpi_errno = PMPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                       tmp_gather_buf, 1, MPI_AINT, ug_win->user_comm);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;

            for (i = 0; i < user_nprocs; i++) {
                ug_win->targets[i].synced_async_stat = (CSP_async_stat) tmp_gather_buf[i];
            }

            /* update gsync system */
            if (CSP_ENV.async_sched_level == CSP_ASYNC_SCHED_ANYTIME) {
                mpi_errno = CSP_win_gsync_update(ug_win);
                if (mpi_errno != MPI_SUCCESS)
                    goto fn_fail;
            }

            ug_win->adapt_remote_exed++;
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
    if (tmp_gather_buf)
        free(tmp_gather_buf);
    return mpi_errno;
  fn_fail:
    CSP_ATTRIBUTE((unused))
        goto fn_exit;
}
