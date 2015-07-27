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
                    if (ug_win->targets[i].async_stat == CSP_TARGET_ASYNC_ON) {
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
