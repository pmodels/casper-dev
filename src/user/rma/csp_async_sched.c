/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <csp.h>

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED

static CSP_target_async_stat CSP_MY_ASYNC_STAT;

CSP_target_async_stat CSP_sched_my_async_stat()
{
    double time;
    unsigned long long freq = 0;

    CSP_target_async_stat old_stat = CSP_MY_ASYNC_STAT;

    /* schedule async config by using dynamic frequency */
    time = PMPI_Wtime() - CSP_RM[CSP_RM_COMM_FREQ].timer_sta;
    freq = (unsigned long long) (CSP_RM[CSP_RM_COMM_FREQ].cnt / time);

    if (freq >= CSP_ENV.async_sched_thr_h) {
        CSP_MY_ASYNC_STAT = CSP_TARGET_ASYNC_OFF;
    }
    else if (freq <= CSP_ENV.async_sched_thr_l) {
        CSP_MY_ASYNC_STAT = CSP_TARGET_ASYNC_ON;
    }

    CSP_DBG_PRINT(" my async stat: cnt=%lld, time=%.4f, freq =%lld, %s->%s\n ",
                  CSP_RM[CSP_RM_COMM_FREQ].cnt, time, freq,
                  CSP_get_target_async_stat_name(old_stat),
                  CSP_get_target_async_stat_name(CSP_MY_ASYNC_STAT));

    if (old_stat != CSP_MY_ASYNC_STAT) {
        CSP_INFO_PRINT(2, "Sched my async stat: freq =%lld, %s->%s \n",
                       freq, CSP_get_target_async_stat_name(old_stat),
                       CSP_get_target_async_stat_name(CSP_MY_ASYNC_STAT));
    }

    CSP_rm_reset(CSP_RM_COMM_FREQ);

    return CSP_MY_ASYNC_STAT;
}

#endif
