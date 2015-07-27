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

static CSP_target_async_stat CSP_MY_ASYNC_STAT = CSP_TARGET_ASYNC_NONE;

void CSP_ra_update_async_stat(CSP_async_config async_config)
{
    switch (async_config) {
    case CSP_ASYNC_CONFIG_ON:
        CSP_MY_ASYNC_STAT = CSP_TARGET_ASYNC_ON;
        break;
    case CSP_ASYNC_CONFIG_OFF:
        CSP_MY_ASYNC_STAT = CSP_TARGET_ASYNC_OFF;
        break;
    case CSP_ASYNC_CONFIG_AUTO:
        /* keep original value. */
        break;
    }
}

CSP_target_async_stat CSP_ra_sched_async_stat()
{
    double interval;
    int freq = 0;

    CSP_target_async_stat old_stat CSP_ATTRIBUTE((unused)) = CSP_MY_ASYNC_STAT;

    /* schedule async config by using dynamic frequency */
    interval = PMPI_Wtime() - CSP_RM[CSP_RM_COMM_FREQ].interval_sta;
    freq = (int) (CSP_RM[CSP_RM_COMM_FREQ].time / interval * 100);

    if (freq >= CSP_ENV.async_sched_thr_h) {
        CSP_MY_ASYNC_STAT = CSP_TARGET_ASYNC_OFF;
    }
    else if (freq <= CSP_ENV.async_sched_thr_l) {
        CSP_MY_ASYNC_STAT = CSP_TARGET_ASYNC_ON;
    }

    CSP_DBG_PRINT(" my async stat: freq=%d(%.4f/%.4f), %s->%s\n ",
                  freq, CSP_RM[CSP_RM_COMM_FREQ].time, interval,
                  CSP_get_target_async_stat_name(old_stat),
                  CSP_get_target_async_stat_name(CSP_MY_ASYNC_STAT));

    CSP_RM[CSP_RM_COMM_FREQ].last_time = CSP_RM[CSP_RM_COMM_FREQ].time;
    CSP_RM[CSP_RM_COMM_FREQ].last_interval = interval;
    CSP_RM[CSP_RM_COMM_FREQ].last_freq = freq;

    CSP_rm_reset(CSP_RM_COMM_FREQ);

    return CSP_MY_ASYNC_STAT;
}

#endif
