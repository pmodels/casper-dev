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

static CSP_async_stat CSP_MY_ASYNC_STAT = CSP_ASYNC_NONE;

void CSP_adpt_update_async_stat(CSP_async_config async_config)
{
    switch (async_config) {
    case CSP_ASYNC_CONFIG_ON:
        CSP_MY_ASYNC_STAT = CSP_ASYNC_ON;
        break;
    case CSP_ASYNC_CONFIG_OFF:
        CSP_MY_ASYNC_STAT = CSP_ASYNC_OFF;
        break;
    case CSP_ASYNC_CONFIG_AUTO:
        /* set to default state for initialization. Otherwise, keep original value. */
        if (CSP_MY_ASYNC_STAT == CSP_ASYNC_NONE) {
            CSP_MY_ASYNC_STAT = CSP_ENV.async_auto_stat;
        }
        break;
    }
}

/* Get current asynchronous status. */
CSP_async_stat CSP_adpt_get_async_stat(void)
{
    return CSP_MY_ASYNC_STAT;
}

/* Schedule local asynchronous status.*/
CSP_async_stat CSP_adpt_sched_async_stat(void)
{
    double interval;
    int freq = 0;
    CSP_async_stat old_stat = CSP_MY_ASYNC_STAT;
    char old_stat_name[16] CSP_ATTRIBUTE((unused));

    /* schedule async config by using dynamic frequency */
    interval = PMPI_Wtime() - CSP_RM[CSP_RM_COMM_FREQ].interval_sta;
    freq = (int) (CSP_RM[CSP_RM_COMM_FREQ].time / interval * 100);

    /* do not change state if interval is too short */
    if (interval < CSP_ENV.adpt_sched_interval)
        return CSP_MY_ASYNC_STAT;

    if (freq >= CSP_ENV.adpt_sched_thr_h) {
        CSP_MY_ASYNC_STAT = CSP_ASYNC_OFF;
    }
    else if (freq <= CSP_ENV.adpt_sched_thr_l) {
        CSP_MY_ASYNC_STAT = CSP_ASYNC_ON;
    }

#if defined(CSP_DEBUG) || defined(CSP_ADAPT_DEBUG)
    strncpy(old_stat_name, CSP_get_target_async_stat_name(old_stat), 16);
#endif
    CSP_ADAPT_DBG_PRINT(" my async stat: freq=%d(%.4f/%.4f), %s->%s\n",
                        freq, CSP_RM[CSP_RM_COMM_FREQ].time, interval, old_stat_name,
                        CSP_get_target_async_stat_name(CSP_MY_ASYNC_STAT));

    CSP_RM[CSP_RM_COMM_FREQ].last_time = CSP_RM[CSP_RM_COMM_FREQ].time;
    CSP_RM[CSP_RM_COMM_FREQ].last_interval = interval;
    CSP_RM[CSP_RM_COMM_FREQ].last_freq = freq;

    /* update ghost caches if my status is changed */
    if (CSP_ENV.async_sched_level == CSP_ASYNC_SCHED_ANYTIME && old_stat != CSP_MY_ASYNC_STAT) {
        int user_rank = 0;
        PMPI_Comm_rank(CSP_COMM_USER_WORLD, &user_rank);
        CSP_gadpt_update(1, &user_rank, &CSP_MY_ASYNC_STAT, CSP_GADPT_UPDATE_GHOST);
    }

    CSP_rm_reset(CSP_RM_COMM_FREQ);

    return CSP_MY_ASYNC_STAT;
}
#endif
