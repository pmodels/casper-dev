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

static CSP_async_stat CSP_MY_ASYNC_STAT = CSP_ASYNC_ON;

void CSP_ra_update_async_stat(CSP_async_config async_config)
{
    switch (async_config) {
    case CSP_ASYNC_CONFIG_ON:
        CSP_MY_ASYNC_STAT = CSP_ASYNC_ON;
        break;
    case CSP_ASYNC_CONFIG_OFF:
        CSP_MY_ASYNC_STAT = CSP_ASYNC_OFF;
        break;
    case CSP_ASYNC_CONFIG_AUTO:
        /* keep original value. */
        break;
    }
}

/* Get current asynchronous status. */
CSP_async_stat CSP_ra_get_async_stat(void)
{
    return CSP_MY_ASYNC_STAT;
}

/* Internal implementation for scheduling local asynchronous status.
 * In any MPI function, this routine should not be directly called,
 * instead, call CSP_ra_sched_async_stat for immediately scheduling,
 * or call other timed scheduling routine such as CSP_win_timed_gsync_all. */
CSP_async_stat CSP_ra_sched_async_stat_impl(void)
{
    double interval;
    int freq = 0;
    CSP_async_stat old_stat CSP_ATTRIBUTE((unused)) = CSP_MY_ASYNC_STAT;
    char old_stat_name[16] CSP_ATTRIBUTE((unused));

    /* schedule async config by using dynamic frequency */
    interval = PMPI_Wtime() - CSP_RM[CSP_RM_COMM_FREQ].interval_sta;
    freq = (int) (CSP_RM[CSP_RM_COMM_FREQ].time / interval * 100);

    if (freq >= CSP_ENV.async_sched_thr_h) {
        CSP_MY_ASYNC_STAT = CSP_ASYNC_OFF;
    }
    else if (freq <= CSP_ENV.async_sched_thr_l) {
        CSP_MY_ASYNC_STAT = CSP_ASYNC_ON;
    }

#if defined(CSP_DEBUG) || defined(CSP_ADAPT_DEBUG)
    strncpy(old_stat_name, CSP_get_target_async_stat_name(old_stat), 16);
#endif
    CSP_DBG_PRINT(" my async stat: freq=%d(%.4f/%.4f), %s->%s\n",
                  freq, CSP_RM[CSP_RM_COMM_FREQ].time, interval, old_stat_name,
                  CSP_get_target_async_stat_name(CSP_MY_ASYNC_STAT));

    CSP_RM[CSP_RM_COMM_FREQ].last_time = CSP_RM[CSP_RM_COMM_FREQ].time;
    CSP_RM[CSP_RM_COMM_FREQ].last_interval = interval;
    CSP_RM[CSP_RM_COMM_FREQ].last_freq = freq;

    CSP_rm_reset(CSP_RM_COMM_FREQ);

    return CSP_MY_ASYNC_STAT;
}

/* Immediately reschedule local asynchronous status according to runtime
 * profiling data.
 * Note that we separate rescheduling and getting functions in order to
 * allow processes to locally reschedule once, and remotely exchange for
 * different windows multiple-times with the same status. */
void CSP_ra_sched_async_stat(void)
{
    /* For anytime level scheduling, the status should be automatically
     * updated by other routines at set interval. */
    if (CSP_ENV.async_sched_level == CSP_ASYNC_SCHED_ANYTIME)
        return;

    CSP_ra_sched_async_stat_impl();
}
#endif
