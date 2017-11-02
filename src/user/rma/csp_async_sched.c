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

#ifdef NWCHEM_ADAPT_TPI_DBG
extern void tpi_dbg_print_file_(const char *text);
extern int dbg_print_file_opened;
#endif

/* Schedule local asynchronous status.*/
CSP_async_stat CSP_adpt_sched_async_stat(void)
{
    double interval, time;
    int freq = 0;
    CSP_async_stat old_stat = CSP_MY_ASYNC_STAT;
    char old_stat_name[16] CSP_ATTRIBUTE((unused));

    /* schedule async config by using dynamic frequency */
    interval = CSP_time_diff(CSP_RM[CSP_RM_COMM_FREQ].interval_sta, CSP_time());
    /* do not change state if interval is too short */
    if (interval < CSP_ENV.adpt_sched_interval)
        return CSP_MY_ASYNC_STAT;

    time = CSP_time_todouble(CSP_RM[CSP_RM_COMM_FREQ].time);
    freq = (int) (time / interval * 100);

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
                        freq, time, interval, old_stat_name,
                        CSP_get_target_async_stat_name(CSP_MY_ASYNC_STAT));
#ifdef NWCHEM_ADAPT_TPI_DBG
    if (dbg_print_file_opened == 1) {
        char temp_str[512];
        memset(temp_str, 0, sizeof(temp_str));
        strncpy(old_stat_name, CSP_get_target_async_stat_name(old_stat), 16);
        sprintf(temp_str, " my async stat: freq=%d(%.4f/%.4f), %s->%s",
                freq, time, interval, old_stat_name,
                CSP_get_target_async_stat_name(CSP_MY_ASYNC_STAT));
        tpi_dbg_print_file_((const char *) temp_str);
    }
#endif

    CSP_RM[CSP_RM_COMM_FREQ].last_time = time;
    CSP_RM[CSP_RM_COMM_FREQ].last_interval = interval;
    CSP_RM[CSP_RM_COMM_FREQ].last_freq = freq;
    CSP_RM[CSP_RM_COMM_FREQ].reported_flag = 0;

    /* update ghost caches if my status is changed */
    if (CSP_ENV.async_sched_level == CSP_ASYNC_SCHED_ANYTIME && old_stat != CSP_MY_ASYNC_STAT) {
        CSP_gadpt_upload_local(CSP_MY_ASYNC_STAT);
    }

    CSP_rm_reset(CSP_RM_COMM_FREQ);

    return CSP_MY_ASYNC_STAT;
}
#endif
