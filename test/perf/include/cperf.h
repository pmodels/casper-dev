/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef CTEST_PERF_ADPAT_H_
#define CTEST_PERF_ADPAT_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CPERF_ENVVAL_MAXLEN 64
static inline void CTEST_perf_read_env(const char *param_name,
                                       const char *default_val, void *param_val)
{
    char *val;

    if (param_name == NULL || param_val == NULL)
        return;

    if (default_val != NULL)
        strncpy((param_val), default_val, CPERF_ENVVAL_MAXLEN);

    val = getenv(param_name);
    if (val && strlen(val)) {
        strncpy((param_val), val, CPERF_ENVVAL_MAXLEN);
    }
}

#endif /* CTEST_PERF_ADPAT_H_ */
