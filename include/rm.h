/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef RM_H_
#define RM_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef CSP_ENABLE_RUNTIME_MONITOR

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    CSP_RM_COMM_FREQ,
    CSP_RM_MAX_TYPE,
} CSP_rm_type;

typedef struct {
    unsigned long long cnt;
    double timer_sta;
} CSP_rm;

/* local runtime monitor */
extern CSP_rm CSP_RM[CSP_RM_MAX_TYPE];

static inline void CSP_rm_count(CSP_rm_type type)
{
    CSP_RM[type].cnt++;
}

static inline void CSP_rm_reset(CSP_rm_type type)
{
    CSP_RM[type].cnt = 0;
    CSP_RM[type].timer_sta = PMPI_Wtime();
}

static inline void CSP_rm_reset_all()
{
    CSP_rm_reset(CSP_RM_COMM_FREQ);
}
#else
#define CSP_rm_count(type) {/*do nothing */}
#define CSP_rm_reset(type) {/*do nothing */}
#define CSP_rm_reset_all() {/*do nothing */}

#endif /* end of CSP_ENABLE_RUNTIME_MONITOR */

#endif /* RM_H_ */
