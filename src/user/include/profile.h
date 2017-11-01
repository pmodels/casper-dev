/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef PROFILE_H_
#define PROFILE_H_
#include "mpi.h"

#ifdef ENABLE_PROFILE
enum CSP_profile_func {
    PROF_MPI_GET,
    PROF_MPI_ACCUMULATE,
    PROF_MPI_WIN_FLUSH_ALL,
    PROF_MPI_WIN_FLUSH,
    PROF_MPI_WIN_FLUSH_ALL_GADPT,
    PROF_MPI_WIN_FLUSH_GADPT,
    PROF_UTIL_COLL_ASYNC_CONFIG,
    PROF_MAX_NUM_PROFILE_FUNC
};

extern int CSP_prof_counters[PROF_MAX_NUM_PROFILE_FUNC];
extern double CSP_prof_timings[PROF_MAX_NUM_PROFILE_FUNC];

extern void CSP_profile_destroy(void);
extern void CSP_profile_init(void);

#define CSP_PROFILE_INIT CSP_profile_init
#define CSP_PROFILE_DESTROY CSP_profile_destroy

#define CSP_FUNC_PROFILE_TIMING_START(func) double _profile_##func##_time_start = MPI_Wtime();
#define CSP_FUNC_PROFILE_TIMING_END(func) {   \
    CSP_assert(PROF_##func < PROF_MAX_NUM_PROFILE_FUNC);             \
    if (PROF_##func >= 0 && PROF_##func < PROF_MAX_NUM_PROFILE_FUNC) \
        CSP_prof_timings[PROF_##func] += MPI_Wtime() - _profile_##func##_time_start;   \
}

#define CSP_FUNC_PROFILE_COUNTER_INC(func) {  \
        CSP_assert(PROF_##func < PROF_MAX_NUM_PROFILE_FUNC);             \
        if (PROF_##func >= 0 && PROF_##func < PROF_MAX_NUM_PROFILE_FUNC) \
            CSP_prof_counters[PROF_##func]++;   \
    }

extern void CSP_profile_reset_counter(void);
extern void CSP_profile_reset_timing(void);
extern void CSP_profile_print_timing(char *name);
extern void CSP_profile_print_counter(char *name);
#else
#define CSP_PROFILE_INIT()
#define CSP_PROFILE_DESTROY()
#define CSP_FUNC_PROFILE_TIMING_START(func)
#define CSP_FUNC_PROFILE_TIMING_END(func)
#define CSP_FUNC_PROFILE_COUNTER_INC(func)

#endif
#endif /* PROFILE_H_ */
