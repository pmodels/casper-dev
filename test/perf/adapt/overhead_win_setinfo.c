/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <cperf.h>

/* This benchmark measures the overhead of win_set_info(symmetric) with different
 * adaptation modes.
 *
 * Note that to avoid performance difference caused by adaptation, this test
 * should not add computing delay, and should use only 1 user process and 1 ghost
 * process on every node to avoid work overload issue.*/

#define SKIP 100
#define ITER 1000
int rank, nprocs;
double *winbuf, *locbuf = NULL;
MPI_Win win;
int size = 16;
char async_sched_level_str[CPERF_ENVVAL_MAXLEN] = { 0 };
char cperf_info_async_config[CPERF_ENVVAL_MAXLEN] = { 0 };

static void issue_ops(MPI_Info async_info)
{
    int i, dst;
    MPI_Win_set_info(win, async_info);
    MPI_Win_lock_all(0, win);

    for (dst = 0; dst < nprocs; dst++) {
        for (i = 1; i < size; i++) {
            MPI_Accumulate(&locbuf[i], 1, MPI_DOUBLE, dst, rank, 1, MPI_DOUBLE, MPI_SUM, win);
        }
    }

    MPI_Win_unlock_all(win);
    MPI_Barrier(MPI_COMM_WORLD);
}

static void run_test(const char *config_val)
{
    int x;
    double t0, t1, t_setinfo;
    MPI_Info async_info = MPI_INFO_NULL;

    MPI_Info_create(&async_info);
    if (config_val != NULL) {
        MPI_Info_set(async_info, (char *) "symmetric", "true");
        MPI_Info_set(async_info, (char *) "async_config", config_val);
    }

    for (x = 0; x < SKIP; x++) {
        issue_ops(async_info);
    }

    t0 = MPI_Wtime();
    for (x = 0; x < ITER; x++) {
        issue_ops(async_info);
    }
    t1 = MPI_Wtime();
    t_setinfo = (t1 - t0) / ITER;

    if (rank == 0)
        fprintf(stdout, "%s%s, nproc %d config %s set_info %lf\n",
#ifdef ENABLE_CSP
                "dac-", async_sched_level_str,
#else
                "original", "",
#endif
                nprocs, config_val == NULL ? "none" : config_val, t_setinfo);

    if (async_info != MPI_INFO_NULL)
        MPI_Info_free(&async_info);
}

int main(int argc, char *argv[])
{
    MPI_Info win_info = MPI_INFO_NULL;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nprocs < 2) {
        fprintf(stderr, "Please run using at least 2 processes\n");
        goto exit;
    }

    if (argc > 1)
        size = atoi(argv[1]);

    if (size <= 0) {
        fprintf(stderr, "wrong size %d\n", size);
        goto exit;
    }

    CTEST_perf_read_env("CSP_ASYNC_SCHED_LEVEL", "per-win", &async_sched_level_str);
    CTEST_perf_read_env("CPERF_INFO_ASYNC_CONFIG", "on", &cperf_info_async_config);

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "epoch_type", "lockall");

    /* size in byte */
    MPI_Win_allocate(sizeof(double) * size, sizeof(double), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &winbuf, &win);

    locbuf = malloc(sizeof(double) * size);
    memset(locbuf, 0, sizeof(double) * size);

    MPI_Barrier(MPI_COMM_WORLD);
    run_test(cperf_info_async_config);

  exit:
    if (win_info != MPI_INFO_NULL)
        MPI_Info_free(&win_info);
    if (locbuf)
        free(locbuf);

    MPI_Win_free(&win);
    MPI_Finalize();

    return 0;
}
