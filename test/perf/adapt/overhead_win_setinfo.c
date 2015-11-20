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
 * adaptation modes. */

#define SKIP 100
#define ITER 500000
#define WIN_SIZE 1024

int rank, nprocs;
double *winbuf, *locbuf = NULL;
MPI_Win win;
char async_sched_level_str[CPERF_ENVVAL_MAXLEN] = { 0 };
char cperf_info_async_config[CPERF_ENVVAL_MAXLEN] = { 0 };

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
        MPI_Win_set_info(win, async_info);
    }

    t0 = MPI_Wtime();
    for (x = 0; x < ITER; x++) {
        MPI_Win_set_info(win, async_info);
    }
    t1 = MPI_Wtime();
    t_setinfo = (t1 - t0) / ITER * 1000 * 1000; /*us */

    if (rank == 0)
        fprintf(stdout, "%s%s, nproc %d config %s set_info %lf\n",
#ifdef ENABLE_CSP
                "csp-", async_sched_level_str,
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

    CTEST_perf_read_env("CSP_ASYNC_SCHED_LEVEL", "per-win", &async_sched_level_str);
    CTEST_perf_read_env("CPERF_INFO_ASYNC_CONFIG", "on", &cperf_info_async_config);

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "epoch_type", "lockall");

    /* size in byte */
    MPI_Win_allocate(sizeof(double) * WIN_SIZE, sizeof(double), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &winbuf, &win);

    locbuf = malloc(sizeof(double) * WIN_SIZE);
    memset(locbuf, 0, sizeof(double) * WIN_SIZE);

    MPI_Barrier(MPI_COMM_WORLD);
    run_test(cperf_info_async_config);

    if (win_info != MPI_INFO_NULL)
        MPI_Info_free(&win_info);
    if (locbuf)
        free(locbuf);

    MPI_Win_free(&win);
    MPI_Finalize();

    return 0;
}
