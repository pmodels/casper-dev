/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

/* This benchmark measures adaptation for computation intensive phase with
 * setting the default asynchronous configuration to off. The number of window
 * creation and the number of operations during each window can be set via input
 * parameter.*/

#define ITER 1000
#define SKIP 10

double *winbuf = NULL;
double locbuf[1];
int rank = 0, nprocs = 0;
int shm_rank = 0, shm_nprocs = 0;
MPI_Win win = MPI_WIN_NULL;
#ifdef ENABLE_CSP
#include <casper.h>
#endif

int NOP_MAX = 16, NOP_MIN = 1, NOP = 1, NOP_ITER = 2;
int NWIN = 1, ITER_WIN = ITER;
int NPHASE = 1, ITER_PHASE = ITER;
unsigned long SLEEP_TIME = 100, MAX_SLEEP_TIME = 100;   /* us */
int *target_nops = NULL;

static int target_computation()
{
    double start = MPI_Wtime() * 1000 * 1000;
    while (MPI_Wtime() * 1000 * 1000 - start < SLEEP_TIME);
    return 0;
}

#if defined(TEST_RMA_OP_ACC)
#define MPI_RMA_OP(x, dst, i) { \
    MPI_Accumulate(&locbuf[0], 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, MPI_SUM, win); \
    }
#elif defined(TEST_RMA_OP_PUT)
#define MPI_RMA_OP(x, dst, i) { \
    MPI_Put(&locbuf[0], 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, win); \
    }
#else
#define MPI_RMA_OP(x, dst, i) { \
    MPI_Get(&locbuf[0], 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, win); \
    }
#endif

static int run_iteration()
{
    int i, x, errs = 0;
    int dst;
    double t0, avg_total_time = 0.0, t_total = 0.0;
    MPI_Info win_info = MPI_INFO_NULL;

    ITER_WIN = ITER / NWIN;
    ITER_PHASE = ITER / NPHASE;

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "epoch_type", (char *) "lockall");

    /* origin process */
    for (x = 0; x < SKIP; x++) {
        MPI_Win_allocate(sizeof(double), sizeof(double), win_info, MPI_COMM_WORLD, &winbuf, &win);
        MPI_Win_lock_all(0, win);
        for (dst = 0; dst < nprocs; dst++)
            MPI_RMA_OP(x, dst, 0);
        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    t0 = MPI_Wtime();

    MPI_Win_allocate(sizeof(double), sizeof(double), win_info, MPI_COMM_WORLD, &winbuf, &win);

    for (x = 0; x < ITER; x++) {
        if (x % ITER_WIN == 0) {
            MPI_Win_free(&win);
            MPI_Win_allocate(sizeof(double), sizeof(double), win_info, MPI_COMM_WORLD, &winbuf,
                             &win);
        }

        if (x % ITER_PHASE == 0) {
            SLEEP_TIME = (SLEEP_TIME > 0) ? 0 : MAX_SLEEP_TIME;
            /* printf("set SLEEP_TIME=%lu (x %d ITER_PHASE %d)\n", SLEEP_TIME, x, ITER_PHASE); */
        }

        MPI_Win_lock_all(0, win);

        for (dst = 0; dst < nprocs; dst++) {
            for (i = 0; i < NOP; i++)
                MPI_RMA_OP(x, dst, i);
        }

        MPI_Win_flush_all(win);

        target_computation();

        for (dst = 0; dst < nprocs; dst++) {
            for (i = 0; i < NOP; i++)
                MPI_RMA_OP(x, dst, i);
        }

        MPI_Win_unlock_all(win);
    }

    MPI_Win_free(&win);

    t_total = (MPI_Wtime() - t0) * 1000 * 1000; /*us */
    t_total /= ITER;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&t_total, &avg_total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        avg_total_time = avg_total_time / nprocs;       /* us */

#ifdef ENABLE_CSP
        const char *async = getenv("CSP_ASYNC_CONFIG");
        const char *sched = getenv("CSP_ASYNC_SCHED_LEVEL");
        const char *sched_l = getenv("CSP_RUNTIME_ASYNC_SCHED_THR_L");
        const char *sched_h = getenv("CSP_RUNTIME_ASYNC_SCHED_THR_H");
        const char *gsync_int = getenv("CSP_RUNTIME_ASYNC_TIMED_GSYNC_INT");
        fprintf(stdout,
                "casper-%s-%s: H %s L %s G %s iter %d nprocs %d comp_size %lu nwin %d nphase %d num_op %d total_time %.2lf\n",
                async, sched, sched_l, sched_h, gsync_int, ITER, nprocs, MAX_SLEEP_TIME, NWIN, NPHASE, NOP,
                avg_total_time);
#else
        fprintf(stdout,
                "orig: iter %d nprocs %d comp_size %lu nwin %d nphase %d num_op %d total_time %.2lf\n",
                ITER, nprocs, MAX_SLEEP_TIME, NWIN, NPHASE, NOP, avg_total_time);
#endif
    }

    MPI_Info_free(&win_info);

    return errs;
}

int main(int argc, char *argv[])
{
    MPI_Comm shm_comm = MPI_COMM_NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_rank(shm_comm, &shm_rank);
    MPI_Comm_size(shm_comm, &shm_nprocs);

    if (nprocs < 2) {
        fprintf(stderr, "Please run using at least two processes\n");
        goto exit;
    }

    if (argc >= 4) {
        NOP_MIN = atoi(argv[1]);
        NOP_MAX = atoi(argv[2]);
        NOP_ITER = atoi(argv[3]);
    }
    if (argc >= 5) {
        MAX_SLEEP_TIME = atoi(argv[4]);
    }
    if (argc >= 6) {
        NWIN = atoi(argv[5]);
    }
    if (argc >= 7) {
        NPHASE = atoi(argv[6]);
    }

    locbuf[0] = (rank + 1) * 1.0;
    for (NOP = NOP_MIN; NOP <= NOP_MAX; NOP *= NOP_ITER) {
        run_iteration();
    }

  exit:
    if (shm_comm)
        MPI_Comm_free(&shm_comm);

    MPI_Finalize();

    return 0;
}
