/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

/* This benchmark evaluates the threshold that benefiting from asynchronous
 * progress in passive all to all communication wich increasing delay time.
 * Every one performs lockall-ACC-compute-ACC-unlockall.*/

#define SLEEP_TIME 100  //us
#define SKIP 100
#define ITER_S 5000
#define ITER_M 2000
#define ITER_L 500
#define ITER_LL 100

MPI_Win win;
double *winbuf, locbuf = 99.0;
int rank, nprocs;
int NOP = 100;
int min_time = SLEEP_TIME, max_time = SLEEP_TIME, iter_time = 2;
int ITER = ITER_S;

static void usleep_by_count(unsigned long us)
{
    double start = MPI_Wtime() * 1000 * 1000;
    while (MPI_Wtime() * 1000 * 1000 - start < us);
    return;
}

static void set_iter(void)
{
    if (nprocs < 24) {
        ITER = ITER_S;
    }
    else if (nprocs >= 24 && nprocs < 96) {
        ITER = ITER_M;
    }
    else if (nprocs >= 96 && nprocs < 192) {
        ITER = ITER_L;
    }
    else {
        ITER = ITER_LL;
    }
}

static int run_test(int time)
{
    int i, x, errs = 0;
    int dst, src;
    double t0, t_total = 0.0, avg_t_total = 0.0;
    MPI_Request request;
    MPI_Status status;
    int buf[1];
    int flag = 0;

    MPI_Win_lock_all(0, win);
    for (x = 0; x < SKIP; x++) {
        for (dst = 0; dst < nprocs; dst++) {
            MPI_Accumulate(&locbuf, 1, MPI_DOUBLE, dst, rank, 1, MPI_DOUBLE, MPI_SUM, win);
        }
        MPI_Win_flush_all(win);
    }
    MPI_Win_unlock_all(win);

    t0 = MPI_Wtime();
    MPI_Win_lock_all(0, win);

    for (x = 0; x < ITER; x++) {
        for (dst = 0; dst < nprocs; dst++) {
            MPI_Accumulate(&locbuf, 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, MPI_SUM, win);
            MPI_Win_flush_all(win);
        }

        usleep_by_count(time);

        for (dst = 0; dst < nprocs; dst++) {
            for (i = 0; i < NOP; i++)
                MPI_Accumulate(&locbuf, 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, MPI_SUM, win);
            MPI_Win_flush_all(win);
        }
    }

    MPI_Win_unlock_all(win);
    t_total += MPI_Wtime() - t0;
    t_total = t_total / ITER * 1000 * 1000;     /* us */

    MPI_Reduce(&t_total, &avg_t_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    avg_t_total = avg_t_total / nprocs;

    if (rank == 0) {
#ifdef ENABLE_CSP
        const char *async = getenv("CSP_ASYNC_CONFIG");
        const char *nh = getenv("CSP_NG");

        fprintf(stdout,
                "casper-%s-nh%s: comp_size %d num_op %d nprocs %d total_time %.2lf %.1f\n",
                async, nh, time, NOP, nprocs, avg_t_total, time / avg_t_total * 100);
#else
        fprintf(stdout,
                "orig: comp_size %d num_op %d nprocs %d total_time %.2lf %.1f\n",
                time, NOP, nprocs, avg_t_total, time / avg_t_total * 100);
#endif
    }

    return errs;
}

int main(int argc, char *argv[])
{
    int i;
    int time = 0;
    MPI_Info win_info = MPI_INFO_NULL;

    MPI_Init(&argc, &argv);

    if (argc >= 4) {
        min_time = atoi(argv[1]);
        max_time = atoi(argv[2]);
        iter_time = atoi(argv[3]);
    }
    if (argc >= 5) {
        NOP = atoi(argv[4]);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (2 > nprocs) {
        if (rank == 0)
            fprintf(stderr, "Please run using at least 2 processes\n");
        goto exit;
    }


    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "epoch_type", (char *) "lockall");

    // size in byte
    MPI_Win_allocate(sizeof(double), sizeof(double), win_info, MPI_COMM_WORLD, &winbuf, &win);

    /* reset window */
    MPI_Win_lock_all(0, win);
    winbuf[0] = 0.0;
    MPI_Win_unlock_all(win);

    set_iter();
    MPI_Barrier(MPI_COMM_WORLD);

    for (time = min_time; time <= max_time; time *= iter_time) {
        run_test(time);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Info_free(&win_info);
    MPI_Win_free(&win);

  exit:

    MPI_Finalize();

    return 0;
}
