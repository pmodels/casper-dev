/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

/* This benchmark evaluates the threshold that requiring asynchronous progress
 * in lockall epoch using 2 processes. Rank 0 performs lockall-accumulate-flush-unlockall,
 * and rank 1 performs compute(busy wait)-test(poll MPI progress).*/

#define SIZE 4
#define SLEEP_TIME 100  //us
#define SKIP 100
#define ITER 10000

MPI_Win win;
double *winbuf, locbuf[SIZE];
int rank, nprocs;
int NOP = 100;
int min_time = SLEEP_TIME, max_time = SLEEP_TIME, iter_time = 2;

static void usleep_by_count(unsigned long us)
{
    double start = MPI_Wtime() * 1000 * 1000;
    while (MPI_Wtime() * 1000 * 1000 - start < us);
    return;
}

static int run_test(int time, const char *config)
{
    int i, x, errs = 0;
    int dst, src;
    double t0, t_total = 0.0;
    MPI_Request request;
    MPI_Status status;
    int buf[1];
    int flag = 0;

    if (rank == 0) {
        dst = 1;
        buf[0] = 99;
        MPI_Win_lock_all(0, win);

        for (x = 0; x < SKIP; x++) {
            for (i = 0; i < NOP; i++)
                MPI_Accumulate(&locbuf[0], 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, MPI_SUM, win);
            MPI_Win_flush_all(win);
        }
        MPI_Send(buf, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }
    else {
        src = 0;
        buf[0] = 0;
        MPI_Irecv(buf, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);    /* just poll progress when warming up */

        /* issue for real run */
        MPI_Irecv(buf, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &request);
    }

    t0 = MPI_Wtime();
    for (x = 0; x < ITER; x++) {
        // rank 0 does RMA communication
        if (rank == 0) {
            for (i = 0; i < NOP; i++)
                MPI_Accumulate(&locbuf[0], 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, MPI_SUM, win);
            MPI_Win_flush_all(win);
        }
        // rank 1 does sleep and test
        else {
            usleep_by_count(time);
            MPI_Test(&request, &flag, &status);
        }
    }

    t_total += MPI_Wtime() - t0;
    t_total = t_total / ITER * 1000 * 1000;     /* us */

    if (rank == 0) {
        MPI_Win_unlock_all(win);
        MPI_Send(buf, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }
    else {
        if (!flag)
            MPI_Wait(&request, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
#ifdef ENABLE_CSP
        fprintf(stdout,
                "casper-%s: comp_size %d num_op %d nprocs %d total_time %.2lf %.1f\n",
                config, time, NOP, nprocs, t_total, time / t_total * 100);
#else
        fprintf(stdout,
                "orig: comp_size %d num_op %d nprocs %d total_time %.2lf %.1f\n",
                time, NOP, nprocs, t_total, time / t_total * 100);
#endif
    }

    return errs;
}

void run_with_async_config(const char *config)
{
    int time = 0;
    MPI_Info win_info = MPI_INFO_NULL;
    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "epoch_type", (char *) "lockall");
    MPI_Info_set(win_info, (char *) "async_config", config);

    // size in byte
    MPI_Win_allocate(sizeof(double), sizeof(double), win_info, MPI_COMM_WORLD, &winbuf, &win);

    /* reset window */
    MPI_Win_lock_all(0, win);
    winbuf[0] = 0.0;
    MPI_Win_unlock_all(win);
    MPI_Barrier(MPI_COMM_WORLD);

    for (time = min_time; time <= max_time; time *= iter_time) {
        run_test(time, config);
    }

    if (win_info != MPI_INFO_NULL)
        MPI_Info_free(&win_info);

    MPI_Win_free(&win);
}

int main(int argc, char *argv[])
{
    int i;

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

    if (2 != nprocs) {
        if (rank == 0)
            fprintf(stderr, "Please run using 2 processes\n");
        goto exit;
    }

    for (i = 0; i < SIZE; i++) {
        locbuf[i] = (i + 1) * 0.5;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    run_with_async_config("on");

    /* original version only need once */
#ifdef ENABLE_CSP
    MPI_Barrier(MPI_COMM_WORLD);
    run_with_async_config("off");
#endif

  exit:

    MPI_Finalize();

    return 0;
}
