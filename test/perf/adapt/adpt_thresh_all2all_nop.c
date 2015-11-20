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
 * progress in passive all to all communication with increasing number of operations,
 * this benchmark can be also used for the test with increasing number of processes.
 * Every one performs lockall-RMA-compute-unlockall.
 *
 * It should run with per-coll level adaptation, to ensure both on/off use AM for local
 * processes.*/

#define SLEEP_TIME 100  //us
#define SKIP 100
#define ITER_S 500
#define ITER_M 500
#define ITER_L 500
#define ITER_LL 500
#define NOP 1

MPI_Win win;
double *winbuf, locbuf = 99.0;
int rank, nprocs;

int nop_min = NOP, nop_max = NOP, nop_iter = 2;
int ITER = ITER_S;

#if defined(TEST_RMA_OP_GET)
const char *op_name = "get";
#elif defined(TEST_RMA_OP_PUT)
const char *op_name = "put";
#else
const char *op_name = "acc";
#endif

static void usleep_by_count(unsigned long us)
{
    double start = MPI_Wtime() * 1000 * 1000;
    while (MPI_Wtime() * 1000 * 1000 - start < us);
    return;
}

static void set_iter(void)
{
    if (nprocs < 48) {
        ITER = ITER_S;
    }
    else if (nprocs >= 48 && nprocs < 120) {
        ITER = ITER_M;
    }
    else if (nprocs >= 120 && nprocs < 240) {
        ITER = ITER_L;
    }
    else {
        ITER = ITER_LL;
    }
}

static double run_test(int time, int nop)
{
    int i, x;
    int dst;
    double t0, t_total = 0.0;

    MPI_Win_lock_all(0, win);
    for (x = 0; x < SKIP; x++) {
        for (dst = 0; dst < nprocs; dst++) {
#if defined(TEST_RMA_OP_GET)
            MPI_Get(&locbuf, 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, win);
#elif defined(TEST_RMA_OP_PUT)
            MPI_Put(&locbuf, 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, win);
#else
            MPI_Accumulate(&locbuf, 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, MPI_SUM, win);
#endif
            MPI_Win_flush(dst, win);
        }
    }

    t0 = MPI_Wtime();
    for (x = 0; x < ITER; x++) {
        for (dst = 0; dst < nprocs; dst++) {
            for (i = 0; i < nop; i++) {
#if defined(TEST_RMA_OP_GET)
                MPI_Get(&locbuf, 1, MPI_DOUBLE, dst, i, 1, MPI_DOUBLE, win);
#elif defined(TEST_RMA_OP_PUT)
                MPI_Put(&locbuf, 1, MPI_DOUBLE, dst, i, 1, MPI_DOUBLE, win);
#else
                MPI_Accumulate(&locbuf, 1, MPI_DOUBLE, dst, i, 1, MPI_DOUBLE, MPI_SUM, win);
#endif
                MPI_Win_flush(dst, win);
            }
        }

        usleep_by_count(time);
    }
    t_total += MPI_Wtime() - t0;
    t_total = t_total / ITER * 1000 * 1000;     /* us */

    MPI_Win_unlock_all(win);

    return t_total;
}

static double on_t_total = 0.0;

static void run_with_async_config(const char *config, int time, int nop)
{
    MPI_Info win_info = MPI_INFO_NULL;
    double t_total = 0.0, avg_t_total = 0.0;
    int i;

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "epoch_type", (char *) "lockall");
    MPI_Info_set(win_info, (char *) "async_config", config);
/*  MPI_Info_set(win_info, (char *) "alloc_shm", "true"); */

    // size in byte
    MPI_Win_allocate(sizeof(double) * nop, sizeof(double), win_info, MPI_COMM_WORLD, &winbuf, &win);

    /* reset window */
    MPI_Win_lock_all(0, win);
    for (i = 0; i < nop; i++)
        winbuf[i] = 0.0;
    MPI_Win_unlock_all(win);

    t_total = run_test(time, nop);

    MPI_Reduce(&t_total, &avg_t_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    avg_t_total = avg_t_total / nprocs;



    if (rank == 0) {
        const char *nh = getenv("CSP_NG");
        double async_speedup = 1;

        if (config == "off") {
            async_speedup = avg_t_total / on_t_total;
        }
        else {
            on_t_total = avg_t_total;   /* store it for speedup at off */
        }

        fprintf(stdout,
                "csp-%s-%s-nh%s: comp_size %d num_op %d nprocs %d total_time %.2lf freq %.1f sp %.2f\n",
                op_name, config, nh, time, nop, nprocs, avg_t_total,
                (1 - time / avg_t_total) * 100 /*comm freq */ , async_speedup);
    }

    MPI_Info_free(&win_info);
    MPI_Win_free(&win);
}


int main(int argc, char *argv[])
{
    int time = SLEEP_TIME;
    int nop = NOP;

    MPI_Init(&argc, &argv);

    if (argc >= 4) {
        nop_min = atoi(argv[1]);
        nop_max = atoi(argv[2]);
        nop_iter = atoi(argv[3]);
    }
    if (argc >= 5) {
        time = atoi(argv[4]);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (2 > nprocs) {
        if (rank == 0)
            fprintf(stderr, "Please run using at least 2 processes\n");
        goto exit;
    }

    for (nop = nop_min; nop <= nop_max; nop *= nop_iter) {
        set_iter();

        MPI_Barrier(MPI_COMM_WORLD);
        run_with_async_config("on", time, nop);

        MPI_Barrier(MPI_COMM_WORLD);
        run_with_async_config("off", time, nop);
    }

  exit:
    MPI_Finalize();

    return 0;
}
