/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

/* This benchmark evaluates the threshold that benefiting from asynchronous
 * progress in passive all to all communication with increasing size of operations.
 * Every one performs lockall-RMA-flush-compute-unlockall.
 *
 * It should run with per-coll level adaptation, to ensure both on/off use AM for
 * local processes.*/

#define SLEEP_TIME 100  //us
#define SKIP 10
#define ITER_S 500
#define ITER_M 500
#define ITER_L 500
#define ITER_LL 500
#define NOP 1
#define OPSIZE 2

/* #define CHECK */

/* 3D matrix on target window and local 3D submatrix */
#define DLEN 8
#define SUB_DLEN 4
int WINSIZE = (DLEN * DLEN * (OPSIZE * 2)), BUFSIZE = (SUB_DLEN * SUB_DLEN * OPSIZE);

MPI_Win win;
double *winbuf = NULL, *locbuf = NULL;
int rank, nprocs;

int opsize_min = OPSIZE, opsize_max = OPSIZE, opsize_iter = 2;
int ITER = ITER_S;

/* #define CHECK */
#ifdef CHECK
#include <ctest.h>
double *checkbuf = NULL;
#endif


#if defined(TEST_RMA_OP_GET)
const char *op_name = "get";
#elif defined(TEST_RMA_OP_PUT)
const char *op_name = "put";
#else
const char *op_name = "acc";
#endif

MPI_Datatype target_type = MPI_DATATYPE_NULL;
int target_size = 0;
MPI_Aint target_ext = 0;

static void usleep_by_count(unsigned long us)
{
    double start = MPI_Wtime() * 1000 * 1000;
    while (MPI_Wtime() * 1000 * 1000 - start < us);
    return;
}

#ifdef CHECK
static int check_error(int opsize)
{
    int err = 0;
    int d1, d2, d3;

    MPI_Get(checkbuf, BUFSIZE, MPI_DOUBLE, rank, 0, 1, target_type, win);
    MPI_Win_flush(rank, win);

    for (d1 = 0; d1 < SUB_DLEN; d1++) {
        for (d2 = 0; d2 < SUB_DLEN; d2++) {
            for (d3 = 0; d3 < opsize; d3++) {
                int idx = d1 * SUB_DLEN * SUB_DLEN + d2 * SUB_DLEN + d3;
#if defined(TEST_RMA_OP_GET)
                err = CTEST_double_diff(locbuf[idx], 100.0);
#elif defined(TEST_RMA_OP_PUT)
                err = CTEST_double_diff(checkbuf[idx], idx * 1.0);
#else
                err = CTEST_double_diff(checkbuf[idx], 100.0 + SKIP * idx * nprocs * 1.0);
#endif
                if (err)
                    break;
            }
        }
    }

    if (err) {
        for (d1 = 0; d1 < SUB_DLEN; d1++) {
            for (d2 = 0; d2 < SUB_DLEN; d2++) {
                for (d3 = 0; d3 < opsize; d3++) {
                    int idx = d1 * SUB_DLEN * SUB_DLEN + d2 * SUB_DLEN + d3;
                    printf("%.1f ", checkbuf[idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    return err;
}
#endif

static double run_test(int time, int nop, int opsize)
{
    int i, x;
    int dst;
    double t0, t_total = 0.0;

    MPI_Win_lock_all(0, win);
    for (x = 0; x < SKIP; x++) {
        for (dst = 0; dst < nprocs; dst++) {
#if defined(TEST_RMA_OP_GET)
            MPI_Get(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win);
#elif defined(TEST_RMA_OP_PUT)
            MPI_Put(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win);
#else
            MPI_Accumulate(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, MPI_SUM, win);
#endif
            MPI_Win_flush(dst, win);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef CHECK
    if (rank == 0) {
        int err = 0;
        err = check_error(opsize);
        if (err) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    t0 = MPI_Wtime();
    for (x = 0; x < ITER; x++) {
        for (dst = 0; dst < nprocs; dst++) {
            for (i = 0; i < nop; i++) {
#if defined(TEST_RMA_OP_GET)
                MPI_Get(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win);
#elif defined(TEST_RMA_OP_PUT)
                MPI_Put(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win);
#else
                MPI_Accumulate(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, MPI_SUM, win);
#endif
                MPI_Win_flush(dst, win);
            }
        }

        usleep_by_count(time);

        for (dst = 0; dst < nprocs; dst++) {
#if defined(TEST_RMA_OP_GET)
            MPI_Get(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win);
#elif defined(TEST_RMA_OP_PUT)
            MPI_Put(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win);
#else
            MPI_Accumulate(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, MPI_SUM, win);
#endif
            MPI_Win_flush(dst, win);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    t_total += MPI_Wtime() - t0;
    t_total = t_total / ITER * 1000 * 1000;     /* us */

    MPI_Win_unlock_all(win);

    return t_total;
}

static double on_t_total = 0.0;

static void run_with_async_config(const char *config, int time, int nop, int opsize)
{
    MPI_Info win_info = MPI_INFO_NULL;
    double t_total = 0.0, avg_t_total = 0.0;
    int i;

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "epoch_type", (char *) "lockall");
    MPI_Info_set(win_info, (char *) "async_config", config);
/*    MPI_Info_set(win_info, (char *) "alloc_shm", "true"); */

    // size in byte
    MPI_Win_allocate(sizeof(double) * WINSIZE, sizeof(double), win_info,
                     MPI_COMM_WORLD, &winbuf, &win);

    locbuf = malloc(sizeof(double) * BUFSIZE);
    for (i = 0; i < BUFSIZE; i++)
        locbuf[i] = 1.0 * i;

#ifdef CHECK
    checkbuf = malloc(sizeof(double) * BUFSIZE);
    memset(checkbuf, 0, sizeof(double) * BUFSIZE);
#endif

    /* reset window */
    MPI_Win_lock_all(0, win);
    for (i = 0; i < WINSIZE; i++)
        winbuf[i] = 100.0;
    MPI_Win_unlock_all(win);

    t_total = run_test(time, nop, opsize);

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
                "csp-%s-%s-nh%s: comp_size %d num_op %d opsize %d nprocs %d total_time %.2lf freq %.1f sp %.2f\n",
                op_name, config, nh, time, nop, opsize, nprocs, avg_t_total,
                (1 - time / avg_t_total) * 100 /*comm freq */ , async_speedup);
    }

    MPI_Info_free(&win_info);
    MPI_Win_free(&win);
    free(locbuf);
#ifdef CHECK
    free(checkbuf);
#endif
}

static void create_datatype(int opsize)
{
    int sizes[3], subsizes[3], starts[3];
    MPI_Aint lb = 0;

    sizes[0] = opsize + DLEN;
    sizes[1] = sizes[2] = DLEN;

    subsizes[0] = opsize;
    subsizes[1] = subsizes[2] = SUB_DLEN;
    starts[0] = starts[1] = starts[2] = 0;

    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &target_type);
    MPI_Type_commit(&target_type);

    MPI_Type_get_extent(target_type, &lb, &target_ext);
    MPI_Type_size(target_type, &target_size);

    if (target_ext != WINSIZE * sizeof(double) || target_size != BUFSIZE * sizeof(double)) {
        printf("wrong datatype size : extent %ld, size %d, winbuf %d, bufsize %d\n",
               target_ext, target_size, WINSIZE, WINSIZE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char *argv[])
{
    int time = SLEEP_TIME;
    int nop = NOP;
    int opsize = OPSIZE;

    MPI_Init(&argc, &argv);

    if (argc >= 4) {
        opsize_min = atoi(argv[1]);
        opsize_max = atoi(argv[2]);
        opsize_iter = atoi(argv[3]);
    }
    if (argc >= 5) {
        nop = atoi(argv[4]);
    }
    if (argc >= 6) {
        time = atoi(argv[5]);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (2 > nprocs) {
        if (rank == 0)
            fprintf(stderr, "Please run using at least 2 processes\n");
        goto exit;
    }

    for (opsize = opsize_min; opsize <= opsize_max; opsize *= opsize_iter) {
        /* increase len of X dimension */
        WINSIZE = ((opsize + DLEN) * DLEN * DLEN);      /* ensure always 3D noncontig */
        BUFSIZE = opsize * SUB_DLEN * SUB_DLEN;

        create_datatype(opsize);

        MPI_Barrier(MPI_COMM_WORLD);
        run_with_async_config("on", time, nop, opsize);

        MPI_Barrier(MPI_COMM_WORLD);
        run_with_async_config("off", time, nop, opsize);
    }

  exit:
    MPI_Finalize();

    return 0;
}
