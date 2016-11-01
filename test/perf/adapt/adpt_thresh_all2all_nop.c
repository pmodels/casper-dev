/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>

/* This benchmark evaluates the threshold that benefiting from asynchronous
 * progress in passive all to all communication with increasing number of operations,
 * this benchmark can be also used for the test with increasing number of processes.
 * Every one performs lockall-RMA-compute-RMA-unlockall with 3D non-contigunous double target type.
 *
 * It should run with per-coll level adaptation, to ensure both on/off use AM for local
 * processes.*/

#define SLEEP_TIME 100  //us
#define SKIP 50
#define ITER_S 500
#define ITER_M 500
#define ITER_L 500
#define ITER_LL 500
#define NOP 1

#define DLEN 4
#define SUB_DLEN 2

/* 3D matrix on target window */
#define WINSIZE (DLEN*DLEN*DLEN)
/* Local 3D submatrix */
#define BUFSIZE (SUB_DLEN*SUB_DLEN*SUB_DLEN)

MPI_Win win;
double *winbuf = NULL;
double *locbuf = NULL;

/* #define CHECK */
#ifdef CHECK
#include <ctest.h>
double *checkbuf = NULL;
#endif

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
static int check_error(void)
{
    int err = 0;
    int d1, d2, d3;

    MPI_Get(checkbuf, BUFSIZE, MPI_DOUBLE, rank, 0, 1, target_type, win);
    MPI_Win_flush(rank, win);

    for (d1 = 0; d1 < SUB_DLEN; d1++) {
        for (d2 = 0; d2 < SUB_DLEN; d2++) {
            for (d3 = 0; d3 < SUB_DLEN; d3++) {
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
                for (d3 = 0; d3 < SUB_DLEN; d3++) {
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

static double run_test(int time, int nop)
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
        err = check_error();
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
    MPI_Win_allocate(sizeof(double) * WINSIZE, sizeof(double), win_info, MPI_COMM_WORLD, &winbuf,
                     &win);

    /* reset window */
    MPI_Win_lock_all(0, win);
    for (i = 0; i < WINSIZE; i++)
        winbuf[i] = 100.0;
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

static void create_datatype(void)
{
    int sizes[3], subsizes[3], starts[3];
    MPI_Aint lb = 0;

    sizes[0] = sizes[1] = sizes[2] = DLEN;
    subsizes[0] = subsizes[1] = subsizes[2] = SUB_DLEN;
    starts[0] = starts[1] = starts[2] = 0;

    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &target_type);
    MPI_Type_commit(&target_type);

    MPI_Type_get_extent(target_type, &lb, &target_ext);
    MPI_Type_size(target_type, &target_size);
}

int main(int argc, char *argv[])
{
    int time = SLEEP_TIME;
    int nop = NOP, i;

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

    create_datatype();
    /* initialize local buffer */
    locbuf = malloc(BUFSIZE * sizeof(double));
    for (i = 0; i < BUFSIZE; i++)
        locbuf[i] = i;

#ifdef CHECK
    checkbuf = malloc(BUFSIZE * sizeof(double));
    memset(checkbuf, 0, BUFSIZE * sizeof(double));
#endif

    for (nop = nop_min; nop <= nop_max; nop *= nop_iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        run_with_async_config("on", time, nop);

        MPI_Barrier(MPI_COMM_WORLD);
        run_with_async_config("off", time, nop);
    }

  exit:
    MPI_Type_free(&target_type);
    free(locbuf);
#ifdef CHECK
    free(checkbuf);
#endif
    MPI_Finalize();

    return 0;
}
