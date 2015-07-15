/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

/* -*- Mode: C; c-basic-offset:4 ; -*- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>

/*
 * This test checks different asynchronous configuration.
 */

#define NUM_OPS 5
#define CHECK
#define OUTPUT_FAIL_DETAIL

double *winbuf = NULL;
double *locbuf = NULL;
double *checkbuf = NULL;
int rank, nprocs;
MPI_Win win = MPI_WIN_NULL;
int ITER = 10;

static void change_data(int nop, int x)
{
    int dst, i;
    for (dst = 0; dst < nprocs; dst++) {
        for (i = 0; i < nop; i++) {
            locbuf[dst * nop + i] = 1.0 * (x + 1) * (i + 1) + nop * dst;
        }
    }
}

static void print_buffers(int nop)
{
    int i = 0;
#ifdef OUTPUT_FAIL_DETAIL
    fprintf(stderr, "[%d] locbuf:\n", rank);
    for (i = 0; i < nop * nprocs; i++) {
        fprintf(stderr, "%.1lf ", locbuf[i]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "[%d] winbuf:\n", rank);
    for (i = 0; i < nop * nprocs; i++) {
        fprintf(stderr, "%.1lf ", checkbuf[i]);
    }
    fprintf(stderr, "\n");
#endif
}

#ifdef USE_LOCAL_CHECK
/* check window data by each local process via local load */
static int check_local_data(int nop, int x)
{
    int errs = 0;
    int i;

    /* note that it is in an epoch */
    MPI_Barrier(MPI_COMM_WORLD);
    for (i = 0; i < nop; i++) {
        if (winbuf[i] != locbuf[rank * nop + i]) {
            fprintf(stderr, "[%d] winbuf[%d] %.1lf != %.1lf\n", rank, i,
                    winbuf[i], locbuf[rank * nop + i]);
            errs++;
        }
    }

#ifdef OUTPUT_FAIL_DETAIL
    if (errs > 0)
        print_buffers(nop);
#endif
    MPI_Barrier(MPI_COMM_WORLD);

    return errs;
}
#else

/* check window data by origin process via remote get */
static int check_data_all(int nop, int x)
{
    int errs = 0;
    int dst, i;

    /* note that it is in an epoch */

    memset(checkbuf, 0, NUM_OPS * nprocs * sizeof(double));

    for (dst = 0; dst < nprocs; dst++) {
        MPI_Get(&checkbuf[dst * nop], nop, MPI_DOUBLE, dst, 0, nop, MPI_DOUBLE, win);
    }
    MPI_Win_flush_all(win);

    for (dst = 0; dst < nprocs; dst++) {
        for (i = 0; i < nop; i++) {
            if (checkbuf[dst * nop + i] != locbuf[dst * nop + i]) {
                fprintf(stderr, "[%d] winbuf[%d] %.1lf != %.1lf\n", dst, i,
                        checkbuf[dst * nop + i], locbuf[dst * nop + i]);
                errs++;
            }
        }
    }

#ifdef OUTPUT_FAIL_DETAIL
    if (errs > 0)
        print_buffers(nop);
#endif

    return errs;
}
#endif

static void reset_window(int op_issued)
{
    int i;

    if (op_issued) {
        MPI_Win_flush_all(win);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (i = 0; i < NUM_OPS; i++) {
        winbuf[i] = 0.0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

static void issue_ops(int nop)
{
    int dst;
    int i;

    for (dst = 0; dst < nprocs; dst++) {
        MPI_Put(&locbuf[dst * nop], 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, win);
    }
    MPI_Win_flush_all(win);

    for (dst = 0; dst < nprocs; dst++) {
        for (i = 1; i < nop; i++) {
            MPI_Put(&locbuf[dst * nop + i], 1, MPI_DOUBLE, dst, i, 1, MPI_DOUBLE, win);
        }
    }
    MPI_Win_flush_all(win);
}

/* Test win_allocate(async_config=off, epoch_type=lockall) with lockall epoch. */
static int run_test1(int nop)
{
    int x, errs = 0;
    MPI_Info win_info = MPI_INFO_NULL;

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "async_config", (char *) "off");

    /* size in byte */
    MPI_Win_allocate(sizeof(double) * NUM_OPS, sizeof(double), win_info,
                     MPI_COMM_WORLD, &winbuf, &win);

    MPI_Win_lock_all(0, win);
    reset_window(0);

    for (x = 0; x < ITER; x++) {
        /* change date in every iteration */
        change_data(nop, x);

        if (rank == 0) {
            issue_ops(nop);
        }

        /* check in every iteration */
#ifdef USE_LOCAL_CHECK
        errs += check_local_data(nop, x);
#else
        if (rank == 0) {
            errs += check_data_all(nop, x);
        }
#endif
    }
    MPI_Win_unlock_all(win);

    if (rank == 0 && errs > 0) {
        fprintf(stderr, "Test win_allocate(async_config=off) found %d errors\n", errs);
    }

    MPI_Bcast(&errs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (win != MPI_WIN_NULL)
        MPI_Win_free(&win);
    if (win_info != MPI_INFO_NULL)
        MPI_Info_free(&win_info);
    return errs;
}

/* Test win_allocate(async_config=on, epoch_type=lockall) with lockall epoch. */
static int run_test2(int nop)
{
    int x, errs = 0;
    MPI_Info win_info = MPI_INFO_NULL;

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "async_config", (char *) "on");

    /* size in byte */
    MPI_Win_allocate(sizeof(double) * NUM_OPS, sizeof(double), win_info,
                     MPI_COMM_WORLD, &winbuf, &win);

    MPI_Win_lock_all(0, win);
    reset_window(0);

    for (x = 0; x < ITER; x++) {
        /* change date in every iteration */
        change_data(nop, x);

        if (rank == 0) {
            issue_ops(nop);
        }

        /* check in every iteration */
#ifdef USE_LOCAL_CHECK
        errs += check_local_data(nop, x);
#else
        if (rank == 0) {
            errs += check_data_all(nop, x);
        }
#endif
    }
    MPI_Win_unlock_all(win);

    if (rank == 0 && errs > 0) {
        fprintf(stderr, "Test win_allocate(async_config=on) found %d errors\n", errs);
    }

    MPI_Bcast(&errs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (win != MPI_WIN_NULL)
        MPI_Win_free(&win);
    if (win_info != MPI_INFO_NULL)
        MPI_Info_free(&win_info);
    return errs;
}

/* Test win_allocate(async_config=auto, epoch_type=lockall) with lockall epoch. */
static int run_test3(int nop)
{
    int x, errs = 0;
    MPI_Info win_info = MPI_INFO_NULL;

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "async_config", (char *) "auto");

    MPI_Win_allocate(sizeof(double) * NUM_OPS, sizeof(double), win_info,
                     MPI_COMM_WORLD, &winbuf, &win);
    MPI_Win_lock_all(0, win);
    reset_window(0);

    for (x = 0; x < ITER; x++) {
        /* change date in every iteration */
        change_data(nop, x);

        if (rank == 1) {
            issue_ops(nop);
        }

        /* check in every iteration */
#ifdef USE_LOCAL_CHECK
        errs += check_local_data(nop, x);
#else
        if (rank == 1) {
            errs += check_data_all(nop, x);
        }
#endif
    }

    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);

    MPI_Win_allocate(sizeof(double) * NUM_OPS, sizeof(double), win_info,
                     MPI_COMM_WORLD, &winbuf, &win);
    MPI_Win_lock_all(0, win);
    reset_window(0);

    for (x = 0; x < ITER; x++) {
        /* change date in every iteration */
        change_data(nop, x);

        if (rank == 0) {
            issue_ops(nop);
        }

        /* check in every iteration */
#ifdef USE_LOCAL_CHECK
        errs += check_local_data(nop, x);
#else
        if (rank == 0) {
            errs += check_data_all(nop, x);
        }
#endif
    }

    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);

    if (rank == 0 && errs > 0) {
        fprintf(stderr, "Test win_allocate(async_config=auto) found %d errors\n", errs);
    }

    MPI_Bcast(&errs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Info_free(&win_info);
    return errs;
}

int main(int argc, char *argv[])
{
    int size = NUM_OPS;
    int i, errs = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nprocs < 2) {
        fprintf(stderr, "Please run using at least 2 processes\n");
        goto exit;
    }

    locbuf = calloc(NUM_OPS * nprocs, sizeof(double));
    checkbuf = calloc(NUM_OPS * nprocs, sizeof(double));
    for (i = 0; i < NUM_OPS * nprocs; i++) {
        locbuf[i] = 1.0 * i;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    errs = run_test1(size);
    if (errs)
        goto exit;

    MPI_Barrier(MPI_COMM_WORLD);
    errs = run_test2(size);
    if (errs)
        goto exit;

    MPI_Barrier(MPI_COMM_WORLD);
    errs = run_test3(size);
    if (errs)
        goto exit;

  exit:

    if (rank == 0) {
        fprintf(stdout, "%d errors\n", errs);
    }
    if (locbuf)
        free(locbuf);
    if (checkbuf)
        free(checkbuf);

    MPI_Finalize();

    return 0;
}
