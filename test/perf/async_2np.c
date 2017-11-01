/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

/* This benchmark evaluates asynchronous progress in lockall epoch using 2 processes.
 * Rank 0 performs lockall-accumulate-flush-unlockall, and rank 1 performs
 * compute(busy wait)-test(poll MPI progress).*/

#define SLEEP_TIME 100  //us

//#define DEBUG
#define CHECK
#define ITER 10000
#define SUB_DLEN 2
#define DLEN 4
#define SIZE (DLEN*DLEN*DLEN)

#ifdef DEBUG
#define debug_printf(str,...) {fprintf(stdout, str, ## __VA_ARGS__);fflush(stdout);}
#else
#define debug_printf(str,...) {}
#endif

#define MPI_DATATYPE MPI_DOUBLE
#define DATATYPE double

MPI_Win win;
double *winbuf, locbuf[SIZE];
int locbuf_int[SIZE], resbuf_int[1], compbuf_int[1];    /* used for FOP or CAS. */
int rank, nprocs;
int NOP = 1;
MPI_Datatype target_type = MPI_DATATYPE;

#ifdef TEST_RMA_CONTIG
const char *dt_str = "contig";
#else
const char *dt_str = "subarray";
#endif

enum {
    OP_ACC,
    OP_PUT,
    OP_GET,
    OP_FOP,
    OP_CAS,
    OP_MAX
};
const char *OP_TYPE_NM[OP_MAX] = { "ACC", "PUT", "GET", "FOP", "CAS" };

int OP_TYPE = OP_ACC;

static void usleep_by_count(unsigned long us)
{
    double start = MPI_Wtime() * 1000 * 1000;
    while (MPI_Wtime() * 1000 * 1000 - start < us);
    return;
}

static void create_datatype(void)
{
#if defined(TEST_RMA_CONTIG)
    target_type = MPI_DATATYPE;
#else
    int sizes[3], subsizes[3], starts[3];
    MPI_Aint lb = 0;

    sizes[0] = DLEN;
    sizes[1] = DLEN;
    sizes[2] = DLEN;
    subsizes[0] = SUB_DLEN;
    subsizes[1] = SUB_DLEN;
    subsizes[2] = SUB_DLEN;
    starts[0] = starts[1] = starts[2] = 0;

    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DATATYPE, &target_type);
    MPI_Type_commit(&target_type);
#endif
}

static void release_datatype(void)
{
#if !defined(TEST_RMA_CONTIG)
    MPI_Type_free(&target_type);
#endif
}

static int run_test(int time)
{
    int i, x, errs = 0;
    int dst, src;
    double t0, t_total = 0.0;
    MPI_Request request;
    MPI_Status status;
    int buf[1];
    int flag = 0;
    int origin_count = 0, target_count = 0;

    if (rank == 0) {
        dst = 1;
        buf[0] = 99;
        MPI_Win_lock_all(0, win);
    }
    else {
        src = 0;
        buf[0] = 0;
        MPI_Irecv(buf, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &request);
    }

#if defined(TEST_RMA_CONTIG)
    origin_count = 1;
    target_count = 1;
#else
    origin_count = SUB_DLEN * SUB_DLEN * SUB_DLEN;
    target_count = 1;
#endif
    t0 = MPI_Wtime();
    for (x = 0; x < ITER; x++) {

        // rank 0 does RMA communication
        if (rank == 0) {
            switch (OP_TYPE) {
            case OP_PUT:
                for (i = 0; i < NOP; i++)
                    MPI_Put(locbuf, origin_count, MPI_DATATYPE, dst, 0, target_count, target_type,
                            win);
                MPI_Win_flush_all(win);
                break;
            case OP_GET:
                for (i = 0; i < NOP; i++)
                    MPI_Get(locbuf, origin_count, MPI_DATATYPE, dst, 0, target_count, target_type,
                            win);
                MPI_Win_flush_all(win);
                break;
            case OP_FOP:
                for (i = 0; i < NOP; i++)
                    MPI_Fetch_and_op(&locbuf_int[0], resbuf_int, MPI_INT, dst, 0, MPI_SUM, win);
                MPI_Win_flush_all(win);
                break;
            case OP_CAS:
                for (i = 0; i < NOP; i++)
                    MPI_Compare_and_swap(&locbuf_int[0], compbuf_int, resbuf_int, MPI_INT, dst, 0,
                                         win);
                MPI_Win_flush_all(win);
                break;
            case OP_ACC:
            default:
                for (i = 0; i < NOP; i++)
                    MPI_Accumulate(locbuf, origin_count, MPI_DATATYPE, dst, 0, target_count,
                                   target_type, MPI_SUM, win);
                MPI_Win_flush_all(win);
                break;
            }
        }
        // rank 1 does sleep and test
        else {
            usleep_by_count(time);
            MPI_Test(&request, &flag, &status);
        }
    }

    t_total += MPI_Wtime() - t0;
    t_total /= ITER;

    if (rank == 0) {
        MPI_Win_unlock_all(win);
        MPI_Send(buf, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }
    else {
        if (!flag)
            MPI_Wait(&request, &status);
        if (buf[0] != 99) {
            fprintf(stderr, "[%d]error: recv data %d != %d\n", rank, buf[0], 99);
            return errs;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
#ifdef ENABLE_CSP
        fprintf(stdout,
                "casper: %s, %s, comp_size %d num_op %d nprocs %d total_time %.2lf\n",
                OP_TYPE_NM[OP_TYPE], dt_str, time, NOP, nprocs, t_total * 1000 * 1000);
#else
        fprintf(stdout,
                "orig: %s, %s, comp_size %d num_op %d nprocs %d total_time %.2lf\n",
                OP_TYPE_NM[OP_TYPE], dt_str, time, NOP, nprocs, t_total * 1000 * 1000);
#endif
    }

    return errs;
}

int main(int argc, char *argv[])
{
    int i;
    int min_time = SLEEP_TIME, max_time = SLEEP_TIME, iter_time = 2, time;
    MPI_Info win_info = MPI_INFO_NULL;

    MPI_Init(&argc, &argv);
    debug_printf("[%d]init done\n", rank);

    if (argc >= 4) {
        min_time = atoi(argv[1]);
        max_time = atoi(argv[2]);
        iter_time = atoi(argv[3]);
        NOP = atoi(argv[4]);
    }
    if (argc >= 5) {
        NOP = atoi(argv[4]);
    }
    if (argc >= 6) {
        OP_TYPE = atoi(argv[5]);
    }

#if !defined(TEST_RMA_CONTIG)
    if (OP_TYPE == OP_FOP || OP_TYPE == OP_CAS) {
        if (rank == 0)
            fprintf(stderr, "Do not support FOP or CAS with derived datatype.\n");
        goto exit;
    }
#endif

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    debug_printf("[%d]comm_size done\n", rank);

    if (2 != nprocs) {
        if (rank == 0)
            fprintf(stderr, "Please run using 2 processes\n");
        goto exit;
    }

    for (i = 0; i < SIZE; i++) {
        locbuf[i] = (i + 1) * 0.5;
    }

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "epoch_type", (char *) "lockall");

    // size in byte
    MPI_Win_allocate(sizeof(DATATYPE) * SIZE, sizeof(DATATYPE), win_info, MPI_COMM_WORLD,
                     &winbuf, &win);
    create_datatype();

    /* reset window */
    MPI_Win_lock_all(0, win);
    winbuf[0] = 0.0;
    MPI_Win_unlock_all(win);

    debug_printf("[%d]win_allocate done\n", rank);

    for (time = min_time; time <= max_time; time *= iter_time) {
        run_test(time);
    }

    if (win_info != MPI_INFO_NULL)
        MPI_Info_free(&win_info);

    MPI_Win_free(&win);

  exit:
    release_datatype();
    MPI_Finalize();

    return 0;
}
