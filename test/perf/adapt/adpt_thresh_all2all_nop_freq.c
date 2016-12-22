/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
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
#define ITER_M 200
#define ITER_L 100
#define NOP 1

#define DLEN2 4
#define DLEN3 4
#define SUB_DLEN2 2
#define SUB_DLEN3 2
#define DLEN (36*1024)  /* window size is 36M*4*4*sizeof(double)=4Mbyts+512K
                         * extra 4K at dimension-1 to ensure noncontig target type.*/
#define SUB_DLEN 2      /* default submatrix size 2*2*2=8B */

/* 3D matrix on target window */
#define WINSIZE (DLEN*DLEN2*DLEN3)

MPI_Win win;
double *winbuf = NULL;
double *locbuf = NULL;

int verbose = 1;
int rank, nprocs;
int ITER = ITER_S;
/* parameters can be updated by input */
int comp_time = SLEEP_TIME;
int nop_min = NOP, nop_max = NOP, nop_iter = 2;
int sub_dlen = SUB_DLEN, bufsize = SUB_DLEN * SUB_DLEN3 * SUB_DLEN3;

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

#if defined(TEST_RMA_OP_GET)
#define ISSUE_RMA_OP(locbuf, origin_cnt, origin_type, target_rank, target_disp, target_cnt, target_type, win)   \
            MPI_Get(locbuf, origin_cnt, origin_type, target_rank, target_disp, target_cnt, target_type, win)
#elif defined(TEST_RMA_OP_PUT)
#define ISSUE_RMA_OP(locbuf, origin_cnt, origin_type, target_rank, target_disp, target_cnt, target_type, win)   \
            MPI_Put(locbuf, origin_cnt, origin_type, target_rank, target_disp, target_cnt, target_type, win)
#else
#define ISSUE_RMA_OP(locbuf, origin_cnt, origin_type, target_rank, target_disp, target_cnt, target_type, win)   \
            MPI_Accumulate(locbuf, origin_cnt, origin_type, target_rank, target_disp, target_cnt, target_type, MPI_SUM, win)
#endif

static double run_test(int nop)
{
    int i, x;
    int dst;
    double t0, t_total = 0.0;

    MPI_Win_lock_all(0, win);
    for (x = 0; x < SKIP; x++) {
        for (dst = 0; dst < nprocs; dst++) {
            ISSUE_RMA_OP(locbuf, bufsize, MPI_DOUBLE, dst, 0, 1, target_type, win);
#if !defined(TEST_RMA_FLUSH_ALL)
            MPI_Win_flush(dst, win);
#endif
        }
#if defined(TEST_RMA_FLUSH_ALL)
        MPI_Win_flush_all(win);
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);

    t0 = MPI_Wtime();
    for (x = 0; x < ITER; x++) {
        for (dst = 0; dst < nprocs; dst++) {
            ISSUE_RMA_OP(locbuf, bufsize, MPI_DOUBLE, dst, 0, 1, target_type, win);
#if !defined(TEST_RMA_FLUSH_ALL)
            MPI_Win_flush(dst, win);
#endif
        }
#if defined(TEST_RMA_FLUSH_ALL)
        MPI_Win_flush_all(win);
#endif

        usleep_by_count(comp_time);

        for (dst = 0; dst < nprocs; dst++) {
            for (i = 0; i < nop; i++) {
                ISSUE_RMA_OP(locbuf, bufsize, MPI_DOUBLE, dst, 0, 1, target_type, win);
#if !defined(TEST_RMA_FLUSH_ALL)
                MPI_Win_flush(dst, win);
#endif
            }
        }
#if defined(TEST_RMA_FLUSH_ALL)
        MPI_Win_flush_all(win);
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_total += MPI_Wtime() - t0;
    t_total = t_total / ITER * 1000 * 1000;     /* us */

    MPI_Win_unlock_all(win);

    return t_total;
}

static double run_with_async_config(const char *config, int nop)
{
    MPI_Info win_info = MPI_INFO_NULL;
    double t_total = 0.0, avg_t_total = 0.0;
    int i;

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "epoch_type", (char *) "fence|lockall");
    MPI_Info_set(win_info, (char *) "async_config", config);

    MPI_Win_allocate(sizeof(double) * WINSIZE, sizeof(double), win_info, MPI_COMM_WORLD, &winbuf,
                     &win);

    /* reset window */
    MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
    for (i = 0; i < WINSIZE; i++)
        winbuf[i] = 100.0;
    MPI_Win_fence(0, win);

    t_total = run_test(nop);

    MPI_Allreduce(&t_total, &avg_t_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avg_t_total = avg_t_total / nprocs;

    MPI_Info_free(&win_info);
    MPI_Win_free(&win);

    return avg_t_total;
}

static void create_datatype(void)
{
    int sizes[3], subsizes[3], starts[3];
    MPI_Aint lb = 0;

    sizes[0] = DLEN;
    sizes[1] = DLEN2;
    sizes[2] = DLEN3;
    subsizes[0] = sub_dlen;
    subsizes[1] = SUB_DLEN2;
    subsizes[2] = SUB_DLEN3;
    starts[0] = starts[1] = starts[2] = 0;

    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &target_type);
    MPI_Type_commit(&target_type);

    MPI_Type_get_extent(target_type, &lb, &target_ext);
    MPI_Type_size(target_type, &target_size);
    if (target_size != bufsize * sizeof(double)) {
        if (rank == 0)
            fprintf(stderr, "Internal error: target_size %d != bufsize %d\n", bufsize, bufsize);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

/* adjust interval when diff becomes small */
#define DIFF_REDUCE_ITER 30.0
/* report range */
#define DIFF_REPORT 10.0
/* stop after sufficient reports */
#define MAX_NREPORTS 5

static int nreports = 0;

static void inline report_output(int ng, int nop, double on_time, double diff)
{
    if (rank == 0) {
        fprintf(stdout,
                "csp-%s-ng%d: comp_time %d sub_dlen %d size %d nprocs %d nop %d total_time %.2lf diff %.2lf freq %.1f\n",
                op_name, ng, comp_time, sub_dlen, target_size, nprocs, nop, on_time,
                diff, (1 - comp_time / on_time) * 100 /*comm freq */);
        fflush(stdout);
    }
    nreports++;
}

int main(int argc, char *argv[])
{
    int nop = NOP, i;
    int ng = 1;
    char *ng_str = "";
    double diff_report = DIFF_REPORT, diff_reduce_iter = DIFF_REDUCE_ITER;

    MPI_Init(&argc, &argv);

    if (argc >= 4) {
        nop_min = atoi(argv[1]);
        nop_max = atoi(argv[2]);
        nop_iter = atoi(argv[3]);
    }
    if (argc >= 5) {
        comp_time = atoi(argv[4]);
    }
    if (argc >= 6) {
        sub_dlen = atoi(argv[5]);
    }
    if (argc >= 7) {
        ITER = atoi(argv[6]);
    }

    ng_str = getenv("CSP_NG");
    if (ng_str != "") {
        ng = atoi(ng_str);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (2 > nprocs) {
        if (rank == 0)
            fprintf(stderr, "Please run using at least 2 processes\n");
        goto exit;
    }

    if (comp_time < 1 || sub_dlen < 1 || sub_dlen > DLEN || ng < 1) {
        if (rank == 0)
            fprintf(stderr, "Invalid input arguments: comp_time = %d, sub_dlen = %d, ng = %d\n",
                    comp_time, sub_dlen, ng);
        goto exit;
    }

    diff_report = DIFF_REPORT / (ng * 0.8);     /* reduce report range for large ng */
    diff_reduce_iter = (DIFF_REDUCE_ITER / (ng * 0.8)); /* similar for fine-grained adjusting range */

    bufsize = sub_dlen * SUB_DLEN2 * SUB_DLEN3;
    create_datatype();

    /* initialize local buffer */
    locbuf = malloc(bufsize * sizeof(double));
    for (i = 0; i < bufsize; i++)
        locbuf[i] = i;

    nop = nop_min;
    while (nop <= nop_max) {
        double on_time = 0, off_time = 0, diff = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        on_time = run_with_async_config("on", nop);

        MPI_Barrier(MPI_COMM_WORLD);
        off_time = run_with_async_config("off", nop);

        /* off_time and on_time are already averaged. */
        diff = (off_time - on_time) / off_time * 100;

        /* report results for (-10%) < diff < (10%) */
        if (fabs(diff) < diff_report) {
            report_output(ng, nop, on_time, diff);
        }
        else if (verbose && rank == 0) {
            fprintf(stdout,
                    "verbose: nop %d iter %d on_time %.2lf off_time %.2lf diff %.2lf freq %.1f\n",
                    nop, ITER, on_time, off_time, diff,
                    (1 - comp_time / on_time) * 100 /*comm freq */);
            fflush(stdout);
        }

        /* stop test */
        if ((on_time > off_time && fabs(diff) >= diff_report)
            || nreports > MAX_NREPORTS) {
            break;
        }

        /* when diff becomes small, reduce interval */
        if (diff < diff_reduce_iter) {
            nop += nop_iter;
        }
        else {
            nop *= nop_iter;
        }

        /* reduce iteration when single run becomes expensive (time in us) */
        if ((on_time > 10000 || off_time > 10000) && ITER > ITER_M) {
            ITER = ITER_M;
        }
        if ((on_time > 25000 || off_time > 25000) && ITER > ITER_L) {
            ITER = ITER_L;
        }
    }

    if (nop == nop_max) {
        fprintf(stderr, "Cannot find the threshold. "
                "Please adjust input arguments: comp_time %d, sub_dlen %d, nop_min %d, nop_max %d\n",
                comp_time, sub_dlen, nop_min, nop_max);
        fflush(stderr);
    }

  exit:
    MPI_Type_free(&target_type);
    free(locbuf);
    MPI_Finalize();

    return 0;
}
