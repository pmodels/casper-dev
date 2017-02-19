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
 * progress in passive all to N-nodes communication with increasing number of operations.
 * Every one performs lockall-RMA-compute-RMA-unlockall with 3D non-contigunous double target type.
 * The N can be specified at input. */

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

#if defined(TEST_RMA_OP_FOP) || defined(TEST_RMA_OP_CAS)
#define MPI_DATATYPE MPI_LONG
#define DATATYPE long
#else
#define MPI_DATATYPE MPI_DOUBLE
#define DATATYPE double
#endif
MPI_Win win;
DATATYPE *winbuf = NULL;
DATATYPE *locbuf = NULL;
DATATYPE *resbuf = NULL;
DATATYPE compbuf[1];            /* only used in CAS. */

int verbose = 1;
int rank, nprocs;
int ITER = ITER_S;
int NNODES = 1;                 /* number of target nodes */
int ntargets = 1;               /* dynamically calculated by NNODES */

/* parameters can be updated by input */
int comp_time = SLEEP_TIME;
int nop_min = NOP, nop_max = NOP, nop_iter = 2;
int sub_dlen = SUB_DLEN, bufsize = SUB_DLEN * SUB_DLEN3 * SUB_DLEN3;

#if defined(TEST_RMA_OP_GET)
const char *op_name = "get";
#elif defined(TEST_RMA_OP_PUT)
const char *op_name = "put";
#elif defined(TEST_RMA_OP_GETACC)
const char *op_name = "gacc";
#elif defined(TEST_RMA_OP_FOP)
const char *op_name = "fop";
#elif defined(TEST_RMA_OP_CAS)
const char *op_name = "cas";
#else
const char *op_name = "acc";
#endif

MPI_Datatype target_type = MPI_DATATYPE_NULL;
int target_size = 0;
MPI_Aint target_ext = 0;

/* nprocs bits, set bit[x] to 1 if x is target */
int *target_bits = NULL;

static void usleep_by_count(unsigned long us)
{
    double start = MPI_Wtime() * 1000 * 1000;
    while (MPI_Wtime() * 1000 * 1000 - start < us);
    return;
}

#if defined(TEST_RMA_OP_GET)
#define ISSUE_RMA_OP(locbuf, origin_cnt, origin_type, compbuf, resbuf, target_rank, target_disp, target_cnt, target_type, win)   \
            MPI_Get(locbuf, origin_cnt, origin_type, target_rank, target_disp, target_cnt, target_type, win)
#elif defined(TEST_RMA_OP_PUT)
#define ISSUE_RMA_OP(locbuf, origin_cnt, origin_type, compbuf, resbuf, target_rank, target_disp, target_cnt, target_type, win)   \
            MPI_Put(locbuf, origin_cnt, origin_type, target_rank, target_disp, target_cnt, target_type, win)
#elif defined(TEST_RMA_OP_GETACC)
#define ISSUE_RMA_OP(locbuf, origin_cnt, origin_type, compbuf, resbuf, target_rank, target_disp, target_cnt, target_type, win)   \
            MPI_Get_accumulate(locbuf, origin_cnt, origin_type, resbuf, origin_cnt, origin_type, \
                               target_rank, target_disp, target_cnt, target_type, MPI_SUM, win)
#elif defined(TEST_RMA_OP_FOP)
#define ISSUE_RMA_OP(locbuf, origin_cnt, origin_type, compbuf, resbuf, target_rank, target_disp, target_cnt, target_type, win)   \
            MPI_Fetch_and_op(locbuf, resbuf, origin_type, target_rank, target_disp, MPI_SUM, win)
#elif defined(TEST_RMA_OP_CAS)
#define ISSUE_RMA_OP(locbuf, origin_cnt, origin_type, compbuf, resbuf, target_rank, target_disp, target_cnt, target_type, win)   \
            MPI_Compare_and_swap(locbuf, compbuf, resbuf, origin_type, target_rank, target_disp, win)
#else
#define ISSUE_RMA_OP(locbuf, origin_cnt, origin_type, compbuf, resbuf, target_rank, target_disp, target_cnt, target_type, win)   \
            MPI_Accumulate(locbuf, origin_cnt, origin_type, target_rank, target_disp, target_cnt, target_type, MPI_SUM, win)
#endif

static double run_test(int nop)
{
    int i, x;
    int dst;
    double t0, t_total = 0.0;
    int target_cnt;

#ifdef TEST_RMA_CONTIG
    target_cnt = bufsize;
#else
    target_cnt = 1;
#endif

    MPI_Win_lock_all(0, win);
    for (x = 0; x < SKIP; x++) {
        for (dst = 0; dst < nprocs; dst++) {
            ISSUE_RMA_OP(locbuf, bufsize, MPI_DATATYPE, compbuf, resbuf, dst, 0,
                         target_cnt, target_type, win);
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
            if (target_bits[dst] == 1) {
                ISSUE_RMA_OP(locbuf, bufsize, MPI_DATATYPE, compbuf, resbuf, dst, 0,
                             target_cnt, target_type, win);
#if !defined(TEST_RMA_FLUSH_ALL)
                MPI_Win_flush(dst, win);
#endif
            }
        }
#if defined(TEST_RMA_FLUSH_ALL)
        MPI_Win_flush_all(win);
#endif

        usleep_by_count(comp_time);

        for (dst = 0; dst < nprocs; dst++) {
            if (target_bits[dst] == 1) {
                for (i = 0; i < nop; i++) {
                    ISSUE_RMA_OP(locbuf, bufsize, MPI_DATATYPE, compbuf, resbuf, dst, 0,
                                 target_cnt, target_type, win);
#if !defined(TEST_RMA_FLUSH_ALL)
                    MPI_Win_flush(dst, win);
#endif
                }
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

    MPI_Win_allocate(sizeof(DATATYPE) * WINSIZE, sizeof(DATATYPE), win_info,
                     MPI_COMM_WORLD, &winbuf, &win);

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
#if defined(TEST_RMA_CONTIG) || defined(TEST_RMA_OP_FOP) || defined(TEST_RMA_OP_CAS)
    target_type = MPI_DATATYPE;
    target_size = sizeof(DATATYPE);
    target_ext = sizeof(DATATYPE);
#else
    int sizes[3], subsizes[3], starts[3];
    MPI_Aint lb = 0;

    sizes[0] = DLEN;
    sizes[1] = DLEN2;
    sizes[2] = DLEN3;
    subsizes[0] = sub_dlen;
    subsizes[1] = SUB_DLEN2;
    subsizes[2] = SUB_DLEN3;
    starts[0] = starts[1] = starts[2] = 0;

    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DATATYPE, &target_type);
    MPI_Type_commit(&target_type);

    MPI_Type_get_extent(target_type, &lb, &target_ext);
    MPI_Type_size(target_type, &target_size);
    if (target_size != bufsize * sizeof(DATATYPE)) {
        if (rank == 0)
            fprintf(stderr, "Internal error: target_size %d != bufsize %d\n", bufsize, bufsize);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif
}

/* adjust interval when diff becomes small */
#define DIFF_REDUCE_ITER 30.0
/* report range */
#define DIFF_REPORT 10.0
/* stop after sufficient reports */
#define MAX_NREPORTS 3

static int nreports = 0;

static void inline report_output(int ng, int nop, double on_time, double diff)
{
    if (rank == 0) {
        fprintf(stdout,
                "csp-%s-ng%d: comp_time %d count %d target_size %d nprocs %d ntnodes %d ntargets %d nop %d total_time %.2lf diff %.2lf freq %.1f\n",
                op_name, ng, comp_time, bufsize, target_size, nprocs, NNODES,
                ntargets, nop, on_time, diff, (1 - comp_time / on_time) * 100 /*comm freq */);
        fflush(stdout);
    }
    nreports++;
}

static void set_target_bits(void)
{
    int shm_rank, shm_nprocs, node_id = 0;
    int i;
    MPI_Comm shm_comm = MPI_COMM_NULL, node_comm = MPI_COMM_NULL;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_rank(shm_comm, &shm_rank);
    MPI_Comm_size(shm_comm, &shm_nprocs);

    /* first rank on every node gets node id */
    MPI_Comm_split(MPI_COMM_WORLD, shm_rank == 0, shm_rank, &node_comm);
    if (shm_rank == 0) {
        MPI_Comm_rank(node_comm, &node_id);
    }

    MPI_Bcast(&node_id, 1, MPI_INT, 0, shm_comm);

    if (node_id < NNODES) {
        target_bits[rank] = 1;
    }
    else {
        target_bits[rank] = 0;
    }

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, target_bits, 1, MPI_INT, MPI_COMM_WORLD);

    /* count number of targets in world */
    ntargets = 0;
    for (i = 0; i < nprocs; i++) {
        if (target_bits[i] == 1)
            ntargets++;
    }
    MPI_Comm_free(&node_comm);
    MPI_Comm_free(&shm_comm);
}

int main(int argc, char *argv[])
{
    int nop = NOP, i;
    int ng = 1;
    char *ng_str = "";
    double diff_report = DIFF_REPORT, diff_reduce_iter = DIFF_REDUCE_ITER;
    int backwarded = 0;

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
    if (argc >= 8) {
        NNODES = atoi(argv[7]);
    }

    ng_str = getenv("CSP_NG");
    if (ng_str != NULL && strlen(ng_str) > 0) {
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
            fprintf(stderr,
                    "Invalid input arguments: comp_time = %d, sub_dlen = %d, ng = %d\n",
                    comp_time, sub_dlen, ng);
        goto exit;
    }

    target_bits = malloc(sizeof(int) * nprocs);
    memset(target_bits, 0, sizeof(int) * nprocs);
    set_target_bits();

    diff_report = DIFF_REPORT / (ng * 0.8);     /* reduce report range for large ng */
    diff_reduce_iter = (DIFF_REDUCE_ITER / (ng * 0.8)); /* similar for fine-grained adjusting range */

#if defined(TEST_RMA_OP_FOP) || defined(TEST_RMA_OP_CAS)
    bufsize = 1;
#else
    bufsize = sub_dlen * SUB_DLEN2 * SUB_DLEN3;
#endif
    create_datatype();

    /* initialize local buffers */
    locbuf = malloc(bufsize * sizeof(DATATYPE));
    resbuf = malloc(bufsize * sizeof(DATATYPE));
    for (i = 0; i < bufsize; i++) {
        locbuf[i] = i;
        resbuf[i] = i;
    }

    nop = nop_min;
    while (nop <= nop_max && nop >= nop_min) {
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
                    "verbose: nop %d bufsize %d iter %d on_time %.2lf off_time %.2lf diff %.2lf freq %.1f\n",
                    nop, bufsize, ITER, on_time, off_time, diff,
                    (1 - comp_time / on_time) * 100 /*comm freq */);
            fflush(stdout);
        }

        /* stop test */
        if ((on_time > off_time && fabs(diff) >= diff_report && nreports > 0)
            || nreports > MAX_NREPORTS) {
            break;
        }

        /* forwarding */
        if (backwarded == 0) {
            /* when diff becomes small or negative, reduce interval */
            if (diff < diff_reduce_iter) {
                /* when diff directly increases to negative with 0 report,
                 * start backward with smaller interval. */
                if (diff < 0 && nreports == 0) {
                    nop -= 1;
                    backwarded = 1;
                }
                else {
                    nop += nop_iter;
                }
            }
            else if (backwarded == 0) {
                nop *= nop_iter;
            }
        }
        /* slightly increase diff in backward within range */
        else if (diff < diff_reduce_iter) {
            nop -= 1;
        }
        /* diff increases out of range in backward */
        else {
            break;
        }

        /* reduce iteration when single run becomes expensive (time in us) */
        if ((on_time > 10000 || off_time > 10000) && ITER > ITER_M) {
            ITER = ITER_M;
        }
        if ((on_time > 25000 || off_time > 25000) && ITER > ITER_L) {
            ITER = ITER_L;
        }
    }

    if ((nop == nop_max || nreports == 0) && rank == 0) {
        fprintf(stderr,
                "Cannot find the threshold. "
                "Please adjust input arguments: nprocs %d comp_time %d, sub_dlen %d, nop_min %d, nop_max %d\n",
                nprocs, comp_time, sub_dlen, nop_min, nop_max);
        fflush(stderr);
    }

  exit:
    free(target_bits);
#if !defined(TEST_RMA_CONTIG) && !defined(TEST_RMA_OP_FOP) && !defined(TEST_RMA_OP_CAS)
    MPI_Type_free(&target_type);
#endif
    free(locbuf);
    free(resbuf);
    MPI_Finalize();

    return 0;
}
