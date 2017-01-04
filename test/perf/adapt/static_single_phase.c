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
#include <mkl.h>

/* This benchmark demonstrates the improvement of asynchronous progress
 * or the load imbalance issue in single phase with 3D non-contigunous double data.
 * Every process performs all-to-all GET-flush-compute-ACC-flush-barrier,
 * with fixed computing time and increasing number of operations.
 * It is only used to compare the static version with increasing number of
 * ghost processes.
 * */

#define DLEN 4
#define SUB_DLEN 2

/* 3D matrix on target window */
#define WINSIZE (DLEN*DLEN*DLEN)
/* Local 3D submatrix */
#define BUFSIZE (SUB_DLEN*SUB_DLEN*SUB_DLEN)

double *winbuf = NULL;
double *locbuf = NULL;
int rank = 0, nprocs = 0;
int shm_rank = 0, shm_nprocs = 0;
MPI_Win win = MPI_WIN_NULL;

int PHASE_ITER = 10, COLL_ITER = 50, SKIP = 10;
int NOP = 1;
unsigned long SLEEP_TIME = 100; /* us */

MPI_Datatype target_type = MPI_DATATYPE_NULL;
int target_size = 0;
MPI_Aint target_ext = 0;

double *A, *B, *C;
int M = 1, K = 1, N = 1;        /* global size */
int m, n, k;                    /* local size */
double t_comp = 0.0;

static void target_computation_init(void)
{
    int i, j;

    m = M / nprocs;
    k = K / nprocs;
    n = N / nprocs;

    A = (double *) mkl_malloc(m * k * sizeof(double), 64);
    B = (double *) mkl_malloc(k * n * sizeof(double), 64);
    C = (double *) mkl_malloc(m * n * sizeof(double), 64);
}

static void target_computation(void)
{
    int i, j;
    double alpha = 1.0, beta = 0.0;
    double t0;

    t0 = MPI_Wtime();

    /* reset data */
    for (i = 0; i < (m * k); i++)
        A[i] = (double) (i + 1);

    for (i = 0; i < (k * n); i++)
        B[i] = (double) (-i - 1);

    for (i = 0; i < (m * n); i++)
        C[i] = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);

    t_comp += (MPI_Wtime() - t0);
}

static void target_computation_destroy(void)
{
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    A = NULL;
    B = NULL;
    C = NULL;
}

static int run_iteration(void)
{
    int i, px, x, nwin, nphase, ncoll, errs = 0;
    int dst;
    double t0, avg_total_time = 0.0, t_total = 0.0, avg_comp_time = 0.0;
    MPI_Info win_info = MPI_INFO_NULL;

    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, (char *) "epoch_type", (char *) "lockall|fence");

#if defined(DISABLE_SHM)
    /* disable shm rma for original case because too heavy per-op lock contention */
    MPI_Info_set(win_info, (char *) "alloc_shm", (char *) "false");
#endif

    MPI_Win_allocate(WINSIZE * sizeof(double), sizeof(double), win_info,
                     MPI_COMM_WORLD, &winbuf, &win);

    /* skip */
    for (x = 0; x < SKIP; x++) {
        /* reset window */
        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
        memset(winbuf, 0, sizeof(double) * WINSIZE);
        MPI_Win_fence(0, win);

        MPI_Win_lock_all(0, win);
        for (dst = 0; dst < nprocs; dst++) {
            MPI_Get(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win);
            MPI_Win_flush(dst, win);
        }
        MPI_Win_unlock_all(win);
    }

    /* start */
    t0 = MPI_Wtime();
    MPI_Win_lock_all(0, win);

    for (px = 0; px < PHASE_ITER; px++) {
        for (x = 0; x < COLL_ITER; x += 1) {
            for (dst = 0; dst < nprocs; dst++) {
                for (i = 0; i < NOP; i++) {
                    MPI_Get(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win);
                    MPI_Win_flush(dst, win);
                }
            }

            target_computation();

            for (dst = 0; dst < nprocs; dst++) {
                for (i = 0; i < NOP; i++) {
                    MPI_Accumulate(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, MPI_SUM,
                                   win);
                    MPI_Win_flush(dst, win);
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_unlock_all(win);

    t_total = (MPI_Wtime() - t0);       /* s */
    MPI_Reduce(&t_total, &avg_total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp, &avg_comp_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char header[256];

        memset(header, 0, sizeof(header));
        avg_total_time = avg_total_time / nprocs;
        avg_comp_time = avg_comp_time / nprocs;

#ifdef ENABLE_CSP
        const char *nh = getenv("CSP_NG");
        const char *gasync = getenv("CSP_ASYNC_CONFIG");

        /* static on/off */
        sprintf(header, "csp-%s-nh%s", gasync, nh);
#elif defined(DISABLE_SHM)
        sprintf(header, "orig-noshm");
#else
        sprintf(header, "orig");
#endif

        fprintf(stdout,
                "%s: nprocs %d M %d N %d K %d m %d n %d k %d iter %d %d comp %.2lf num_op %d total_time %.2lf\n",
                header, nprocs, M, N, K, m, n, k, PHASE_ITER, COLL_ITER,
                avg_comp_time, NOP, avg_total_time);
    }

    MPI_Win_free(&win);
    MPI_Info_free(&win_info);

    return errs;
}

static void create_datatype(void)
{
    int sizes[3], subsizes[3], starts[3];
    MPI_Aint lb = 0;

    sizes[0] = sizes[1] = sizes[2] = DLEN;
    subsizes[0] = subsizes[1] = subsizes[2] = SUB_DLEN;
    starts[0] = starts[1] = starts[2] = 1;

    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &target_type);
    MPI_Type_commit(&target_type);

    MPI_Type_get_extent(target_type, &lb, &target_ext);
    MPI_Type_size(target_type, &target_size);
}

int main(int argc, char *argv[])
{
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nprocs < 2) {
        fprintf(stderr, "Please run using at least two processes\n");
        goto exit;
    }

    if (argc >= 5) {
        NOP = atoi(argv[1]);
        M = atoi(argv[2]);
        N = atoi(argv[3]);
        K = atoi(argv[4]);
    }

    if (argc >= 6) {
        PHASE_ITER = atoi(argv[5]);
    }
    if (argc >= 7) {
        COLL_ITER = atoi(argv[6]);
    }
    SKIP = COLL_ITER / 10;

    /* initialize local buffer */
    locbuf = malloc(BUFSIZE * sizeof(double));
    for (i = 0; i < BUFSIZE; i++)
        locbuf[i] = i;

    create_datatype();
    target_computation_init();

    run_iteration();

  exit:
    target_computation_destroy();
    MPI_Type_free(&target_type);
    free(locbuf);
    MPI_Finalize();

    return 0;
}
