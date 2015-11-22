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

/* This benchmark measures adaptation for the execution contains multiple
 * communication-intensive phases and computation-intensive phases with 3D
 * non-contigunous double data.
 * -During the execution of each window, multiple phases exist;
 * -During each phase, multiple collective calls (win_set_info) exist;
 * -Between two collective calls, every process performs all-to-all
 *  RMA-flush-compute-RMA-flush-barrier, with fixed computing time and increasing
 *  number of operations. The computation intensive phase runs with larger computing
 *  time and a small number of operations; the communication intensive phase runs
 *  with small computing time and increasing number of operations.
 *
 * It generates following modes :
 * - no adaptation (static on of off)
 * - user guided adaptation, set hint in every win_allocate
 * - user guided adaptation, set hint in every win_set_info
 * - self-profiling based adaptation
 * - self-profiling based adaptation with ghost-synchronization.
 * */

#define ITER 400
#define WIN_ITER 200
#define PHASE_ITER 100  /* change computation/communication proportion every 100 iteration */
#define COLL_ITER 50    /* insert collective call every 50 iteration */
#define SKIP 10

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

#if defined(TEST_RMA_OP_GET)
const char *op_name = "get";
#elif defined(TEST_RMA_OP_PUT)
const char *op_name = "put";
#else
const char *op_name = "acc";
#endif

int NOP_L_MAX = 16, NOP_L_MIN = 16, NOP_L_ITER = 2, NOP_S = 1, NOP_L = 1;
unsigned long SLEEP_TIME_S = 1, SLEEP_TIME_L = 100;     /* us */

MPI_Datatype target_type = MPI_DATATYPE_NULL;
int target_size = 0;
MPI_Aint target_ext = 0;

static int target_computation(unsigned long us)
{
    double start = MPI_Wtime() * 1000 * 1000;

    if (us == 0)
        return 0;

    while (MPI_Wtime() * 1000 * 1000 - start < us);
    return 0;
}

#if defined(TEST_RMA_OP_GET)
#define MPI_RMA_OP(x, dst, i) { \
    MPI_Get(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win); \
    }
#elif defined(TEST_RMA_OP_PUT)
#define MPI_RMA_OP(x, dst, i) { \
    MPI_Put(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win); \
    }
#else
#define MPI_RMA_OP(x, dst, i) { \
    MPI_Accumulate(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, MPI_SUM, win); \
    }
#endif

static int run_iteration()
{
    int i, wx, cx, px, x, nwin, nphase, ncoll, errs = 0;
    int dst;
    int nop = NOP_S;
    unsigned long sleep_time = SLEEP_TIME_S;
    double t0, avg_total_time = 0.0, t_total = 0.0;
    MPI_Info win_info = MPI_INFO_NULL;
    MPI_Info async_info = MPI_INFO_NULL;
    double st0 = 0.0, t_comm_phase = 0.0, t_comp_phase = 0.0;
    double avg_t_comp_phase = 0.0, avg_t_comm_phase = 0.0;
    int comp_pcnt = 0, comm_pcnt = 0;

    nwin = ITER / WIN_ITER;
    nphase = WIN_ITER / PHASE_ITER;
    ncoll = PHASE_ITER / COLL_ITER;

    MPI_Info_create(&win_info);
    MPI_Info_create(&async_info);

    MPI_Info_set(win_info, (char *) "epoch_type", (char *) "lockall|fence");
    MPI_Info_set(async_info, (char *) "symmetric", (char *) "true");

#if !defined(ENABLE_CSP) && !defined(ENABLE_CSP_ADPT_U)
    /* disable shm rma for original case because too heavy per-op lock contention */
    MPI_Info_set(win_info, (char *) "alloc_shm", (char *) "false");
#endif

    /* skip */
    MPI_Win_allocate(WINSIZE * sizeof(double), sizeof(double), win_info,
                     MPI_COMM_WORLD, &winbuf, &win);
    for (x = 0; x < SKIP; x++) {
        /* reset window */
        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
        memset(winbuf, 0, sizeof(double) * WINSIZE);
        MPI_Win_fence(0, win);

        MPI_Win_lock_all(0, win);
        for (dst = 0; dst < nprocs; dst++) {
            MPI_RMA_OP(x, dst, 0);
            MPI_Win_flush(dst, win);
        }
        MPI_Win_unlock_all(win);
    }
    MPI_Win_free(&win);

    /* start */
    t0 = MPI_Wtime();
    for (wx = 0; wx < nwin; wx++) {
#if defined(ENABLE_CSP_ADPT_U)
        /* first phase is comp heavy */
        MPI_Info_set(win_info, (char *) "async_config", (char *) "on");
#endif
        MPI_Win_allocate(WINSIZE * sizeof(double), sizeof(double), win_info,
                         MPI_COMM_WORLD, &winbuf, &win);

        /* reset window */
        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
        memset(winbuf, 0, sizeof(double) * WINSIZE);
        MPI_Win_fence(0, win);
        MPI_Win_lock_all(0, win);

        for (px = 0; px < nphase; px += 1) {
            st0 = MPI_Wtime();

            if (px % 2 == 0) {  /* heavy comp */
                sleep_time = SLEEP_TIME_L;
                nop = NOP_S;
                comp_pcnt++;
#if defined(ENABLE_CSP_ADPT_U)
                MPI_Info_set(async_info, (char *) "async_config", (char *) "on");
#endif
            }
            else {      /* heavy comm */
                sleep_time = SLEEP_TIME_S;
                nop = NOP_L;
                comm_pcnt++;
#if defined(ENABLE_CSP_ADPT_U)
                MPI_Info_set(async_info, (char *) "async_config", (char *) "off");
#endif
            }

            for (cx = 0; cx < ncoll; cx += 1) {
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Win_set_info(win, async_info);

                for (x = 0; x < COLL_ITER; x += 1) {
                    for (dst = 0; dst < nprocs; dst++) {
                        for (i = 0; i < nop; i++) {
                            MPI_RMA_OP(x, dst, i);
                            MPI_Win_flush(dst, win);
                        }
                    }

                    target_computation(sleep_time);

                    for (dst = 0; dst < nprocs; dst++) {
                        MPI_RMA_OP(x, dst, i);
                        MPI_Win_flush(dst, win);
                    }
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);

            if (px % 2 == 0) {  /* heavy comp */
                t_comp_phase += (MPI_Wtime() - st0);
            }
            else {      /* heavy comm */
                t_comm_phase += (MPI_Wtime() - st0);
            }
        }

        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);
    }

    t_total = (MPI_Wtime() - t0) * 1000;        /*ms */
    t_comp_phase = t_comp_phase * 1000;
    t_comm_phase = t_comm_phase * 1000;

    MPI_Reduce(&t_total, &avg_total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp_phase, &avg_t_comp_phase, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_phase, &avg_t_comm_phase, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char header[256];

        memset(header, 0, sizeof(header));
        avg_total_time = avg_total_time / nprocs;       /* us */
        avg_t_comp_phase = avg_t_comp_phase / nprocs;   /* us */
        avg_t_comm_phase = avg_t_comm_phase / nprocs;   /* us */

#ifdef ENABLE_CSP
        const char *nh = getenv("CSP_NG");
        const char *gasync = getenv("CSP_ASYNC_CONFIG");
        if (!strncmp(gasync, "auto", strlen("auto"))) {
            const char *p_lev = getenv("CSP_ASYNC_SCHED_LEVEL");
            const char *p_int = getenv("CSP_RUNTIME_ASYNC_SCHED_MIN_INT");
            const char *p_freq_l = getenv("CSP_RUNTIME_ASYNC_SCHED_THR_L");
            const char *p_freq_h = getenv("CSP_RUNTIME_ASYNC_SCHED_THR_H");
            const char *g_int = getenv("CSP_RUNTIME_ASYNC_TIMED_GSYNC_INT");

            if (!strncmp(p_lev, "anytime", strlen("anytime"))) {
                sprintf(header, "csp-%s-%s-I%sL%sH%sG%s-nh%s", op_name, p_lev, p_int, p_freq_l,
                        p_freq_h, g_int, nh);
            }
            else {
                sprintf(header, "csp-%s-%s-I%sL%sH%s-nh%s", op_name, p_lev, p_int, p_freq_l,
                        p_freq_h, nh);
            }
        }
        else {
            /* static on/off */
            sprintf(header, "csp-%s-%s-nh%s", op_name, gasync, nh);
        }
#elif defined(ENABLE_CSP_ADPT_U)
        const char *nh = getenv("CSP_NG");
        const char *gasync = getenv("CSP_ASYNC_CONFIG");
        const char *p_lev = getenv("CSP_ASYNC_SCHED_LEVEL");
        /* per-window or per-collective user guide */
        sprintf(header, "csp-%s-u-%s-nh%s", op_name, p_lev, nh);
#else
        sprintf(header, "orig-%s", op_name);
#endif

        fprintf(stdout,
                "%s: nprocs %d sleep_s %lu sleep_l %lu num_op_s %d num_op_l %d "
                "total_time %.2lf comp-p %.2lf comm-p %.2lf nwin %d nphase %d ncoll %d\n",
                header, nprocs, SLEEP_TIME_S, SLEEP_TIME_L, NOP_S, NOP_L, avg_total_time,
                avg_t_comp_phase, avg_t_comm_phase, nwin, nphase, ncoll);
    }

    MPI_Info_free(&win_info);
    MPI_Info_free(&async_info);

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
    MPI_Comm shm_comm = MPI_COMM_NULL;
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_rank(shm_comm, &shm_rank);
    MPI_Comm_size(shm_comm, &shm_nprocs);

    if (nprocs < 2) {
        fprintf(stderr, "Please run using at least two processes\n");
        goto exit;
    }

    if (argc >= 4) {
        NOP_S = atoi(argv[1]);
        NOP_L_MIN = atoi(argv[2]);
        NOP_L_MAX = atoi(argv[3]);
        NOP_L_ITER = atoi(argv[4]);
    }
    if (argc >= 6) {
        SLEEP_TIME_S = atoi(argv[5]);
        SLEEP_TIME_L = atoi(argv[6]);
    }

    /* initialize local buffer */
    locbuf = malloc(BUFSIZE * sizeof(double));
    for (i = 0; i < BUFSIZE; i++)
        locbuf[i] = i;

    create_datatype();

    for (NOP_L = NOP_L_MIN; NOP_L <= NOP_L_MAX; NOP_L *= NOP_L_ITER) {
        run_iteration();
    }

  exit:
    if (shm_comm)
        MPI_Comm_free(&shm_comm);
    MPI_Type_free(&target_type);
    free(locbuf);
    MPI_Finalize();

    return 0;
}
