/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"

/* Global environment setting */
CSP_env_param CSP_ENV;

/* Global process information object */
CSP_proc CSP_PROC;

/* User world communicator, including all users in the world */
MPI_Comm CSP_COMM_USER_WORLD = MPI_COMM_NULL;

#define CSP_SET_GLOBAL_COMM(gcomm, comm)   {        \
    gcomm = comm;                                   \
    comm = MPI_COMM_NULL;   /* avoid local free */  \
    }

static inline int check_valid_ghosts(void)
{
    int local_nprocs;
    int err_flag = 0;

    PMPI_Comm_size(CSP_PROC.local_comm, &local_nprocs);

    if (local_nprocs < 2) {
        CSP_ERR_PRINT("Can not create shared memory region, %d process in "
                      "MPI_COMM_TYPE_SHARED subcommunicator.\n", local_nprocs);
        err_flag++;
    }

    if (CSP_ENV.num_g < 1 || CSP_ENV.num_g >= local_nprocs) {
        CSP_ERR_PRINT("Wrong value of number of ghosts, %d. lt 1 or ge %d.\n",
                      CSP_ENV.num_g, local_nprocs);
        err_flag++;
    }

    return err_flag;
}

static inline int setup_common_info(void)
{
    int mpi_errno = MPI_SUCCESS;
    int local_rank;
    MPI_Comm node_comm = MPI_COMM_WORLD;
    int tmp_bcast_buf[2] = { 0, 0 };

    PMPI_Comm_rank(CSP_PROC.local_comm, &local_rank);
    mpi_errno = PMPI_Comm_split(MPI_COMM_WORLD, local_rank == 0, 1, &node_comm);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    if (local_rank == 0) {
        PMPI_Comm_rank(node_comm, &tmp_bcast_buf[0]);   /* node_id */
        PMPI_Comm_size(node_comm, &tmp_bcast_buf[1]);   /* num_nodes */
    }

    PMPI_Bcast(tmp_bcast_buf, 2, MPI_INT, 0, CSP_PROC.local_comm);
    CSP_PROC.node_id = tmp_bcast_buf[0];
    CSP_PROC.num_nodes = tmp_bcast_buf[1];

  fn_exit:
    if (node_comm != MPI_COMM_NULL)
        PMPI_Comm_free(&node_comm);
    return mpi_errno;

  fn_fail:
    /* Free global objects in MPI_Init_thread. */
    goto fn_exit;
}

/* Initialize environment setting */
static int CSP_initialize_env()
{
    char *val;
    int mpi_errno = MPI_SUCCESS;

    memset(&CSP_ENV, 0, sizeof(CSP_ENV));

    CSP_ENV.seg_size = CSP_DEFAULT_SEG_SIZE;
    val = getenv("CSP_SEG_SIZE");
    if (val && strlen(val)) {
        CSP_ENV.seg_size = atoi(val);
    }
    if (CSP_ENV.seg_size <= 0) {
        CSP_ERR_PRINT("Wrong CSP_SEG_SIZE %d\n", CSP_ENV.seg_size);
        return -1;
    }

    CSP_ENV.num_g = CSP_DEFAULT_NG;
    val = getenv("CSP_NG");
    if (val && strlen(val)) {
        CSP_ENV.num_g = atoi(val);
    }
    if (CSP_ENV.num_g <= 0) {
        CSP_ERR_PRINT("Wrong CSP_NG %d\n", CSP_ENV.num_g);
        return -1;
    }

    CSP_ENV.verbose = 0;
    val = getenv("CSP_VERBOSE");
    if (val && strlen(val)) {
        /* VERBOSE level */
        CSP_ENV.verbose = atoi(val);
        if (CSP_ENV.verbose < 0)
            CSP_ENV.verbose = 0;
    }

    CSP_ENV.lock_binding = CSP_LOCK_BINDING_RANK;
    val = getenv("CSP_LOCK_METHOD");
    if (val && strlen(val)) {
        if (!strncmp(val, "rank", strlen("rank"))) {
            CSP_ENV.lock_binding = CSP_LOCK_BINDING_RANK;
        }
        else if (!strncmp(val, "segment", strlen("segment"))) {
            CSP_ENV.lock_binding = CSP_LOCK_BINDING_SEGMENT;
        }
        else {
            CSP_ERR_PRINT("Unknown CSP_LOCK_METHOD %s\n", val);
            return -1;
        }
    }

    CSP_ENV.async_config = CSP_ASYNC_CONFIG_ON;
    val = getenv("CSP_ASYNC_CONFIG");
    if (val && strlen(val)) {
        if (!strncmp(val, "on", strlen("on"))) {
            CSP_ENV.async_config = CSP_ASYNC_CONFIG_ON;
        }
        else if (!strncmp(val, "off", strlen("off"))) {
            CSP_ENV.async_config = CSP_ASYNC_CONFIG_OFF;
        }
        else {
            CSP_ERR_PRINT("Unknown CSP_ASYNC_CONFIG %s\n", val);
            return -1;
        }
    }

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    CSP_ENV.load_opt = CSP_LOAD_OPT_RANDOM;

    val = getenv("CSP_RUMTIME_LOAD_OPT");
    if (val && strlen(val)) {
        if (!strncmp(val, "random", strlen("random"))) {
            CSP_ENV.load_opt = CSP_LOAD_OPT_RANDOM;
        }
        else if (!strncmp(val, "op", strlen("op"))) {
            CSP_ENV.load_opt = CSP_LOAD_OPT_COUNTING;
        }
        else if (!strncmp(val, "byte", strlen("byte"))) {
            CSP_ENV.load_opt = CSP_LOAD_BYTE_COUNTING;
        }
        else {
            CSP_ERR_PRINT("Unknown CSP_RUMTIME_LOAD_OPT %s\n", val);
            return -1;
        }
    }

    CSP_ENV.load_lock = CSP_LOAD_LOCK_NATURE;
    val = getenv("CSP_RUNTIME_LOAD_LOCK");
    if (val && strlen(val)) {
        if (!strncmp(val, "nature", strlen("nature"))) {
            CSP_ENV.load_lock = CSP_LOAD_LOCK_NATURE;
        }
        else if (!strncmp(val, "force", strlen("force"))) {
            CSP_ENV.load_lock = CSP_LOAD_LOCK_FORCE;
        }
        else {
            CSP_ERR_PRINT("Unknown CSP_RUNTIME_LOAD_LOCK %s\n", val);
            return -1;
        }
    }
#else
    CSP_ENV.load_opt = CSP_LOAD_OPT_STATIC;
    CSP_ENV.load_lock = CSP_LOAD_LOCK_NATURE;
#endif

    if (CSP_ENV.verbose && CSP_PROC.wrank == 0) {
        CSP_INFO_PRINT(1, "CASPER Configuration:  \n"
#ifdef CSP_ENABLE_EPOCH_STAT_CHECK
                       "    EPOCH_STAT_CHECK (enabled) \n"
#endif
#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
                       "    RUMTIME_LOAD_OPT (enabled) \n"
#endif
                       "    CSP_NG = %d \n"
                       "    CSP_LOCK_METHOD = %s \n"
                       "    CSP_ASYNC_CONFIG = %s\n",
                       CSP_ENV.num_g,
                       (CSP_ENV.lock_binding == CSP_LOCK_BINDING_RANK) ? "rank" : "segment",
                       (CSP_ENV.async_config == CSP_ASYNC_CONFIG_ON) ? "on" : "off");

        if (CSP_ENV.lock_binding == CSP_LOCK_BINDING_SEGMENT) {
            CSP_INFO_PRINT(1, "    CSP_SEG_SIZE = %d \n", CSP_ENV.seg_size);
        }

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
        CSP_INFO_PRINT(1, "Runtime Load Balancing Options:  \n"
                       "    CSP_RUMTIME_LOAD_OPT = %s \n"
                       "    CSP_RUNTIME_LOAD_LOCK = %s \n",
                       (CSP_ENV.load_opt == CSP_LOAD_OPT_RANDOM) ? "random" :
                       ((CSP_ENV.load_opt == CSP_LOAD_OPT_COUNTING) ? "op" : "byte"),
                       (CSP_ENV.load_lock == CSP_LOAD_LOCK_NATURE) ? "nature" : "force");
#endif
        CSP_INFO_PRINT(1, "\n");
        fflush(stdout);
    }
    return mpi_errno;
}

/* Initialize global communicator objects. */
static int CSP_initialize_proc(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Comm tmp_local_comm = MPI_COMM_NULL, tmp_ur_comm = MPI_COMM_NULL;
    MPI_Comm tmp_comm = MPI_COMM_NULL;
    int local_rank;

    /* Get a communicator only containing processes with shared memory */
    mpi_errno = PMPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                     MPI_INFO_NULL, &CSP_PROC.local_comm);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    PMPI_Comm_rank(CSP_PROC.local_comm, &local_rank);

    /* Check if user specifies valid number of ghosts */
    if (check_valid_ghosts()) {
        mpi_errno = MPI_ERR_OTHER;
        goto fn_fail;
    }

    /* Statically set the lowest ranks on every node as ghosts */
    CSP_PROC.proc_type = (local_rank < CSP_ENV.num_g) ? CSP_PROC_GHOST : CSP_PROC_USER;

    /* Reset user/ghost global object */
    CSP_reset_typed_proc();

    mpi_errno = PMPI_Comm_group(MPI_COMM_WORLD, &CSP_PROC.wgroup);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* Create a user comm_world including all the users,
     * user will access it instead of comm_world */
    mpi_errno = PMPI_Comm_split(MPI_COMM_WORLD, CSP_IS_USER, 1, &tmp_comm);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    if (CSP_IS_USER)
        CSP_SET_GLOBAL_COMM(CSP_COMM_USER_WORLD, tmp_comm);

    /* Create a user/ghost comm_local */
    mpi_errno = PMPI_Comm_split(CSP_PROC.local_comm, CSP_IS_USER, 1, &tmp_local_comm);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;
    if (CSP_IS_USER) {
        CSP_SET_GLOBAL_COMM(CSP_PROC.user.u_local_comm, tmp_local_comm);
    }
    else {
        CSP_SET_GLOBAL_COMM(CSP_PROC.ghost.g_local_comm, tmp_local_comm);
    }

    /* Create a user root communicator including the first user on every node */
    if (CSP_IS_USER) {
        int local_user_rank = -1;
        PMPI_Comm_rank(CSP_PROC.user.u_local_comm, &local_user_rank);
        mpi_errno = PMPI_Comm_split(CSP_COMM_USER_WORLD, local_user_rank == 0, 1, &tmp_ur_comm);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        if (local_user_rank == 0)
            CSP_SET_GLOBAL_COMM(CSP_PROC.user.ur_comm, tmp_ur_comm);
    }

    /* Set name for user comm_world.  */
    if (CSP_IS_USER) {
        mpi_errno = PMPI_Comm_set_name(CSP_COMM_USER_WORLD, "MPI_COMM_WORLD");
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    mpi_errno = setup_common_info();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    if (CSP_IS_USER) {
        mpi_errno = CSP_setup_proc();
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        CSP_print_proc();
    }
    else {
        mpi_errno = CSPG_setup_proc();
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        CSPG_print_proc();
    }

  fn_exit:
    /* Free unused communicators */
    if (tmp_comm != MPI_COMM_NULL)
        PMPI_Comm_free(&tmp_comm);
    if (tmp_ur_comm != MPI_COMM_NULL)
        PMPI_Comm_free(&tmp_ur_comm);
    return mpi_errno;

  fn_fail:
    /* Free global objects in MPI_Init_thread. */
    goto fn_exit;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
{
    int mpi_errno = MPI_SUCCESS;

    CSP_DBG_PRINT_FCNAME();

    if (required == 0 && provided == NULL) {
        /* default init */
        mpi_errno = PMPI_Init(argc, argv);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
    else {
        /* user init thread */
        mpi_errno = PMPI_Init_thread(argc, argv, required, provided);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    PMPI_Comm_rank(MPI_COMM_WORLD, &CSP_PROC.wrank);

    /* Initialize environment setting */
    mpi_errno = CSP_initialize_env();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* Initialize global process object */
    mpi_errno = CSP_initialize_proc();
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* User processes */
    if (CSP_IS_USER) {
        /* Other user-specific initialization */
        mpi_errno = CSP_init();
    }
    /* Ghost processes */
    else {
        /* Start ghost routine */
        CSPG_init();
        exit(0);
    }

  fn_exit:
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLING-- */

    if (CSP_IS_USER) {
        CSP_destroy_proc();
    }
    else {
        CSPG_destroy_proc();
    }

    PMPI_Abort(MPI_COMM_WORLD, 0);

    goto fn_exit;
    /* --END ERROR HANDLING-- */
}
