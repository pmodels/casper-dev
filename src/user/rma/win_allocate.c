/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include "csp.h"
#include "info.h"

#include <ctype.h>

/* String of CSP_target_epoch_stat enum (for debug). */
const char *CSP_target_epoch_stat_name[4] = {
    "NO_EPOCH",
    "LOCK",
    "PSCW"
};

/* String of CSP_win_epoch_stat enum (for debug). */
const char *CSP_win_epoch_stat_name[4] = {
    "NO_EPOCH",
    "FENCE",
    "LOCK_ALL",
    "PER_TARGET"
};

static unsigned long win_id = 0;

static int read_win_info(MPI_Info info, CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;

    ug_win->info_args.no_local_load_store = 0;
    ug_win->info_args.epoch_type = CSP_EPOCH_LOCK_ALL | CSP_EPOCH_LOCK |
        CSP_EPOCH_PSCW | CSP_EPOCH_FENCE;
    ug_win->info_args.async_config = CSP_ENV.async_config;      /* default */

    /* default window name */
    sprintf(ug_win->info_args.win_name, "win-%d-%lu", CSP_MY_RANK_IN_WORLD, win_id);
    win_id++;

    if (info != MPI_INFO_NULL) {
        int info_flag = 0;
        char info_value[MPI_MAX_INFO_VAL + 1];

        /* Check if user wants to turn off async */
        memset(info_value, 0, sizeof(info_value));
        mpi_errno = PMPI_Info_get(info, "async_config", MPI_MAX_INFO_VAL, info_value, &info_flag);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        if (info_flag == 1) {
            if (!strncmp(info_value, "off", strlen("off"))) {
                ug_win->info_args.async_config = CSP_ASYNC_CONFIG_OFF;
            }
            else if (!strncmp(info_value, "on", strlen("on"))) {
                ug_win->info_args.async_config = CSP_ASYNC_CONFIG_ON;
            }
#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
            else if (!strncmp(info_value, "auto", strlen("auto"))) {
                ug_win->info_args.async_config = CSP_ASYNC_CONFIG_AUTO;
            }
#endif
        }

        /* Check if we are allowed to ignore force-lock for local target,
         * require force-lock by default. */
        memset(info_value, 0, sizeof(info_value));
        mpi_errno = PMPI_Info_get(info, "no_local_load_store", MPI_MAX_INFO_VAL,
                                  info_value, &info_flag);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        if (info_flag == 1) {
            if (!strncmp(info_value, "true", strlen("true")))
                ug_win->info_args.no_local_load_store = 1;
        }

        /* Check if user specifies epoch types */
        memset(info_value, 0, sizeof(info_value));
        mpi_errno = PMPI_Info_get(info, "epoch_type", MPI_MAX_INFO_VAL, info_value, &info_flag);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        if (info_flag == 1) {
            int user_epoch_type = 0;
            char *type = NULL;

            type = strtok(info_value, "|");
            while (type != NULL) {
                if (!strncmp(type, "lockall", strlen("lockall"))) {
                    user_epoch_type |= CSP_EPOCH_LOCK_ALL;
                }
                else if (!strncmp(type, "lock", strlen("lock"))) {
                    user_epoch_type |= CSP_EPOCH_LOCK;
                }
                else if (!strncmp(type, "pscw", strlen("pscw"))) {
                    user_epoch_type |= CSP_EPOCH_PSCW;
                }
                else if (!strncmp(type, "fence", strlen("fence"))) {
                    user_epoch_type |= CSP_EPOCH_FENCE;
                }
                type = strtok(NULL, "|");
            }

            if (user_epoch_type != 0)
                ug_win->info_args.epoch_type = user_epoch_type;
        }

        /* Check if user sets window name.
         * It is not passed to MPI, only for casper debug use). For the name
         * user wants to pass to MPI, should call MPI_Win_set_name instead. */
        memset(info_value, 0, sizeof(info_value));
        mpi_errno = PMPI_Info_get(info, "win_name", MPI_MAX_INFO_VAL, info_value, &info_flag);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        if (info_flag == 1) {
            memset(ug_win->info_args.win_name, 0, sizeof(ug_win->info_args.win_name));
            strncpy(ug_win->info_args.win_name, info_value, MPI_MAX_OBJECT_NAME);
        }
    }

    CSP_DBG_PRINT("no_local_load_store %d, epoch_type=%s|%s|%s|%s\n",
                  ug_win->info_args.no_local_load_store,
                  ((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK_ALL) ? "lockall" : ""),
                  ((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) ? "lock" : ""),
                  ((ug_win->info_args.epoch_type & CSP_EPOCH_PSCW) ? "pscw" : ""),
                  ((ug_win->info_args.epoch_type & CSP_EPOCH_FENCE) ? "fence" : ""));

    if (CSP_ENV.verbose) {
        int user_rank = -1;
        PMPI_Comm_rank(ug_win->user_comm, &user_rank);
        if (user_rank == 0) {
            CSP_INFO_PRINT(2, "CASPER Window: %s \n"
                           "    no_local_load_store = %s\n"
                           "    epoch_type = %s%s%s%s\n"
                           "    async_config = %s\n\n",
                           ug_win->info_args.win_name,
                           (ug_win->info_args.no_local_load_store ? "TRUE" : " FALSE"),
                           ((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK_ALL) ? "lockall" : ""),
                           ((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) ? "|lock" : ""),
                           ((ug_win->info_args.epoch_type & CSP_EPOCH_PSCW) ? "|pscw" : ""),
                           ((ug_win->info_args.epoch_type & CSP_EPOCH_FENCE) ? "|fence" : ""),
                           CSP_get_async_config_name(ug_win->info_args.async_config));
        }
    }

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

static int gather_ranks(CSP_win * win, int *num_ghosts, int *gp_ranks_in_world,
                        int *unique_gp_ranks_in_world)
{
    int mpi_errno = MPI_SUCCESS;
    int user_nprocs;
    int *gp_bitmap = NULL;
    int user_world_rank, tmp_num_ghosts;
    int i, j, gp_rank;
    int world_nprocs;

    PMPI_Comm_size(MPI_COMM_WORLD, &world_nprocs);
    PMPI_Comm_size(win->user_comm, &user_nprocs);

    gp_bitmap = CSP_calloc(world_nprocs, sizeof(int));
    if (gp_bitmap == NULL)
        goto fn_fail;

    /* Get ghost ranks of each USER process.
     *
     * The ghosts of user_world rank x are stored as x*num_g: (x+1)*num_g-1,
     * it is used to catch ghosts for a target rank in epoch.
     * Unique ghost ranks are only used for creating communicators.*/
    tmp_num_ghosts = 0;
    for (i = 0; i < user_nprocs; i++) {
        user_world_rank = win->targets[i].user_world_rank;

        for (j = 0; j < CSP_ENV.num_g; j++) {
            gp_rank = CSP_ALL_G_RANKS_IN_WORLD[user_world_rank * CSP_ENV.num_g + j];
            gp_ranks_in_world[i * CSP_ENV.num_g + j] = gp_rank;

            /* Unique ghost ranks */
            if (!gp_bitmap[gp_rank]) {
                unique_gp_ranks_in_world[tmp_num_ghosts++] = gp_rank;
                gp_bitmap[gp_rank] = 1;

                CSP_assert(tmp_num_ghosts <= CSP_NUM_NODES * CSP_ENV.num_g);
            }
        }
    }
    *num_ghosts = tmp_num_ghosts;

  fn_exit:
    if (gp_bitmap)
        free(gp_bitmap);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

static int send_win_general_parameters(MPI_Info info, CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int is_u_world = 0;
    CSP_info_keyval_t *info_keyvals = NULL;
    int npairs = 0;
    int func_params[4];

    is_u_world = (ug_win->user_comm == CSP_COMM_USER_WORLD) ? 1 : 0;

    mpi_errno = CSP_info_deserialize(info, &info_keyvals, &npairs);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* Send first part of parameters */
    func_params[0] = ug_win->max_local_user_nprocs;
    func_params[1] = ug_win->info_args.epoch_type;
    func_params[2] = is_u_world;
    func_params[3] = npairs;

    mpi_errno = CSP_func_set_param((char *) func_params, sizeof(func_params), ug_win->ur_g_comm);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;
    CSP_DBG_PRINT(" Send parameters: max_local_user_nprocs %d, epoch_type %d, "
                  "is_u_world=%d, info npairs=%d\n", ug_win->max_local_user_nprocs,
                  ug_win->info_args.epoch_type, is_u_world, npairs);

    /* Send user info */
    if (npairs > 0 && info_keyvals) {
        mpi_errno = CSP_func_set_param((char *) info_keyvals,
                                       sizeof(CSP_info_keyval_t) * npairs, ug_win->ur_g_comm);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
        CSP_DBG_PRINT(" Send parameters: info\n");
    }

  fn_exit:
    if (info_keyvals)
        free(info_keyvals);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

static int create_ug_comm(int num_ghosts, int *gp_ranks_in_world, CSP_win * win)
{
    int mpi_errno = MPI_SUCCESS;
    int user_nprocs, user_rank, world_nprocs;
    int *ug_ranks_in_world = NULL;
    int i, num_ug_ranks;

    PMPI_Comm_size(win->user_comm, &user_nprocs);
    PMPI_Comm_rank(win->user_comm, &user_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_nprocs);

    /* maximum amount equals to world size */
    ug_ranks_in_world = CSP_calloc(world_nprocs, sizeof(int));
    if (ug_ranks_in_world == NULL)
        goto fn_fail;

    /* -Create ug communicator including all USER processes and Ghost processes. */
    num_ug_ranks = num_ghosts;
    memcpy(ug_ranks_in_world, gp_ranks_in_world, num_ghosts * sizeof(int));
    for (i = 0; i < user_nprocs; i++) {
        ug_ranks_in_world[num_ug_ranks++] = win->targets[i].world_rank;
    }
    if (num_ug_ranks > world_nprocs) {
        fprintf(stderr, "num_ug_ranks %d > world_nprocs %d, num_ghosts=%d, user_nprocs=%d\n",
                num_ug_ranks, world_nprocs, num_ghosts, user_nprocs);
    }
    CSP_assert(num_ug_ranks <= world_nprocs);

    PMPI_Group_incl(CSP_GROUP_WORLD, num_ug_ranks, ug_ranks_in_world, &win->ug_group);
    mpi_errno = PMPI_Comm_create_group(MPI_COMM_WORLD, win->ug_group, 0, &win->ug_comm);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

  fn_exit:
    if (ug_ranks_in_world)
        free(ug_ranks_in_world);

    return mpi_errno;

  fn_fail:
    if (win->ug_comm != MPI_COMM_NULL) {
        PMPI_Comm_free(&win->ug_comm);
        win->ug_comm = MPI_COMM_NULL;
    }
    if (win->ug_group != MPI_GROUP_NULL) {
        PMPI_Group_free(&win->ug_group);
        win->ug_group = MPI_GROUP_NULL;
    }

    goto fn_exit;
}

static int create_communicators(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int *func_params = NULL;
    int *user_ranks_in_world = NULL;
    int *gp_ranks_in_world = NULL;
    int num_ghosts = 0, max_num_ghosts;
    int user_nprocs, user_local_rank;
    int *gp_ranks_in_ug = NULL;
    int *user_ranks_in_ug = NULL;
    int i;

    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);
    max_num_ghosts = CSP_ENV.num_g * CSP_NUM_NODES;

    PMPI_Comm_rank(ug_win->local_user_comm, &user_local_rank);
    ug_win->num_g_ranks_in_ug = CSP_ENV.num_g * ug_win->num_nodes;

    /* Optimization for user world communicator */
    if (ug_win->user_comm == CSP_COMM_USER_WORLD) {

        /* Create communicators
         *  local_ug_comm: including local USER and Ghost processes
         *  ug_comm: including all USER and Ghost processes
         */
        ug_win->local_ug_comm = CSP_COMM_LOCAL;
        ug_win->ug_comm = MPI_COMM_WORLD;
        PMPI_Comm_group(ug_win->local_ug_comm, &ug_win->local_ug_group);
        PMPI_Comm_group(ug_win->ug_comm, &ug_win->ug_group);

        /* -Get rank of all ghost and user processes in ug communicator */
        memcpy(ug_win->g_ranks_in_ug, CSP_ALL_UNIQUE_G_RANKS_IN_WORLD,
               ug_win->num_g_ranks_in_ug * sizeof(int));
        for (i = 0; i < user_nprocs; i++) {
            memcpy(ug_win->targets[i].g_ranks_in_ug,
                   &CSP_ALL_G_RANKS_IN_WORLD[i * CSP_ENV.num_g], sizeof(int) * CSP_ENV.num_g);
            ug_win->targets[i].ug_rank = CSP_USER_RANKS_IN_WORLD[i];
        }
    }
    else {
        /* ghost ranks for every user process, used for ghost fetching in epoch */
        gp_ranks_in_world = CSP_calloc(CSP_ENV.num_g * user_nprocs, sizeof(int));
        gp_ranks_in_ug = CSP_calloc(CSP_ENV.num_g * user_nprocs, sizeof(int));
        user_ranks_in_world = CSP_calloc(user_nprocs, sizeof(int));
        user_ranks_in_ug = CSP_calloc(user_nprocs, sizeof(int));

        /* Gather user rank information */
        mpi_errno = gather_ranks(ug_win, &num_ghosts, gp_ranks_in_world, ug_win->g_ranks_in_ug);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        if (user_local_rank == 0) {
            /* Set parameters to local Ghosts
             *  [0]: num_ghosts
             *  [1:N]: user ranks in comm_world
             *  [N+1:]: ghost ranks in comm_world
             */
            int *user_ranks_in_world_ptr = NULL;
            int func_param_size = user_nprocs + max_num_ghosts + 1;
            func_params = CSP_calloc(func_param_size, sizeof(int));

            func_params[0] = num_ghosts;

            /* user ranks in comm_world */
            user_ranks_in_world_ptr = &func_params[1];
            for (i = 0; i < user_nprocs; i++)
                user_ranks_in_world_ptr[i] = ug_win->targets[i].world_rank;

            /* ghost ranks in comm_world */
            memcpy(&func_params[user_nprocs + 1], ug_win->g_ranks_in_ug, num_ghosts * sizeof(int));
            mpi_errno = CSP_func_set_param((char *) func_params, sizeof(int) * func_param_size,
                                           ug_win->ur_g_comm);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }

        /* Create communicators
         *  ug_comm: including all USER and Ghost processes
         *  local_ug_comm: including local USER and Ghost processes
         */
        mpi_errno = create_ug_comm(num_ghosts, ug_win->g_ranks_in_ug, ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

#ifdef CSP_DEBUG
        {
            int ug_rank, ug_nprocs;
            PMPI_Comm_rank(ug_win->ug_comm, &ug_rank);
            PMPI_Comm_size(ug_win->ug_comm, &ug_nprocs);
            CSP_DBG_PRINT("created ug_comm, my rank %d/%d\n", ug_rank, ug_nprocs);
        }
#endif

        mpi_errno = PMPI_Comm_split_type(ug_win->ug_comm, MPI_COMM_TYPE_SHARED, 0,
                                         MPI_INFO_NULL, &ug_win->local_ug_comm);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

#ifdef CSP_DEBUG
        {
            int ug_rank, ug_nprocs;
            PMPI_Comm_rank(ug_win->local_ug_comm, &ug_rank);
            PMPI_Comm_size(ug_win->local_ug_comm, &ug_nprocs);
            CSP_DBG_PRINT("created local_ug_comm, my rank %d/%d\n", ug_rank, ug_nprocs);
        }
#endif

        /* Get all ghost rank in ug communicator */
        mpi_errno = PMPI_Group_translate_ranks(CSP_GROUP_WORLD, user_nprocs * CSP_ENV.num_g,
                                               gp_ranks_in_world, ug_win->ug_group, gp_ranks_in_ug);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        /* Get all user rank in ug communicator */
        for (i = 0; i < user_nprocs; i++)
            user_ranks_in_world[i] = ug_win->targets[i].world_rank;

        mpi_errno = PMPI_Group_translate_ranks(CSP_GROUP_WORLD, user_nprocs,
                                               user_ranks_in_world, ug_win->ug_group,
                                               user_ranks_in_ug);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        PMPI_Comm_group(ug_win->local_ug_comm, &ug_win->local_ug_group);

        for (i = 0; i < user_nprocs; i++) {
            memcpy(ug_win->targets[i].g_ranks_in_ug, &gp_ranks_in_ug[i * CSP_ENV.num_g],
                   sizeof(int) * CSP_ENV.num_g);
            ug_win->targets[i].ug_rank = user_ranks_in_ug[i];
        }
    }

#ifdef CSP_DEBUG
    {
        CSP_DBG_PRINT("%d unique g_ranks:\n", ug_win->num_g_ranks_in_ug);
        for (i = 0; i < ug_win->num_g_ranks_in_ug; i++) {
            CSP_DBG_PRINT("\t[%d] %d\n", i, ug_win->g_ranks_in_ug[i]);
        }
        CSP_DBG_PRINT("ug_ranks:\n");
        for (i = 0; i < user_nprocs; i++) {
            CSP_DBG_PRINT("\t[%d] %d\n", i, ug_win->targets[i].ug_rank);
        }
    }
#endif

  fn_exit:
    if (func_params)
        free(func_params);
    if (gp_ranks_in_world)
        free(gp_ranks_in_world);
    if (gp_ranks_in_ug)
        free(gp_ranks_in_ug);
    if (user_ranks_in_world)
        free(user_ranks_in_world);
    if (user_ranks_in_ug)
        free(user_ranks_in_ug);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

static int gather_base_offsets(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint tmp_u_offsets;
    int i, j;
    int user_local_rank, user_local_nprocs, user_rank, user_nprocs;
    MPI_Aint *base_g_offsets;
    MPI_Aint root_g_size = 0;

    PMPI_Comm_rank(ug_win->local_user_comm, &user_local_rank);
    PMPI_Comm_size(ug_win->local_user_comm, &user_local_nprocs);
    PMPI_Comm_rank(ug_win->user_comm, &user_rank);
    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);

    base_g_offsets = CSP_calloc(user_nprocs * CSP_ENV.num_g, sizeof(MPI_Aint));

#ifdef CSP_ENABLE_GRANT_LOCK_HIDDEN_BYTE
    /* All the ghosts use the byte located on ghost 0. */
    ug_win->grant_lock_g_offset = 0;
#endif

    /* Calculate the window size of ghost 0, because it contains extra space
     * for sync. */
    root_g_size = CSP_GP_SHARED_SG_SIZE;
#ifdef CSP_ENABLE_GRANT_LOCK_HIDDEN_BYTE
    root_g_size = CSP_max(root_g_size, sizeof(CSP_GRANT_LOCK_DATATYPE));
#endif

    /* Calculate my offset on the local shared buffer.
     * Note that all the ghosts start the window from baseptr of ghost 0,
     * hence all the local ghosts use the same offset of user buffers.
     * My offset is the total window size of all ghosts and all users in front of
     * me on the node (loop world ranks to get its window size without rank translate).*/
    tmp_u_offsets = root_g_size + CSP_GP_SHARED_SG_SIZE * (CSP_ENV.num_g - 1);

    i = 0;
    while (i < user_rank) {
        if (ug_win->targets[i].node_id == ug_win->node_id) {
            tmp_u_offsets += ug_win->targets[i].size;   /* size in bytes */
        }
        i++;
    }

    for (j = 0; j < CSP_ENV.num_g; j++) {
        base_g_offsets[user_rank * CSP_ENV.num_g + j] = tmp_u_offsets;
    }

    CSP_DBG_PRINT("[%d] local base_g_offset 0x%lx\n", user_rank, tmp_u_offsets);

    /* -Receive the address of all the shared user buffers on Ghost processes. */
    mpi_errno = PMPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, base_g_offsets,
                               CSP_ENV.num_g, MPI_AINT, ug_win->user_comm);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    for (i = 0; i < user_nprocs; i++) {
        for (j = 0; j < CSP_ENV.num_g; j++) {
            ug_win->targets[i].base_g_offsets[j] = base_g_offsets[i * CSP_ENV.num_g + j];
            CSP_DBG_PRINT("\t.base_g_offsets[%d] = 0x%lx\n",
                          j, ug_win->targets[i].base_g_offsets[j]);
        }
    }

  fn_exit:
    if (base_g_offsets)
        free(base_g_offsets);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

static int create_lock_windows(MPI_Aint size, int disp_unit, MPI_Info info, CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int i, j;
    int user_rank, user_nprocs;

    PMPI_Comm_size(ug_win->user_comm, &user_nprocs);
    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

    /* Need multiple windows for single lock synchronization */
    if (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) {
        ug_win->num_ug_wins = ug_win->max_local_user_nprocs;
    }
    /* Need a single window for lock_all only synchronization */
    else if (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK_ALL) {
        ug_win->num_ug_wins = 1;
    }

    ug_win->ug_wins = CSP_calloc(ug_win->num_ug_wins, sizeof(MPI_Win));
    for (i = 0; i < ug_win->num_ug_wins; i++) {
        mpi_errno = PMPI_Win_create(ug_win->base, size, disp_unit, info,
                                    ug_win->ug_comm, &ug_win->ug_wins[i]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    for (i = 0; i < user_nprocs; i++) {
        int win_off;
        CSP_DBG_PRINT("[%d] targets[%d].\n", user_rank, i);

        /* unique windows of each process, used in lock/flush */
        win_off = ug_win->targets[i].local_user_rank % ug_win->num_ug_wins;
        ug_win->targets[i].ug_win = ug_win->ug_wins[win_off];
        CSP_DBG_PRINT("\t\t .ug_win=0x%x (win_off %d)\n", ug_win->targets[i].ug_win, win_off);

        /* windows of each segment, used in OPs */
        for (j = 0; j < ug_win->targets[i].num_segs; j++) {
            win_off = ug_win->targets[i].local_user_rank % ug_win->num_ug_wins;
            ug_win->targets[i].segs[j].ug_win = ug_win->ug_wins[win_off];

            CSP_DBG_PRINT("\t\t .seg[%d].ug_win=0x%x (win_off %d)\n",
                          j, ug_win->targets[i].segs[j].ug_win, win_off);
        }
    }

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info,
                     MPI_Comm user_comm, void *baseptr, MPI_Win * win)
{
    int mpi_errno = MPI_SUCCESS;
    int ug_rank, ug_nprocs, user_nprocs, user_rank, user_world_rank, world_rank,
        user_local_rank, user_local_nprocs, ug_local_rank, ug_local_nprocs;
    CSP_win *ug_win = NULL;
    int i;
    void **base_pp = (void **) baseptr;
    MPI_Aint *tmp_gather_buf = NULL;
    int tmp_gather_cnt = 7;

    int tmp_bcast_buf[2];
    MPI_Info shared_info = MPI_INFO_NULL;

    CSP_DBG_PRINT_FCNAME();
    CSP_rm_count_start(CSP_RM_COMM_FREQ);

    ug_win = CSP_calloc(1, sizeof(CSP_win));

    /* If user specifies comm_world directly, use user comm_world instead;
     * else this communicator directly, because it should be created from user comm_world */
    if (user_comm == MPI_COMM_WORLD) {
        user_comm = CSP_COMM_USER_WORLD;
        ug_win->local_user_comm = CSP_COMM_USER_LOCAL;
        ug_win->user_root_comm = CSP_COMM_UR_WORLD;

        ug_win->node_id = CSP_MY_NODE_ID;
        ug_win->num_nodes = CSP_NUM_NODES;
        ug_win->user_comm = user_comm;
    }
    else {
        ug_win->user_comm = user_comm;
        mpi_errno = PMPI_Comm_split_type(user_comm, MPI_COMM_TYPE_SHARED, 0,
                                         MPI_INFO_NULL, &ug_win->local_user_comm);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        /* Create a user root communicator in order to figure out node_id and
         * num_nodes of this user communicator */
        PMPI_Comm_rank(ug_win->local_user_comm, &user_local_rank);
        mpi_errno = PMPI_Comm_split(ug_win->user_comm,
                                    user_local_rank == 0, 1, &ug_win->user_root_comm);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        if (user_local_rank == 0) {
            int node_id, num_nodes;
            PMPI_Comm_size(ug_win->user_root_comm, &num_nodes);
            PMPI_Comm_rank(ug_win->user_root_comm, &node_id);

            tmp_bcast_buf[0] = node_id;
            tmp_bcast_buf[1] = num_nodes;
        }

        PMPI_Bcast(tmp_bcast_buf, 2, MPI_INT, 0, ug_win->local_user_comm);
        ug_win->node_id = tmp_bcast_buf[0];
        ug_win->num_nodes = tmp_bcast_buf[1];
    }

    /* Read window configuration */
    mpi_errno = read_win_info(info, ug_win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    CSP_ra_update_async_stat(ug_win->info_args.async_config);
#endif

    /* If user turns off asynchronous redirection, simply return normal window; */
    if (ug_win->info_args.async_config == CSP_ASYNC_CONFIG_OFF)
        goto fn_noasync;

    PMPI_Comm_group(user_comm, &ug_win->user_group);
    PMPI_Comm_size(user_comm, &user_nprocs);
    PMPI_Comm_rank(user_comm, &user_rank);
    PMPI_Comm_size(ug_win->local_user_comm, &user_local_nprocs);
    PMPI_Comm_rank(ug_win->local_user_comm, &user_local_rank);
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_rank(CSP_COMM_USER_WORLD, &user_world_rank);

    ug_win->g_ranks_in_ug = CSP_calloc(CSP_ENV.num_g * ug_win->num_nodes, sizeof(MPI_Aint));
    ug_win->targets = CSP_calloc(user_nprocs, sizeof(CSP_win_target));
    for (i = 0; i < user_nprocs; i++) {
        ug_win->targets[i].base_g_offsets = CSP_calloc(CSP_ENV.num_g, sizeof(MPI_Aint));
        ug_win->targets[i].g_ranks_in_ug = CSP_calloc(CSP_ENV.num_g, sizeof(MPI_Aint));
    }

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    CSP_target_async_stat my_async_stat = CSP_TARGET_ASYNC_ON;
    int all_targets_async_off = 1;

    /* If runtime scheduling is enabled for this window, we exchange the
     * asynchronous configure with every target, since its value might be different. */
    if (ug_win->info_args.async_config == CSP_ASYNC_CONFIG_AUTO) {
        my_async_stat = CSP_ra_sched_async_stat();
    }
    tmp_gather_cnt++;
#endif

    /* Gather users' disp_unit, size, ranks and node_id */
    tmp_gather_buf = calloc(user_nprocs * tmp_gather_cnt, sizeof(MPI_Aint));
    tmp_gather_buf[tmp_gather_cnt * user_rank] = (MPI_Aint) disp_unit;
    tmp_gather_buf[tmp_gather_cnt * user_rank + 1] = size;      /* MPI_Aint, size in bytes */
    tmp_gather_buf[tmp_gather_cnt * user_rank + 2] = (MPI_Aint) user_local_rank;
    tmp_gather_buf[tmp_gather_cnt * user_rank + 3] = (MPI_Aint) world_rank;
    tmp_gather_buf[tmp_gather_cnt * user_rank + 4] = (MPI_Aint) user_world_rank;
    tmp_gather_buf[tmp_gather_cnt * user_rank + 5] = (MPI_Aint) ug_win->node_id;
    tmp_gather_buf[tmp_gather_cnt * user_rank + 6] = (MPI_Aint) user_local_nprocs;
#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    tmp_gather_buf[tmp_gather_cnt * user_rank + 7] = (MPI_Aint) my_async_stat;
#endif

    mpi_errno = PMPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                               tmp_gather_buf, tmp_gather_cnt, MPI_AINT, user_comm);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;
    for (i = 0; i < user_nprocs; i++) {
        ug_win->targets[i].disp_unit = (int) tmp_gather_buf[tmp_gather_cnt * i];
        ug_win->targets[i].size = tmp_gather_buf[tmp_gather_cnt * i + 1];
        ug_win->targets[i].local_user_rank = (int) tmp_gather_buf[tmp_gather_cnt * i + 2];
        ug_win->targets[i].world_rank = (int) tmp_gather_buf[tmp_gather_cnt * i + 3];
        ug_win->targets[i].user_world_rank = (int) tmp_gather_buf[tmp_gather_cnt * i + 4];
        ug_win->targets[i].node_id = (int) tmp_gather_buf[tmp_gather_cnt * i + 5];
        ug_win->targets[i].local_user_nprocs = (int) tmp_gather_buf[tmp_gather_cnt * i + 6];
#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
        ug_win->targets[i].async_stat = (CSP_async_config) tmp_gather_buf[tmp_gather_cnt * i + 7];
        all_targets_async_off &= (ug_win->targets[i].async_stat == CSP_TARGET_ASYNC_OFF);
#endif

        /* Calculate the maximum number of processes per node */
        ug_win->max_local_user_nprocs = CSP_max(ug_win->max_local_user_nprocs,
                                                ug_win->targets[i].local_user_nprocs);
    }

#ifdef CSP_DEBUG
    CSP_DBG_PRINT("my user local rank %d/%d, max_local_user_nprocs=%d, num_nodes=%d\n",
                  user_local_rank, user_local_nprocs, ug_win->max_local_user_nprocs,
                  ug_win->num_nodes);
    for (i = 0; i < user_nprocs; i++) {
        CSP_DBG_PRINT("\t targets[%d].disp_unit=%d, size=%ld, local_user_rank=%d, "
                      "world_rank=%d, user_world_rank=%d, node_id=%d, local_user_nprocs=%d\n",
                      i, ug_win->targets[i].disp_unit, ug_win->targets[i].size,
                      ug_win->targets[i].local_user_rank, ug_win->targets[i].world_rank,
                      ug_win->targets[i].user_world_rank, ug_win->targets[i].node_id,
                      ug_win->targets[i].local_user_nprocs);
    }
#endif

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    if (ug_win->info_args.async_config == CSP_ASYNC_CONFIG_AUTO) {
        /* Only the root user prints a summary of the async status. */
        if (user_rank == 0 && CSP_ENV.verbose >= 3) {
            int async_on_cnt = 0;
            int async_off_cnt = 0;
            for (i = 0; i < user_nprocs; i++) {
                if (ug_win->targets[i].async_stat == CSP_TARGET_ASYNC_ON) {
                    async_on_cnt++;
                }
                else {
                    async_off_cnt++;
                }
            }
            CSP_INFO_PRINT(3, "    Per-target async_config summary: on %d; off %d\n",
                           async_on_cnt, async_off_cnt);
        }

        /* The root user prints out the frequency number on every user process. */
        if (CSP_ENV.verbose >= 4) {
            /* Gather frequency numbers to root user */
            double *tmp_async_gather_buf = NULL;
            tmp_async_gather_buf = CSP_calloc(3 * user_nprocs, sizeof(double));

            tmp_async_gather_buf[3 * user_rank] = CSP_RM[CSP_RM_COMM_FREQ].last_freq * 1.0;
            tmp_async_gather_buf[3 * user_rank + 1] = CSP_RM[CSP_RM_COMM_FREQ].last_time;
            tmp_async_gather_buf[3 * user_rank + 2] = CSP_RM[CSP_RM_COMM_FREQ].last_interval;
            mpi_errno = PMPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, tmp_async_gather_buf,
                                       3, MPI_DOUBLE, user_comm);
            if (mpi_errno != MPI_SUCCESS) {
                free(tmp_async_gather_buf);
                goto fn_fail;
            }

            if (user_rank == 0) {
                CSP_INFO_PRINT(4, "    Per-target async_config:\n");
                for (i = 0; i < user_nprocs; i++)
                    CSP_INFO_PRINT(4, "    [%d] async stat: freq=%.0f(%.4f/%.4f)\n",
                                   i, tmp_async_gather_buf[3 * i],
                                   tmp_async_gather_buf[3 * i + 1],
                                   tmp_async_gather_buf[3 * i + 2]);
            }
            free(tmp_async_gather_buf);
        }
    }
#endif

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    if (all_targets_async_off)
        goto fn_noasync;
#endif

    /* Notify Ghosts start and create user root + ghosts communicator for
     * internal information exchange between users and ghosts. */
    if (user_local_rank == 0) {
        mpi_errno = CSP_func_start(CSP_FUNC_WIN_ALLOCATE, user_nprocs, user_local_nprocs);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        mpi_errno = CSP_func_new_ur_g_comm(&ug_win->ur_g_comm);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    /* Send parameter to ghosts */
    if (user_local_rank == 0) {
        mpi_errno = send_win_general_parameters(info, ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    /* Create communicators
     *  ug_comm: including all USER and Ghost processes
     *  local_ug_comm: including local USER and Ghost processes
     */
    mpi_errno = create_communicators(ug_win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    PMPI_Comm_rank(ug_win->local_ug_comm, &ug_local_rank);
    PMPI_Comm_size(ug_win->local_ug_comm, &ug_local_nprocs);
    PMPI_Comm_size(ug_win->ug_comm, &ug_nprocs);
    PMPI_Comm_rank(ug_win->ug_comm, &ug_rank);
    PMPI_Comm_rank(ug_win->ug_comm, &ug_win->my_rank_in_ug_comm);
    CSP_DBG_PRINT(" Created ug_comm: %d/%d, local_ug_comm: %d/%d\n",
                  ug_rank, ug_nprocs, ug_local_rank, ug_local_nprocs);

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    ug_win->g_ops_counts = CSP_calloc(ug_nprocs, sizeof(int));
    ug_win->g_bytes_counts = CSP_calloc(ug_nprocs, sizeof(unsigned long));
#endif

    /* Allocate a shared window with local ghosts.
     * Always set alloc_shm to true, same_size to false for the shared internal window.
     *
     * We only pass user specified alloc_shm to win_create windows.
     * - If alloc_shm is true, MPI implementation can still provide shm optimization;
     * - If alloc_shm is false, those win_create windows are just handled as normal windows in MPI. */
    if (info != MPI_INFO_NULL) {
        mpi_errno = PMPI_Info_dup(info, &shared_info);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
        mpi_errno = PMPI_Info_set(shared_info, "alloc_shm", "true");
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
        mpi_errno = PMPI_Info_set(shared_info, "same_size", "false");
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    mpi_errno = PMPI_Win_allocate_shared(size, disp_unit, shared_info, ug_win->local_ug_comm,
                                         &ug_win->base, &ug_win->local_ug_win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;
    CSP_DBG_PRINT("[%d] allocate shared base = %p\n", user_rank, ug_win->base);

    /* Gather user offsets on corresponding ghost processes */
    mpi_errno = gather_base_offsets(ug_win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* Bind window to main ghost process */
    mpi_errno = CSP_win_bind_ghosts(ug_win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* Create windows using shared buffers. */
    if ((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) ||
        (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK_ALL)) {

        mpi_errno = create_lock_windows(size, disp_unit, info, ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    /* - Create global active window */
    if ((ug_win->info_args.epoch_type & CSP_EPOCH_FENCE) ||
        (ug_win->info_args.epoch_type & CSP_EPOCH_PSCW)) {
        mpi_errno = PMPI_Win_create(ug_win->base, size, disp_unit, info,
                                    ug_win->ug_comm, &ug_win->active_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        CSP_DBG_PRINT("[%d] Created active window 0x%x\n", user_rank, ug_win->active_win);

        /* Since all processes must be in win_allocate, we do not need worry
         * the possibility losing asynchronous progress.
         * This lock_all guarantees the semantics correctness when internally
         * change to passive mode. */
        mpi_errno = PMPI_Win_lock_all(MPI_MODE_NOCHECK, ug_win->active_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    /* Track epoch status for redirecting RMA to different window. */
    ug_win->epoch_stat = CSP_WIN_NO_EPOCH;
    for (i = 0; i < user_nprocs; i++)
        ug_win->targets->epoch_stat = CSP_TARGET_NO_EPOCH;

    ug_win->start_counter = 0;
    ug_win->lock_counter = 0;

    /* - Only expose user window in order to hide ghosts in all non-wrapped window functions */
    mpi_errno = PMPI_Win_create(ug_win->base, size, disp_unit, info,
                                ug_win->user_comm, &ug_win->win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    CSP_DBG_PRINT("[%d] Created window 0x%x\n", user_rank, ug_win->win);

    ug_win->create_flavor = MPI_WIN_FLAVOR_ALLOCATE;
    *win = ug_win->win;
    *base_pp = ug_win->base;

    /* Gather the handle of Ghosts' win. User root is always rank num_g in user
     * root + ghosts communicator */
    /* TODO:
     * How about use handler on user root ?
     * How to solve the case that different processes may have the same handler ? */
    if (user_local_rank == 0) {
        unsigned long tmp_send_buf;
        ug_win->g_win_handles = CSP_calloc(CSP_ENV.num_g + 1, sizeof(unsigned long));
        mpi_errno = PMPI_Gather(&tmp_send_buf, 1, MPI_UNSIGNED_LONG, ug_win->g_win_handles,
                                1, MPI_UNSIGNED_LONG, CSP_ENV.num_g, ug_win->ur_g_comm);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    CSP_cache_ug_win(ug_win->win, ug_win);

  fn_exit:
    if (shared_info && shared_info != MPI_INFO_NULL)
        PMPI_Info_free(&shared_info);
    if (tmp_gather_buf)
        free(tmp_gather_buf);

    CSP_rm_count_end(CSP_RM_COMM_FREQ);
    return mpi_errno;

  fn_noasync:
    CSP_win_release(ug_win);

    mpi_errno = PMPI_Win_allocate(size, disp_unit, info, user_comm, baseptr, win);
    CSP_DBG_PRINT("Async is turned off in win_allocate, return normal win 0x%x\n", *win);

    goto fn_exit;

  fn_fail:

    /* Caching is the last possible error, so we do not need remove
     * cache here. */

    CSP_win_release(ug_win);

    *win = MPI_WIN_NULL;
    *base_pp = NULL;

    goto fn_exit;
}
