/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef CSP_H_
#define CSP_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <casperconf.h>

#include "rm.h"

/* #define CSP_ENABLE_GRANT_LOCK_HIDDEN_BYTE */

/* #define CSP_ENABLE_LOCAL_LOCK_OPT */
/* Optimization for local target.
 * Lock/RMA/Flush/Unlock local target instead of ghosts.
 * Only available when local lock is granted. */

#ifdef CSP_ENABLE_GRANT_LOCK_HIDDEN_BYTE
#define CSP_GRANT_LOCK_DATATYPE char
#define CSP_GRANT_LOCK_MPI_DATATYPE MPI_CHAR
#endif

#define CSP_SEGMENT_UNIT 16

#define CSP_PSCW_CW_TAG 900
#define CSP_PSCW_PS_TAG 901

/*FIXME: It is a workaround for shared window overlapping problem
 * when shared segment size of each ghost is 0 */
#define CSP_GP_SHARED_SG_SIZE 0


/* Generic MACROs.
 * ====================================================================== */

#ifndef CSP_unlikely
#ifdef HAVE_BUILTIN_EXPECT
#  define CSP_unlikely(x_) __builtin_expect(!!(x_),0)
#else
#  define CSP_unlikely(x_) (x_)
#endif
#endif /* CSP_unlikely */

#ifndef CSP_ATTRIBUTE
#ifdef HAVE_GCC_ATTRIBUTE
#define CSP_ATTRIBUTE(a_) __attribute__(a_)
#else
#define CSP_ATTRIBUTE(a_)
#endif
#endif /* CSP_ATTRIBUTE */

/* Note that, it is recommended to only pass single variables to the following MACROs.
 * Because these input arguments may be executed twice, thus it is risky to use
 * functions if it updates a global state. */
#ifndef CSP_max
#define CSP_max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef CSP_min
#define CSP_min(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef CSP_align
#define CSP_align(val, align) (((val) + (align) - 1) & ~((align) - 1))
#endif

/* ====================================================================== */


/* Casper debugging/info/warning/error MACROs.
 * ====================================================================== */

#ifdef CSP_DEBUG
#define CSP_DBG_PRINT(str,...) do { \
    fprintf(stdout, "[CSP][%d]"str, CSP_MY_RANK_IN_WORLD, ## __VA_ARGS__); \
    fflush(stdout); \
    } while (0)
#else
#define CSP_DBG_PRINT(str,...) {}
#endif

/* #define WARN */
#ifdef CSP_WARN
#define CSP_WARN_PRINT(str,...) do { \
    fprintf(stdout, "[CSP][%d]"str, CSP_MY_RANK_IN_WORLD, ## __VA_ARGS__); \
    fflush(stdout); \
    } while (0)
#else
#define CSP_WARN_PRINT(str,...) {}
#endif

#define CSP_INFO_PRINT(level, str, ...) do { \
    if (CSP_ENV.verbose > 0 && CSP_ENV.verbose >= level) { \
        fprintf(stdout, str, ## __VA_ARGS__); \
        fflush(stdout); \
    }   \
    } while (0)


extern FILE *CSP_appending_fp;
#define CSP_INFO_PRINT_FILE_START(level, fname) do { \
    if (CSP_ENV.file_verbose > 0 && CSP_ENV.file_verbose >= level) {                      \
        CSP_assert(CSP_appending_fp == NULL); /* ensure all previous files are closed */  \
        CSP_appending_fp = fopen(fname, "a");                                             \
    }                                                                                     \
    } while (0)

#define CSP_INFO_PRINT_FILE_APPEND(level, str, ...) do { \
    if (CSP_ENV.file_verbose > 0 && CSP_ENV.file_verbose >= level) { \
        if (CSP_appending_fp != NULL) {                               \
            fprintf(CSP_appending_fp, str, ## __VA_ARGS__);          \
            fflush(CSP_appending_fp);                                \
        }                                                            \
    }                                                                \
    } while (0)
#define CSP_INFO_PRINT_FILE_END(level, fname) do { \
    if (CSP_ENV.file_verbose > 0 && CSP_ENV.file_verbose >= level) { \
        if (CSP_appending_fp != NULL)                                \
            fclose(CSP_appending_fp);                                \
        CSP_appending_fp = NULL;                                     \
    }                                                                \
    } while (0)

#define CSP_INFO_PRINT_FILE(level, fname, str, ...) do { \
    if (CSP_ENV.file_verbose > 0 && CSP_ENV.file_verbose >= level) { \
        FILE *fp = fopen(fname, "a");                                \
        if (fp != NULL) {                                            \
            fprintf(fp, str, ## __VA_ARGS__);                        \
            fflush(fp);                                              \
            fclose(fp);                                              \
        }                                                            \
    }                                                                \
    } while (0)

#define CSP_DBG_PRINT_FCNAME() CSP_DBG_PRINT("in %s\n", __FUNCTION__)
#define CSP_ERR_PRINT(str,...) do { \
    fprintf(stderr, "[CSP][%d]"str, CSP_MY_RANK_IN_WORLD, ## __VA_ARGS__); \
    fflush(stdout); \
    } while (0)

#define CSP_assert(EXPR) do { if (CSP_unlikely(!(EXPR))){ \
            CSP_ERR_PRINT("  assert fail in [%s:%d]: \"%s\"\n", \
                          __FILE__, __LINE__, #EXPR); \
            PMPI_Abort(MPI_COMM_WORLD, -1); \
        }} while (0)

/* ====================================================================== */


typedef enum {
    CSP_LOAD_OPT_STATIC,
    CSP_LOAD_OPT_RANDOM,
    CSP_LOAD_OPT_COUNTING,
    CSP_LOAD_BYTE_COUNTING
} CSP_load_opt;

typedef enum {
    CSP_LOAD_LOCK_NATURE,
    CSP_LOAD_LOCK_FORCE
} CSP_load_lock;

typedef enum {
    CSP_LOCK_BINDING_RANK,
    CSP_LOCK_BINDING_SEGMENT
} CSP_lock_binding;

typedef enum {
    CSP_ASYNC_CONFIG_ON = 0,
    CSP_ASYNC_CONFIG_OFF = 1
#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
        , CSP_ASYNC_CONFIG_AUTO = 2
#endif
} CSP_async_config;

#define CSP_DEFAULT_SEG_SIZE 4096;
#define CSP_DEFAULT_NG 1

typedef struct CSP_env_param {
    int num_g;
    int seg_size;               /* segment size in lock segment binding */
    CSP_load_opt load_opt;      /* runtime load balancing options */
    CSP_load_lock load_lock;    /* how to grant locks for runtime load balancing */

    /* Options for lock permission controlling among multiple ghosts.
     *
     * Since RMA Ops to a given target may be distributed to different ghosts
     * and locks will be guaranteed to be acquired only when an Op happens,
     * two origins may access a target concurrently if their Ops are distributed
     * to different ghosts.
     *
     *  Rank binding:
     *      Statically specify single ghost for each target, thus real locks/Ops
     *      to a given target will only be issued to the same ghost.
     *
     *  Segment binding:
     *      Statically specify single ghost for each segment of shared memory,
     *      thus real locks/Ops to a given byte will only be issued to the same
     *      ghost. This method has additional overhead especially for derived
     *      target datatype, but it is more fine-grained than Rank binding. */
    CSP_lock_binding lock_binding;

    int verbose;                /* verbose level. print configuration information. */
    int file_verbose;           /* verbose level. print configuration information in files. */
    CSP_async_config async_config;

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    /* runtime scheduling options for asynchronous progress configuration */
    unsigned long long async_sched_thr_l;       /* low threshold */
    unsigned long long async_sched_thr_h;       /* high threshold */
#endif
} CSP_env_param;

/* used in runtime load balancing */
typedef enum {
    CSP_MAIN_LOCK_RESET,
    CSP_MAIN_LOCK_OP_ISSUED,
    CSP_MAIN_LOCK_GRANTED
} CSP_main_lock_stat;

typedef enum {
    CSP_TARGET_NO_EPOCH,
    CSP_TARGET_EPOCH_LOCK,
    CSP_TARGET_EPOCH_PSCW
} CSP_target_epoch_stat;

typedef enum {
    CSP_WIN_NO_EPOCH,
    CSP_WIN_EPOCH_FENCE,
    CSP_WIN_EPOCH_LOCK_ALL,
    CSP_WIN_EPOCH_PER_TARGET
} CSP_win_epoch_stat;

typedef enum {
    CSP_WIN_NO_EXP_EPOCH,
    CSP_WIN_EXP_EPOCH_FENCE,
    CSP_WIN_EXP_EPOCH_PSCW
} CSP_win_exp_epoch_stat;

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
typedef enum {
    CSP_TARGET_ASYNC_ON = 0,
    CSP_TARGET_ASYNC_OFF = 1,
    CSP_TARGET_ASYNC_NONE = 99  /* initial state */
} CSP_target_async_stat;
#endif

typedef enum {
    CSP_FUNC_NULL,
    CSP_FUNC_WIN_ALLOCATE,
    CSP_FUNC_WIN_FREE,
    CSP_FUNC_LOCL_ALL,
    CSP_FUNC_UNLOCK_ALL,
    CSP_FUNC_ABORT,
    CSP_FUNC_FINALIZE,
    CSP_FUNC_MAX
} CSP_func;

typedef enum {
    CSP_EPOCH_LOCK_ALL = 1,
    CSP_EPOCH_LOCK = 2,
    CSP_EPOCH_PSCW = 4,
    CSP_EPOCH_FENCE = 8
} CSP_epoch_type;

struct CSP_win_info_args {
    unsigned short no_local_load_store;
    int epoch_type;
    CSP_async_config async_config;
    char win_name[MPI_MAX_OBJECT_NAME + 1];
};

typedef struct CSP_op_segment {
    void *origin_addr;
    int origin_count;
    MPI_Datatype origin_datatype;

    int target_rank;
    int target_seg_off;
    MPI_Aint target_disp;
    int target_count;
    int target_dtsize;
    MPI_Datatype target_datatype;

} CSP_op_segment;

typedef struct CSP_win_target_seg {
    MPI_Aint base_offset;
    int size;

    int main_g_off;
    MPI_Win ug_win;

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    CSP_main_lock_stat main_lock_stat;
#endif
} CSP_win_target_seg;

typedef struct CSP_win_target {
    MPI_Win ug_win;             /* Do not free the window, it is freed in ug_wins */
    int disp_unit;
    MPI_Aint size;

    MPI_Aint *base_g_offsets;   /* CSP_ENV.num_g */
    int *g_ranks_in_ug;         /* CSP_ENV.num_g */
    int remote_lock_assert;

    int local_user_rank;        /* rank in local user communicator */
    int local_user_nprocs;
    int world_rank;             /* rank in world communicator */
    int user_world_rank;        /* rank in user world communicator */
    int ug_rank;                /* rank in user-ghost communicator */
    int node_id;

    MPI_Aint wait_counter_offset;       /* counter for complete-wait synchronization. allocated in main ghost. */
    MPI_Aint post_flg_offset;   /* flag for post-start synchronization. allocated in main ghost. */

    /* Only contain 1 segment in rank binding */
    CSP_win_target_seg *segs;
    int num_segs;

    CSP_target_epoch_stat epoch_stat;   /* indicate which access epoch is opened for the target. */
#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    CSP_target_async_stat async_stat;   /*per-target async status when window async config is auto. */
#endif
} CSP_win_target;

typedef struct CSP_win {
    /* communicator including root user processes and all ghosts,
     * used for internal information exchange between users and ghosts */
    MPI_Comm ur_g_comm;

    /* communicator including local process and ghosts */
    MPI_Comm local_ug_comm;
    MPI_Group local_ug_group;
    MPI_Win local_ug_win;

    int num_g_ranks_in_ug;      /* number of unique ghost ranks */
    int *g_ranks_in_ug;         /* unique ghost ranks in world, used in lockall only epoches. */
    int my_rank_in_ug_comm;     /* remember my rank in internal ug_comm for local RMA. Specified in win_allocate. */
    unsigned short is_self_locked;

    /* communicator including all the user processes and ghosts */
    MPI_Comm ug_comm;
    MPI_Group ug_group;
    MPI_Win *ug_wins;           /* every local process has separate window for permission control,
                                 * processes in different node share one window. */
    int num_ug_wins;            /* = max_local_user_nprocs */

    /* communicator including all the user processes */
    MPI_Comm user_comm;
    MPI_Group user_group;
    MPI_Comm user_root_comm;

    MPI_Comm local_user_comm;
    int max_local_user_nprocs;
    int num_nodes;
    int node_id;

    CSP_win_epoch_stat epoch_stat;      /* indicate which access epoch is opened. Thus operations
                                         * can send to the correct window. Note that only
                                         * change from PER_TARGET to NO_EPOCH when both lock counter
                                         * and start counter are equal to 0, otherwise should check
                                         * per-target epoch status. */
    CSP_win_exp_epoch_stat exp_epoch_stat;      /* indicate which exposure epoch is opened.
                                                 * For now only post-wait/test uses it to avoid duplicate receive.*/
    int lock_counter;
    int start_counter;

    MPI_Win active_win;

    MPI_Group start_group;
    MPI_Group post_group;
    int *start_ranks_in_win_group;
    int *post_ranks_in_win_group;
    MPI_Request *wait_reqs;     /* requests for receiving complete-wait synchronization messages. */

    void *base;
    MPI_Win win;
    CSP_win_target *targets;

    unsigned long *g_win_handles;

#ifdef CSP_ENABLE_GRANT_LOCK_HIDDEN_BYTE
    MPI_Aint grant_lock_g_offset;       /* Hidden byte for granting lock on Ghost0 */
#endif

    struct CSP_win_info_args info_args;

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    int prev_g_off;
    int *g_ops_counts;          /* cnt = g_ops_counts[g_rank_in_ug] */
    unsigned long *g_bytes_counts;      /* byte = g_ops_bytes[g_rank_in_ug] */
#endif

    /* constant flavor attribute to override real flavor when user queries. */
    MPIR_Win_flavor_t create_flavor;

} CSP_win;

typedef struct CSP_func_info {
    CSP_func FUNC;
    int user_nprocs;
    int user_local_nprocs;
} CSP_func_info;

#define CSP_FUNC_TAG 9889

#define CSP_define_win_cache int UG_WIN_HANDLE_KEY = MPI_KEYVAL_INVALID
extern int UG_WIN_HANDLE_KEY;

#define CSP_init_win_cache() {    \
    mpi_errno = PMPI_Win_create_keyval(MPI_WIN_NULL_COPY_FN, \
            MPI_WIN_NULL_DELETE_FN, &UG_WIN_HANDLE_KEY, (void *) 0);    \
    if (mpi_errno != 0) \
        goto fn_fail;   \
}

#define CSP_destroy_win_cache() {    \
    if (UG_WIN_HANDLE_KEY != MPI_KEYVAL_INVALID) {  \
        mpi_errno = PMPI_Win_free_keyval(&UG_WIN_HANDLE_KEY);    \
        if (mpi_errno != MPI_SUCCESS){  \
            CSP_ERR_PRINT("Free UG_WIN_HANDLE_KEY %p\n", &UG_WIN_HANDLE_KEY);   \
        }   /*Do not jump to fn_fail, because it is also used in fn_fail processing */ \
    }   \
}

#define CSP_fetch_ug_win_from_cache(win, ug_win) { \
    int fetch_ug_win_flag = 0;   \
    mpi_errno = PMPI_Win_get_attr(win, UG_WIN_HANDLE_KEY, &ug_win, &fetch_ug_win_flag);   \
    if (!fetch_ug_win_flag || mpi_errno != MPI_SUCCESS){  \
        CSP_DBG_PRINT("Cannot fetch ug_win from win 0x%x\n", win);   \
        ug_win = NULL; \
    }   \
}

#define CSP_cache_ug_win(win, ug_win) { \
    mpi_errno = PMPI_Win_set_attr(win, UG_WIN_HANDLE_KEY, ug_win);  \
    if (mpi_errno != MPI_SUCCESS){  \
        CSP_ERR_PRINT("Cannot cache ug_win %p for win 0x%x\n", ug_win, win);   \
        goto fn_fail;   \
    }   \
    CSP_DBG_PRINT("cache ug_win %p into win 0x%x \n", ug_win, win);  \
}

#define CSP_remove_ug_win_from_cache(win)  {\
    mpi_errno = PMPI_Win_delete_attr(win, UG_WIN_HANDLE_KEY);   \
    if (mpi_errno != MPI_SUCCESS){  \
        CSP_ERR_PRINT("Cannot remove ug_win cache for win 0x%x\n", win);   \
        goto fn_fail;   \
    }   \
}

#define CSP_define_win_name_cache \
    int UG_WIN_NAME_KEY = MPI_KEYVAL_INVALID;   \
    /* track allocated name objects */  \
    int ug_win_name_malloc_cnt = 0

extern int UG_WIN_NAME_KEY;
extern int ug_win_name_malloc_cnt;

#define CSP_init_win_name_cache() {    \
    mpi_errno = PMPI_Win_create_keyval(MPI_WIN_NULL_COPY_FN, \
            MPI_WIN_NULL_DELETE_FN, &UG_WIN_NAME_KEY, (void *) 0);    \
    if (mpi_errno != 0) \
        goto fn_fail;   \
}

#define CSP_destroy_win_name_cache() {    \
    if (ug_win_name_malloc_cnt) \
        CSP_ERR_PRINT("%d name objects are not released yet\n", ug_win_name_malloc_cnt);   \
    if (UG_WIN_NAME_KEY != MPI_KEYVAL_INVALID) {  \
        mpi_errno = PMPI_Win_free_keyval(&UG_WIN_NAME_KEY);    \
        if (mpi_errno != MPI_SUCCESS){  \
            CSP_ERR_PRINT("Free UG_WIN_NAME_KEY %p\n", &UG_WIN_NAME_KEY);   \
        }   /*Do not jump to fn_fail, because it is also used in fn_fail processing */ \
    }   \
}

#define CSP_fetch_win_name_from_cache(win, name) { \
    int fetch_name_flag = 0;   \
    mpi_errno = PMPI_Win_get_attr(win, UG_WIN_NAME_KEY, &name, &fetch_name_flag);   \
    if (!fetch_name_flag || mpi_errno != MPI_SUCCESS){  \
        CSP_DBG_PRINT("Cannot fetch window name from win 0x%x\n", win);   \
        name = NULL; \
    }   \
}

#define CSP_remove_win_name_from_cache(win) { \
    char *name = NULL;  \
    int fetch_name_flag = 0;   \
    mpi_errno = PMPI_Win_get_attr(win, UG_WIN_NAME_KEY, &name, &fetch_name_flag);   \
    if (fetch_name_flag && mpi_errno == MPI_SUCCESS && name){  \
        free(name); \
        ug_win_name_malloc_cnt--;   \
    }   \
    mpi_errno = PMPI_Win_delete_attr(win, UG_WIN_NAME_KEY);   \
}

#define CSP_cache_win_name(win, name) { \
    char *name_str = NULL;  \
    name_str = CSP_calloc(1, strlen(name) + 1);    \
    strncpy(name_str, name, strlen(name));  \
    mpi_errno = PMPI_Win_set_attr(win, UG_WIN_NAME_KEY, name_str);  \
    if (mpi_errno != MPI_SUCCESS){  \
        CSP_ERR_PRINT("Cannot cache window name %p for win 0x%x\n", name, win);   \
        if (name) \
            free(name);     \
        goto fn_fail;   \
    }   \
    ug_win_name_malloc_cnt++;   \
    CSP_DBG_PRINT("cache window name %p into win 0x%x \n", name, win);  \
}

extern MPI_Comm CSP_COMM_USER_WORLD;
extern MPI_Comm CSP_COMM_LOCAL;
extern MPI_Comm CSP_COMM_USER_LOCAL;
extern MPI_Comm CSP_COMM_UR_WORLD;
extern MPI_Comm CSP_COMM_GHOST_LOCAL;
extern MPI_Group CSP_GROUP_WORLD;
extern MPI_Group CSP_GROUP_LOCAL;
extern MPI_Group CSP_GROUP_USER_WORLD;

extern int *CSP_G_RANKS_IN_WORLD;
extern int *CSP_G_RANKS_IN_LOCAL;
extern int *CSP_ALL_G_RANKS_IN_WORLD;
extern int *CSP_ALL_UNIQUE_G_RANKS_IN_WORLD;
extern int *CSP_USER_RANKS_IN_WORLD;
extern int CSP_NUM_NODES;
extern int CSP_MY_NODE_ID;
extern int *CSP_ALL_NODE_IDS;
extern int CSP_MY_RANK_IN_WORLD;

extern CSP_env_param CSP_ENV;

static inline void *CSP_calloc(int n, size_t size)
{
    void *buf = NULL;
    buf = malloc(n * size);
    if (buf == NULL)
        return buf;

    memset(buf, 0, n * size);
    return buf;
}

static inline int CSP_get_node_ids(MPI_Group group, int n, const int ranks[], int node_ids[])
{
    int mpi_errno = MPI_SUCCESS;
    int *ranks_in_world = NULL;
    int i;

    if (n == 0)
        return mpi_errno;

    ranks_in_world = CSP_calloc(n, sizeof(int));

    mpi_errno = PMPI_Group_translate_ranks(group, n, ranks, CSP_GROUP_WORLD, ranks_in_world);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    for (i = 0; i < n; i++) {
        node_ids[i] = CSP_ALL_NODE_IDS[ranks_in_world[i]];
    }

  fn_exit:
    if (ranks_in_world)
        free(ranks_in_world);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
#define CSP_reset_target_opload_op_counting(target_rank, ug_win) {  \
        int g_off, g_rank;  \
        for (g_off = 0; g_off < CSP_ENV.num_g; g_off++) {    \
            g_rank = ug_win->targets[target_rank].g_ranks_in_ug[g_off]; \
            ug_win->g_ops_counts[g_rank] = 0;    \
        }   \
        CSP_DBG_PRINT("[load_opt_op] reset target %d op counting \n", target_rank); \
    }

#define CSP_reset_target_opload_bytes_counting(target_rank, ug_win) {  \
        int g_off, g_rank;  \
        for (g_off = 0; g_off < CSP_ENV.num_g; g_off++) {    \
            g_rank = ug_win->targets[target_rank].g_ranks_in_ug[g_off]; \
            ug_win->g_bytes_counts[g_rank] = 0;    \
        }   \
        CSP_DBG_PRINT("[load_opt_byte] reset target %d byte counting \n", target_rank); \
    }

#define CSP_reset_target_opload(target_rank, ug_win) { \
        if (CSP_ENV.load_opt == CSP_LOAD_OPT_COUNTING){ \
            CSP_reset_target_opload_op_counting(target_rank, ug_win) ; \
        } else if (CSP_ENV.load_opt == CSP_LOAD_BYTE_COUNTING){  \
            CSP_reset_target_opload_bytes_counting(target_rank, ug_win) ; \
        }   \
    }


#define CSP_inc_target_opload_op_counting(g_rank_in_ug, ug_win) {  \
        ug_win->g_ops_counts[g_rank_in_ug]++;   \
        CSP_DBG_PRINT("[load_opt_op] increment ghost %d\n", g_rank_in_ug); \
    }

#define CSP_inc_target_opload_bytes_counting(g_rank_in_ug, size, ug_win) {  \
        ug_win->g_bytes_counts[g_rank_in_ug] += size;   \
        CSP_DBG_PRINT("[load_opt_byte] increment ghost %d\n", g_rank_in_ug); \
    }
#endif


static inline int CSP_win_grant_local_lock(int target_rank, CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int user_rank, j;

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

    /* force lock all the main ghosts for each segment */
    for (j = 0; j < ug_win->targets[target_rank].num_segs; j++) {
        int main_g_off = ug_win->targets[target_rank].segs[j].main_g_off;
        int target_g_rank_in_ug = ug_win->targets[target_rank].g_ranks_in_ug[main_g_off];

#ifdef CSP_ENABLE_GRANT_LOCK_HIDDEN_BYTE
        CSP_GRANT_LOCK_DATATYPE buf[1];
        mpi_errno = PMPI_Get(buf, 1, CSP_GRANT_LOCK_MPI_DATATYPE, target_g_rank_in_ug,
                             ug_win->grant_lock_g_offset, 1, CSP_GRANT_LOCK_MPI_DATATYPE,
                             ug_win->targets[target_rank].segs[j].ug_win);
#else
        /* Simply get 1 byte from start, it does not affect the result of other updates */
        char buf[1];
        mpi_errno = PMPI_Get(buf, 1, MPI_CHAR, target_g_rank_in_ug, 0,
                             1, MPI_CHAR, ug_win->targets[user_rank].segs[j].ug_win);
#endif
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        mpi_errno = PMPI_Win_flush(target_g_rank_in_ug,
                                   ug_win->targets[target_rank].segs[j].ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
        ug_win->targets[target_rank].segs[j].main_lock_stat = CSP_MAIN_LOCK_GRANTED;
#endif
        CSP_DBG_PRINT("[%d]grant local lock(Ghost(%d), ug_wins 0x%x) seg %d\n", user_rank,
                      target_g_rank_in_ug, ug_win->targets[target_rank].segs[j].ug_win, j);

    }

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

extern const char *CSP_target_epoch_stat_name[4];       /* for debug */
extern const char *CSP_win_epoch_stat_name[4];

/* Get appropriate window for the target on the current epoch.
 * The epoch status can be per-target (pscw, lock), or global (fence, lockall). */
#define CSP_target_get_epoch_win(seg, target, ug_win, win_ptr) { \
    if (ug_win->epoch_stat == CSP_WIN_EPOCH_PER_TARGET) {    \
        switch (target->epoch_stat) {   \
            case CSP_TARGET_EPOCH_PSCW:    \
                win_ptr = &ug_win->active_win;   \
                break;  \
            case CSP_TARGET_EPOCH_LOCK:    \
                win_ptr = &target->segs[seg].ug_win;   \
                break;  \
            case CSP_TARGET_NO_EPOCH:   \
                win_ptr = NULL; \
                break;  \
        }   \
    } else {    \
        switch (ug_win->epoch_stat) {   \
            case CSP_WIN_EPOCH_FENCE:    \
                win_ptr = &ug_win->active_win;   \
                break;  \
            case CSP_WIN_EPOCH_LOCK_ALL:    \
                if (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) {  \
                    win_ptr = &target->segs[seg].ug_win;   \
                } else {    \
                    win_ptr = &ug_win->active_win;   \
                }   \
                break;  \
            case CSP_WIN_NO_EPOCH:   \
            case CSP_WIN_EPOCH_PER_TARGET: /* never go here */  \
                win_ptr = NULL; \
                break;  \
        }   \
    }   \
}

/* Check access epoch status per operation.*/
#define CSP_target_check_epoch_per_op(target, ug_win) {   \
    if (ug_win->epoch_stat == CSP_WIN_NO_EPOCH && target->epoch_stat == CSP_TARGET_NO_EPOCH) {  \
        CSP_ERR_PRINT("Wrong synchronization call! "    \
                "No opening epoch in %s\n", __FUNCTION__);  \
        mpi_errno = -1; \
        goto fn_fail;   \
    }   \
}

/* Return name of current epoch status (for debug).*/
static inline const char *CSP_target_get_epoch_stat_name(CSP_win_target * target, CSP_win * ug_win)
{
    if (ug_win->epoch_stat == CSP_WIN_EPOCH_PER_TARGET) {
        return CSP_target_epoch_stat_name[target->epoch_stat];
    }
    else {
        return CSP_win_epoch_stat_name[ug_win->epoch_stat];
    }
}

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
static inline const char *CSP_get_target_async_stat_name(CSP_target_async_stat async_stat)
{
    const char *name = "";
    switch (async_stat) {
    case CSP_TARGET_ASYNC_ON:
        name = "on";
        break;
    case CSP_TARGET_ASYNC_OFF:
        name = "off";
        break;
    case CSP_TARGET_ASYNC_NONE:
        name = "none";
        break;
    }
    return name;
}
#endif

static inline const char *CSP_get_async_config_name(CSP_async_config async_config)
{
    const char *name = "";
    switch (async_config) {
    case CSP_ASYNC_CONFIG_ON:
        name = "on";
        break;
    case CSP_ASYNC_CONFIG_OFF:
        name = "off";
        break;
#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
    case CSP_ASYNC_CONFIG_AUTO:
        name = "auto";
        break;
#endif
    }
    return name;
}

extern char CSP_epoch_types_name[128];
static inline const char *CSP_get_epoch_types_name(int epoch_types)
{
    memset(CSP_epoch_types_name, 0, sizeof(CSP_epoch_types_name));
    sprintf(CSP_epoch_types_name, "%s%s%s%s",
            ((epoch_types & CSP_EPOCH_LOCK_ALL) ? "lockall" : ""),
            ((epoch_types & CSP_EPOCH_LOCK) ? "|lock" : ""),
            ((epoch_types & CSP_EPOCH_PSCW) ? "|pscw" : ""),
            ((epoch_types & CSP_EPOCH_FENCE) ? "|fence" : ""));
    return (const char *) CSP_epoch_types_name;
}

extern int run_g_main(void);

extern int CSP_func_start(CSP_func FUNC, int user_nprocs, int user_local_nprocs);
extern int CSP_func_new_ur_g_comm(MPI_Comm * ur_g_comm);
extern int CSP_func_set_param(char *func_params, int size, MPI_Comm ur_g_comm);


#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)

static inline int CSP_win_grant_lock(int target_rank, int target_seg_off, CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;
    int main_g_off = ug_win->targets[target_rank].segs[target_seg_off].main_g_off;

    mpi_errno = PMPI_Win_flush(ug_win->targets[target_rank].g_ranks_in_ug[main_g_off],
                               ug_win->targets[target_rank].segs[target_seg_off].ug_win);
    if (mpi_errno == MPI_SUCCESS) {
        ug_win->targets[target_rank].segs[target_seg_off].main_lock_stat = CSP_MAIN_LOCK_GRANTED;

        CSP_DBG_PRINT("grant lock(Ghost(%d), ug_wins 0x%x) for target %d seg %d\n",
                      ug_win->targets[target_rank].g_ranks_in_ug[main_g_off],
                      ug_win->targets[target_rank].segs[target_seg_off].ug_win,
                      target_rank, target_seg_off);
    }

    return mpi_errno;
}

static inline void CSP_target_get_ghost_opload_by_random(int target_rank, int is_order_required,
                                                         CSP_win * ug_win,
                                                         int *target_g_rank_in_ug,
                                                         int *target_g_rank_idx,
                                                         MPI_Aint * target_g_offset)
{
    /* Randomly change ghost offset every time using a window-level global recorder */
    int idx = (ug_win->prev_g_off + 1) % CSP_ENV.num_g; /* jump to next ghost offset */
    ug_win->prev_g_off = idx;

    *target_g_rank_in_ug = ug_win->targets[target_rank].g_ranks_in_ug[idx];
    *target_g_offset = ug_win->targets[target_rank].base_g_offsets[idx];
    *target_g_rank_idx = idx;

    CSP_DBG_PRINT("[load_opt_random] randomly choose ghost %d, off 0x%lx for target %d\n",
                  *target_g_rank_in_ug, *target_g_offset, target_rank);

}

extern void CSP_target_get_ghost_opload_by_op(int target_rank, int is_order_required,
                                              CSP_win * ug_win, int *target_g_rank_in_ug,
                                              int *target_g_rank_idx, MPI_Aint * target_g_offset);
extern void CSP_target_get_ghost_opload_by_byte(int target_rank, int is_order_required,
                                                int size, CSP_win * ug_win,
                                                int *target_g_rank_in_ug,
                                                int *target_g_rank_idx, MPI_Aint * target_g_offset);

/**
 * Get ghost with dynamic load balancing.
 */
static inline int CSP_target_get_ghost(int target_rank, int target_seg_off,
                                       int is_order_required,
                                       int size, CSP_win * ug_win,
                                       int *target_g_rank_in_ug, MPI_Aint * target_g_offset)
{
    int mpi_errno = MPI_SUCCESS;
    int main_g_off = ug_win->targets[target_rank].segs[target_seg_off].main_g_off;
    int g_idx = 0;

    /* Force lock when the first operation is issued. Note that nocheck epoch
     * does not need it because no conflicting lock.*/
    if (CSP_ENV.load_lock == CSP_LOAD_LOCK_FORCE &&
        !(ug_win->targets[target_rank].remote_lock_assert & MPI_MODE_NOCHECK) &&
        ug_win->targets[target_rank].segs[target_seg_off].main_lock_stat ==
        CSP_MAIN_LOCK_OP_ISSUED) {
        mpi_errno = CSP_win_grant_lock(target_rank, target_seg_off, ug_win);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;
    }

    /* Upgrade main lock status of target if it is the first operation of that target. */
    if (ug_win->targets[target_rank].segs[target_seg_off].main_lock_stat == CSP_MAIN_LOCK_RESET) {
        ug_win->targets[target_rank].segs[target_seg_off].main_lock_stat = CSP_MAIN_LOCK_OP_ISSUED;
    }

    /* If lock has not been granted yet, we can only use the main ghost.
     * Accumulate operations have to be always sent to main ghost in order to
     * guarantee atomicity and ordering.*/
    if ((!(ug_win->targets[target_rank].remote_lock_assert & MPI_MODE_NOCHECK) &&
         ug_win->targets[target_rank].segs[target_seg_off].main_lock_stat !=
         CSP_MAIN_LOCK_GRANTED) || is_order_required) {
        /* Both serial async and byte tracking options specify the first ghost as
         * the main ghost of that user process.*/
        *target_g_rank_in_ug = ug_win->targets[target_rank].g_ranks_in_ug[main_g_off];
        *target_g_offset = ug_win->targets[target_rank].base_g_offsets[main_g_off];
        CSP_DBG_PRINT("[load_opt] use main ghost %d, off 0x%lx for target %d "
                      "seg %d (main h off %d)\n",
                      *target_g_rank_in_ug, *target_g_offset, target_rank,
                      target_seg_off, main_g_off);

        /* Need increase counters */
        if (CSP_ENV.load_opt == CSP_LOAD_OPT_COUNTING) {
            CSP_inc_target_opload_op_counting(*target_g_rank_in_ug, ug_win);
        }
        else if (CSP_ENV.load_opt == CSP_LOAD_BYTE_COUNTING) {
            CSP_inc_target_opload_bytes_counting(*target_g_rank_in_ug, size, ug_win);
        }

        return mpi_errno;
    }

    /* Runtime load balancing */
    if (CSP_ENV.load_opt == CSP_LOAD_OPT_RANDOM) {
        CSP_target_get_ghost_opload_by_random(target_rank, is_order_required, ug_win,
                                              target_g_rank_in_ug, &g_idx, target_g_offset);
    }
    else if (CSP_ENV.load_opt == CSP_LOAD_OPT_COUNTING) {
        CSP_target_get_ghost_opload_by_op(target_rank, is_order_required, ug_win,
                                          target_g_rank_in_ug, &g_idx, target_g_offset);
    }
    else if (CSP_ENV.load_opt == CSP_LOAD_BYTE_COUNTING) {
        CSP_target_get_ghost_opload_by_byte(target_rank, is_order_required, size,
                                            ug_win, target_g_rank_in_ug, &g_idx, target_g_offset);
    }

    return mpi_errno;
}
#else
/**
 * Get ghost that is statically bound with the target.
 */
static inline int CSP_target_get_ghost(int target_rank, int target_seg_off, int is_order_required CSP_ATTRIBUTE((unused)),      /* arguments used only in dynamic load */
                                       int size CSP_ATTRIBUTE((unused)), CSP_win * ug_win,
                                       int *target_g_rank_in_ug, MPI_Aint * target_g_offset)
{
    int mpi_errno = MPI_SUCCESS;
    int main_g_off = ug_win->targets[target_rank].segs[target_seg_off].main_g_off;

    *target_g_rank_in_ug = ug_win->targets[target_rank].g_ranks_in_ug[main_g_off];
    *target_g_offset = ug_win->targets[target_rank].base_g_offsets[main_g_off];
    CSP_DBG_PRINT("[opt_non] use main ghost %d, off 0x%lx for target %d seg %d\n",
                  *target_g_rank_in_ug, *target_g_offset, target_rank, target_seg_off);
    return mpi_errno;
}
#endif

extern int CSP_op_segments_decode(const void *origin_addr, int origin_count,
                                  MPI_Datatype origin_datatype,
                                  int target_rank, MPI_Aint target_disp,
                                  int target_count, MPI_Datatype target_datatype,
                                  CSP_win * ug_win, CSP_op_segment ** decoded_ops_ptr,
                                  int *num_segs);
extern int CSP_op_segments_decode_basic_datatype(const void *origin_addr, int origin_count,
                                                 MPI_Datatype origin_datatype,
                                                 int target_rank, MPI_Aint target_disp,
                                                 int target_count, MPI_Datatype target_datatype,
                                                 CSP_win * ug_win,
                                                 CSP_op_segment ** decoded_ops_ptr, int *num_segs);
extern void CSP_op_segments_destroy(CSP_op_segment ** decoded_ops_ptr);

extern int CSP_win_bind_ghosts(CSP_win * ug_win);

/* Receive buffer for receiving complete-wait sync message.
 * We don't need its value, so just use a global char variable to ensure
 * receive buffer is always allocated.*/
extern char wait_flg;
extern int CSP_recv_pscw_complete_msg(int post_grp_size, CSP_win * ug_win, int blocking, int *flag);

extern int CSP_win_release(CSP_win * ug_win);

#ifdef CSP_ENABLE_RUNTIME_ASYNC_SCHED
#define CSP_RUNTIME_ASYNC_SCHED_THR_DEFAULT_FREQ (50)

extern void CSP_ra_update_async_stat(CSP_async_config async_config);
extern CSP_target_async_stat CSP_ra_sched_async_stat();

#else
#define CSP_ra_sched_async_stat() {/*do nothing */}

#endif /* end of CSP_ENABLE_RUNTIME_ASYNC_SCHED */


#endif /* CSP_H_ */
