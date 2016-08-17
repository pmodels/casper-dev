/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "cspg.h"

#ifdef CSPG_DEBUG
void CSPG_print_proc(void)
{
    int rank, nprocs, local_rank, local_nprocs;
    int local_ghost_rank, local_ghost_nprocs;

    CSP_assert(CSP_IS_GHOST);

    PMPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PMPI_Comm_size(CSP_PROC.local_comm, &local_nprocs);
    PMPI_Comm_rank(CSP_PROC.local_comm, &local_rank);
    PMPI_Comm_size(CSP_PROC.ghost.g_local_comm, &local_ghost_nprocs);
    PMPI_Comm_rank(CSP_PROC.ghost.g_local_comm, &local_ghost_rank);

    CSPG_DBG_PRINT("I am ghost: %d/%d in world, %d/%d in local\n"
                   "            %d/%d in ghost local\n"
                   "            node_id %d\n",
                   rank, nprocs, local_rank, local_nprocs,
                   local_ghost_rank, local_ghost_nprocs, CSP_PROC.node_id);
}
#endif

/* Setup global ghost-specific information. */
int CSPG_setup_proc(void)
{
    return MPI_SUCCESS;
}

int CSPG_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    int err_class = 0, errstr_len = 0;
    char err_string[MPI_MAX_ERROR_STRING];
    CSP_cmd_pkt_t pkt;
    CSP_cmd_fnc_pkt_t *fnc_pkt = &pkt.fnc;
    int exit_flag = 0;

    CSPG_DBG_PRINT(" main start\n");
    CSPG_cmd_init();

    /* Disable MPI automatic error messages. */
    mpi_errno = PMPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    while (1) {
        mpi_errno = CSPG_cmd_recv(&pkt);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        CSPG_assert(pkt.cmd_type == CSP_CMD_FNC);

        /* skip unknown command */
        if (fnc_pkt->fnc_cmd <= CSP_CMD_FNC_NONE || fnc_pkt->fnc_cmd >= CSP_CMD_FNC_MAX ||
            !fnc_cmd_handlers[fnc_pkt->fnc_cmd]) {
            CSPG_DBG_PRINT(" Received unknown FUNCTION %d\n", (int) (fnc_pkt->fnc_cmd));
            continue;
        }

        mpi_errno = fnc_cmd_handlers[fnc_pkt->fnc_cmd] (fnc_pkt, &exit_flag);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        /* Release local lock if a locked command finished. */
        if (fnc_pkt->lock_flag) {
            mpi_errno = CSPG_cmd_release_lock();
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }

        /* exit when finalize finished */
        if (exit_flag)
            goto fn_exit;
    }

    CSPG_DBG_PRINT(" main done\n");

  fn_exit:
    CSPG_cmd_destory();
    return mpi_errno;

  fn_fail:
    PMPI_Error_class(mpi_errno, &err_class);
    PMPI_Error_string(mpi_errno, err_string, &errstr_len);
    CSPG_ERR_PRINT("MPI reports error code %d, error class %d\n%s",
                   mpi_errno, err_class, err_string);
    goto fn_exit;
}
