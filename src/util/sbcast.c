/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csp.h"
#include "sbcast.h"

/* This file defines the synchronized split broadcast routines.
 * The completion of a synchronized split broadcast guarantees all involved
 * processes have been completed on this communication.*/

/* Issue a synchronized broadcast member call (non-blocking).
 * This call initializes the receiving for a split broadcast communication.
 * Caller must call CSP_sbcast_member_test to wait its completion.
 * Output parameter req is the request object of the synchronized broadcast
 * member call. */
int CSP_sbcast_member(void *buf, int size, MPI_Comm comm, CSP_sbcast_member_req * req)
{
    int mpi_errno = MPI_SUCCESS;
    int rank = 0;

    PMPI_Comm_rank(comm, &rank);

    if (rank == CSP_SBCAST_ROOT_AGENT) {
        mpi_errno = PMPI_Irecv(buf, size, MPI_BYTE, MPI_ANY_SOURCE,
                               CSP_SBCAST_SR_TAG, comm, &req->recv_req);
        CSP_DBG_PRINT(" sbcast_member: agent issue irecv\n");
    }
    else {
        /* non-agent processes just start broadcast from the root agent. */
        CSP_DBG_PRINT(" sbcast_member: users issue ibcast-ibarrier\n");
        mpi_errno = PMPI_Ibcast(buf, size, MPI_BYTE, CSP_SBCAST_ROOT_AGENT, comm,
                                &req->sbcast_reqs[0]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
        mpi_errno = PMPI_Ibarrier(comm, &req->sbcast_reqs[1]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    /* save info for completion call  */
    req->buf = buf;
    req->size = size;
    req->comm = comm;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Test the completion of a synchronized broadcast member call (non-blocking).
 * Output parameter flag indicates the completion (1:completed | 0:not yet). */
int CSP_sbcast_member_test(CSP_sbcast_member_req * req, int *flag)
{
    int mpi_errno = MPI_SUCCESS;
    int rank = 0;
    int wait_flag = 0;

    PMPI_Comm_rank(req->comm, &rank);
    (*flag) = 0;        /* not done */

    /* agent needs to first receive from the root, and then issues ibcast-ibarrier. */
    if (rank == CSP_SBCAST_ROOT_AGENT && req->recv_req != MPI_REQUEST_NULL) {
        int test_flag = 0;
        MPI_Status stat = MPI_STATUS_NULL;
        mpi_errno = PMPI_Test(&req->recv_req, &test_flag, &stat);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;

        /* not receive yet, return. */
        if (!test_flag)
            goto fn_exit;

        CSP_DBG_PRINT(" sbcast_member_test: agent received packet from %d\n", stat.MPI_SOURCE);

        mpi_errno = PMPI_Ibcast(req->buf, req->size, MPI_BYTE,
                                CSP_SBCAST_ROOT_AGENT, req->comm, &req->sbcast_reqs[0]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
        mpi_errno = PMPI_Ibarrier(req->comm, &req->sbcast_reqs[1]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }

    /* all processes test on ibcast-ibarrier. */
    wait_flag = 0;
    mpi_errno = PMPI_Testall(2, req->sbcast_reqs, &wait_flag, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;
    if (wait_flag == 1) {
        CSP_DBG_PRINT(" sbcast_member_test: done\n");
        (*flag) = 1;    /* done */

        /* reset other info, MPI requests are already reset by MPI */
        req->buf = NULL;
        req->comm = MPI_COMM_NULL;
        req->size = 0;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Issue a synchronized broadcast root call (non-blocking).
 * This call initializes the sending for a split broadcast communication.
 * Caller must call CSP_sbcast_root_test to wait its completion.
 * Output parameter req is the request object of the synchronized broadcast
 * root call. */
int CSP_sbcast_root(void *buf, int size, MPI_Comm comm, CSP_sbcast_root_req * req)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = PMPI_Isend(buf, size, MPI_BYTE, CSP_SBCAST_ROOT_AGENT,
                           CSP_SBCAST_SR_TAG, comm, &req->send_req);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;
    CSP_DBG_PRINT(" sbcast_root: issue isend to agent %d\n", CSP_SBCAST_ROOT_AGENT);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Test the completion of a synchronized broadcast root call (non-blocking).
 * Output parameter flag indicates the completion (1:completed | 0:not yet). */
int CSP_sbcast_root_test(CSP_sbcast_root_req * req, int *flag)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = PMPI_Test(&req->send_req, flag, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    if ((*flag) == 1) {
        CSP_DBG_PRINT(" sbcast_root_test_complete: done\n");
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
