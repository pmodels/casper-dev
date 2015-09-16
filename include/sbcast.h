/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef SBCAST_H_
#define SBCAST_H_

typedef struct CSP_sbcast_member_req {
    void *buf;
    int size;
    MPI_Comm comm;
    MPI_Request recv_req;
    MPI_Request sbcast_reqs[2];
} CSP_sbcast_member_req;

typedef struct CSP_sbcast_root_req {
    MPI_Request send_req;
} CSP_sbcast_root_req;

#define CSP_SBCAST_SR_TAG 9900
#define CSP_SBCAST_ROOT_AGENT 0

/* Check the status of a synchronized broadcast member request.
 * Return TURE if the request is empty or already completed, otherwise return FALSE.*/
static inline int CSP_sbcast_member_is_completed(CSP_sbcast_member_req req)
{
    return req.recv_req == MPI_REQUEST_NULL && req.sbcast_reqs[0] == MPI_REQUEST_NULL
        && req.sbcast_reqs[1] == MPI_REQUEST_NULL;
}

/* Check the status of a synchronized broadcast root request.
 * Return TURE if the request is empty or already completed, otherwise return FALSE. */
static inline int CSP_sbcast_root_is_completed(CSP_sbcast_root_req req)
{
    return req.send_req == MPI_REQUEST_NULL;
}

/* Initialize a synchronized broadcast root request.
 * Caller must always first initialize a request object before using it. */
static inline void CSP_sbcast_root_req_init(CSP_sbcast_root_req * req)
{
    req->send_req = MPI_REQUEST_NULL;
}

/* Initialize a synchronized broadcast member request.
 * Caller must always first initialize a request object before using it. */
static inline void CSP_sbcast_member_req_init(CSP_sbcast_member_req * req)
{
    req->buf = NULL;
    req->comm = MPI_COMM_NULL;
    req->size = 0;
    req->recv_req = MPI_REQUEST_NULL;
    req->sbcast_reqs[0] = MPI_REQUEST_NULL;
    req->sbcast_reqs[1] = MPI_REQUEST_NULL;
}

extern int CSP_sbcast_member(void *buf, int size, MPI_Comm comm, CSP_sbcast_member_req * req);
extern int CSP_sbcast_member_test(CSP_sbcast_member_req * req, int *flag);
extern int CSP_sbcast_root(void *buf, int size, MPI_Comm comm, CSP_sbcast_root_req * req);
extern int CSP_sbcast_root_test(CSP_sbcast_root_req * req, int *flag);

#endif /* SBCAST_H_ */
