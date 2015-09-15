/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef CSP_QUEUE_H_
#define CSP_QUEUE_H_

#include <stdio.h>
#include <stdlib.h>

typedef struct CSP_queue_elem {
    void *ubuf;
    struct CSP_queue_elem *next;
} CSP_queue_elem_t;

typedef struct CSP_queue {
    CSP_queue_elem_t *head;
    CSP_queue_elem_t *tail;
    int count;
} CSP_queue_t;

static inline int CSP_queue_count(CSP_queue_t * queue)
{
    return queue->count;
}

extern void CSP_queue_init(CSP_queue_t * queue);
extern void CSP_queue_destroy(CSP_queue_t * queue);
extern void *CSP_queue_dequeue(CSP_queue_t * queue);
extern int CSP_queue_enqueue(void *buf, CSP_queue_t * queue);

#endif /* CSP_QUEUE_H_ */
