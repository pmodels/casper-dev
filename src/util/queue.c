/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csp.h"
#include "queue.h"

/* Initialize a queue structure.
 * The internal structure is a single linked list, elements are always enqueued
 * from the end and dequeued from the head. */
void CSP_queue_init(CSP_queue_t * queue)
{
    queue->head = queue->tail = NULL;
    queue->count = 0;
}

/* Remove one element from the head of the queue.
 * Returns the user buffer of the removed element, and the element object is freed. */
void *CSP_queue_dequeue(CSP_queue_t * queue)
{
    CSP_queue_elem_t *elem = NULL;
    void *ubuf = NULL;

    /* empty */
    if (queue->head == NULL)
        return NULL;

    elem = queue->head;
    queue->head = elem->next;
    queue->count--;

    /* removing the last one */
    if (elem == queue->tail) {
        queue->tail = NULL;
        CSP_assert(queue->head == NULL);
        CSP_assert(queue->count == 0);
    }

    ubuf = elem->ubuf;
    free(elem);

    return ubuf;
}

/* Destroy a queue.
 * All existing elements will be freed. However, the ubuf object of each
 * element must still be freed by caller. */
void CSP_queue_destroy(CSP_queue_t * queue)
{
    while (CSP_queue_count(queue) > 0)
        CSP_queue_dequeue(queue);

    queue->head = queue->tail = NULL;
    queue->count = 0;
}

/* Insert one element into the end of the queue.
 * Returns 0 if succeed, otherwise -1. */
int CSP_queue_enqueue(void *ubuf, CSP_queue_t * queue)
{
    CSP_queue_elem_t *new_e = NULL;

    /* create new element */
    new_e = CSP_calloc(1, sizeof(CSP_queue_elem_t));
    if (new_e == NULL)
        return -1;

    new_e->ubuf = ubuf;
    new_e->next = NULL;

    /* insert at end */
    if (queue->tail) {
        queue->tail->next = new_e;
        queue->tail = new_e;
    }
    /* the first one */
    else {
        CSP_assert(queue->head == NULL);
        queue->head = queue->tail = new_e;
    }

    queue->count++;
    return 0;
}
