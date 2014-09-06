/*
 * win_post.c
 *  <FILE_DESC>
 * 	
 *  Author: Min Si
 */

#include <stdio.h>
#include <stdlib.h>
#include "mtcore.h"

static int fill_ranks_in_win_grp(MTCORE_Win * uh_win)
{
    int mpi_errno = MPI_SUCCESS;
    int *ranks_in_post_grp = NULL;
    int i, post_grp_size;

    mpi_errno = PMPI_Group_size(uh_win->post_group, &post_grp_size);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    ranks_in_post_grp = calloc(post_grp_size, sizeof(int));
    for (i = 0; i < post_grp_size; i++) {
        ranks_in_post_grp[i] = i;
    }

    mpi_errno = PMPI_Group_translate_ranks(uh_win->post_group, post_grp_size,
                                           ranks_in_post_grp, uh_win->user_group,
                                           uh_win->post_ranks_in_win_group);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

  fn_exit:
    if (ranks_in_post_grp)
        free(ranks_in_post_grp);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPI_Win_post(MPI_Group group, int assert, MPI_Win win)
{
    MTCORE_Win *uh_win;
    int mpi_errno = MPI_SUCCESS;
    int post_grp_size = 0;

    MTCORE_Fetch_uh_win_from_cache(win, uh_win);

    if (group == MPI_GROUP_NULL) {
        /* standard says do nothing for empty group */
        MTCORE_DBG_PRINT("Post empty group\n");
        return mpi_errno;
    }

    mpi_errno = PMPI_Group_size(group, &post_grp_size);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    if (post_grp_size <= 0) {
        /* standard says do nothing for empty group */
        MTCORE_DBG_PRINT("Post empty group\n");
        return mpi_errno;
    }

    uh_win->post_group = group;
    uh_win->post_ranks_in_win_group = calloc(post_grp_size, sizeof(int));
    MTCORE_DBG_PRINT("post group %p, size %d\n", uh_win->post_group, post_grp_size);

    /* Both lock and start only allow no_check assert. */
    assert = (assert == MPI_MODE_NOCHECK) ? MPI_MODE_NOCHECK : 0;

    mpi_errno = fill_ranks_in_win_grp(uh_win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

  fn_exit:
    return mpi_errno;

  fn_fail:
    if (uh_win->post_ranks_in_win_group)
        free(uh_win->post_ranks_in_win_group);
    uh_win->post_group = MPI_GROUP_NULL;
    uh_win->post_ranks_in_win_group = NULL;

    return mpi_errno;
}