/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "casper.h"
#include "csp.h"

#ifdef ENABLE_PROFILE

void csp_profile_reset_counter_(void)
{
    CSP_profile_reset_counter();
}

void csp_profile_reset_timing_(void)
{
    CSP_profile_reset_timing();
}

void csp_profile_print_timing_(char *name)
{
    CSP_profile_print_timing(name);
}

void csp_profile_print_counter_(char *name)
{
    CSP_profile_print_counter(name);
}
#else
void csp_profile_reset_counter_(void)
{
}

void csp_profile_reset_timing_(void)
{
}

void csp_profile_print_timing_(char *name)
{
}

void csp_profile_print_counter_(char *name)
{
}
#endif
