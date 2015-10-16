#
# Copyright (C) 2015. See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/perf/include -I$(includedir) 

noinst_PROGRAMS = \
	overhead_win_alloc	\
	csp_overhead_win_alloc \
	overhead_win_fence \
	csp_overhead_win_fence \
	overhead_win_setinfo	\
	csp_overhead_win_setinfo

csp_overhead_win_alloc_SOURCES= overhead_win_alloc.c
csp_overhead_win_alloc_LDADD= -L$(libdir) -lcasper
csp_overhead_win_alloc_LDFLAGS= -Wl,-rpath -Wl,$(libdir)
csp_overhead_win_alloc_CFLAGS= -O2 -DENABLE_CSP

csp_overhead_win_fence_SOURCES= overhead_win_fence.c
csp_overhead_win_fence_LDADD= -L$(libdir) -lcasper
csp_overhead_win_fence_LDFLAGS= -Wl,-rpath -Wl,$(libdir)
csp_overhead_win_fence_CFLAGS= -O2 -DENABLE_CSP

csp_overhead_win_setinfo_SOURCES= overhead_win_setinfo.c
csp_overhead_win_setinfo_LDADD= -L$(libdir) -lcasper
csp_overhead_win_setinfo_LDFLAGS= -Wl,-rpath -Wl,$(libdir)
csp_overhead_win_setinfo_CFLAGS= -O2 -DENABLE_CSP
