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
	csp_overhead_win_setinfo	\
	adapt_nphase_nwin_nop_put	\
	adapt_nphase_nwin_nop_get	\
	adapt_nphase_nwin_nop_acc	\
	adpt_thresh_2np				\
	csp_adpt_thresh_2np			\
	adpt_thresh_all2all			\
	csp_adpt_thresh_all2all		\
	csp_adpt_thresh_all2all_nop_put	\
	csp_adpt_thresh_all2all_nop_get	\
	csp_adpt_thresh_all2all_nop_acc	\
	csp_adpt_thresh_all2all_opsize_put	\
	csp_adpt_thresh_all2all_opsize_get	\
	csp_adpt_thresh_all2all_opsize_acc

csp_overhead_win_alloc_SOURCES= overhead_win_alloc.c
csp_overhead_win_alloc_CFLAGS= -O2 -DENABLE_CSP
csp_overhead_win_alloc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_overhead_win_alloc_LDADD= -L$(libdir) -lcasper 

csp_overhead_win_fence_SOURCES= overhead_win_fence.c
csp_overhead_win_fence_CFLAGS= -O2 -DENABLE_CSP
csp_overhead_win_fence_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_overhead_win_fence_LDADD= -L$(libdir) -lcasper 

csp_overhead_win_setinfo_SOURCES= overhead_win_setinfo.c
csp_overhead_win_setinfo_CFLAGS= -O2 -DENABLE_CSP
csp_overhead_win_setinfo_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_overhead_win_setinfo_LDADD= -L$(libdir) -lcasper 

adapt_nphase_nwin_nop_put_SOURCES= adapt_nphase_nwin_nop.c
adapt_nphase_nwin_nop_put_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_PUT
adapt_nphase_nwin_nop_put_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
adapt_nphase_nwin_nop_put_LDADD= -L$(libdir) -lcasper 

adapt_nphase_nwin_nop_get_SOURCES= adapt_nphase_nwin_nop.c
adapt_nphase_nwin_nop_get_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GET
adapt_nphase_nwin_nop_get_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
adapt_nphase_nwin_nop_get_LDADD= -L$(libdir) -lcasper 

adapt_nphase_nwin_nop_acc_SOURCES= adapt_nphase_nwin_nop.c
adapt_nphase_nwin_nop_acc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_ACC
adapt_nphase_nwin_nop_acc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
adapt_nphase_nwin_nop_acc_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_2np_SOURCES= adpt_thresh_2np.c
csp_adpt_thresh_2np_CFLAGS= -O2 -DENABLE_CSP
csp_adpt_thresh_2np_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_2np_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2all_SOURCES= adpt_thresh_all2all.c
csp_adpt_thresh_all2all_CFLAGS= -O2 -DENABLE_CSP
csp_adpt_thresh_all2all_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2all_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2all_nop_get_SOURCES= adpt_thresh_all2all_nop.c
csp_adpt_thresh_all2all_nop_get_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GET
csp_adpt_thresh_all2all_nop_get_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2all_nop_get_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2all_nop_put_SOURCES= adpt_thresh_all2all_nop.c
csp_adpt_thresh_all2all_nop_put_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_PUT
csp_adpt_thresh_all2all_nop_put_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2all_nop_put_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2all_nop_acc_SOURCES= adpt_thresh_all2all_nop.c
csp_adpt_thresh_all2all_nop_acc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_ACC
csp_adpt_thresh_all2all_nop_acc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2all_nop_acc_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2all_opsize_get_SOURCES= adpt_thresh_all2all_opsize.c
csp_adpt_thresh_all2all_opsize_get_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GET
csp_adpt_thresh_all2all_opsize_get_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2all_opsize_get_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2all_opsize_put_SOURCES= adpt_thresh_all2all_opsize.c
csp_adpt_thresh_all2all_opsize_put_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_PUT
csp_adpt_thresh_all2all_opsize_put_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2all_opsize_put_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2all_opsize_acc_SOURCES= adpt_thresh_all2all_opsize.c
csp_adpt_thresh_all2all_opsize_acc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_ACC
csp_adpt_thresh_all2all_opsize_acc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2all_opsize_acc_LDADD= -L$(libdir) -lcasper 