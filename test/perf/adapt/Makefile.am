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
	adpt_phase_pcoll_nop	\
	adpt_phase_pcoll_nop_outputall	\
	adpt_phase_pcoll_nop_disable_shm	\
	adpt_phase_pcoll_nop_disable_shm_outputall	\
	csp_adpt_phase_pcoll_nop	\
	csp_adpt_phase_pcoll_nop_outputall	\
	csp_adpt_u_phase_pcoll_nop	\
	csp_adpt_u_phase_pcoll_nop_outputall	\
	static_single_phase	\
	static_single_phase_disable_shm	\
	csp_static_single_phase	\
	csp_adpt_thresh_all2all_nop_put	\
	csp_adpt_thresh_all2all_nop_get	\
	csp_adpt_thresh_all2all_nop_acc	\
	csp_adpt_thresh_all2all_opsize_put	\
	csp_adpt_thresh_all2all_opsize_get	\
	csp_adpt_thresh_all2all_opsize_acc	\
	csp_adpt_thresh_all2n_nop_freq_put	\
	csp_adpt_thresh_all2n_nop_freq_get	\
	csp_adpt_thresh_all2n_nop_freq_acc	\
	csp_adpt_thresh_all2n_nop_flushall_freq_put	\
	csp_adpt_thresh_all2n_nop_flushall_freq_get	\
	csp_adpt_thresh_all2n_nop_flushall_freq_acc
	
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

adpt_phase_pcoll_nop_SOURCES= adpt_phase_pcoll_nop.c
adpt_phase_pcoll_nop_CFLAGS= -O2 -mkl=sequential
adpt_phase_pcoll_nop_LDFLAGS= -mkl=sequential

adpt_phase_pcoll_nop_outputall_SOURCES= adpt_phase_pcoll_nop.c
adpt_phase_pcoll_nop_outputall_CFLAGS= -O2 -mkl=sequential -DOUTPUT_ALL_PHASES
adpt_phase_pcoll_nop_outputall_LDFLAGS= -mkl=sequential

adpt_phase_pcoll_nop_disable_shm_SOURCES= adpt_phase_pcoll_nop.c
adpt_phase_pcoll_nop_disable_shm_CFLAGS= -O2 -mkl=sequential -DDISABLE_SHM
adpt_phase_pcoll_nop_disable_shm_LDFLAGS= -mkl=sequential

adpt_phase_pcoll_nop_disable_shm_outputall_SOURCES= adpt_phase_pcoll_nop.c
adpt_phase_pcoll_nop_disable_shm_outputall_CFLAGS= -O2 -mkl=sequential -DDISABLE_SHM -DOUTPUT_ALL_PHASES
adpt_phase_pcoll_nop_disable_shm_outputall_LDFLAGS= -mkl=sequential

csp_adpt_phase_pcoll_nop_SOURCES= adpt_phase_pcoll_nop.c
csp_adpt_phase_pcoll_nop_CFLAGS= -O2 -DENABLE_CSP -mkl=sequential
csp_adpt_phase_pcoll_nop_LDFLAGS= -Wl,-rpath -Wl,$(libdir) -mkl=sequential
csp_adpt_phase_pcoll_nop_LDADD= -L$(libdir) -lcasper 

csp_adpt_phase_pcoll_nop_outputall_SOURCES= adpt_phase_pcoll_nop.c
csp_adpt_phase_pcoll_nop_outputall_CFLAGS= -O2 -DENABLE_CSP -DOUTPUT_ALL_PHASES -mkl=sequential
csp_adpt_phase_pcoll_nop_outputall_LDFLAGS= -Wl,-rpath -Wl,$(libdir) -mkl=sequential
csp_adpt_phase_pcoll_nop_outputall_LDADD= -L$(libdir) -lcasper 

csp_adpt_u_phase_pcoll_nop_SOURCES= adpt_phase_pcoll_nop.c
csp_adpt_u_phase_pcoll_nop_CFLAGS= -O2 -DENABLE_CSP_ADPT_U -mkl=sequential
csp_adpt_u_phase_pcoll_nop_LDFLAGS= -Wl,-rpath -Wl,$(libdir) -mkl=sequential
csp_adpt_u_phase_pcoll_nop_LDADD= -L$(libdir) -lcasper 

csp_adpt_u_phase_pcoll_nop_outputall_SOURCES= adpt_phase_pcoll_nop.c
csp_adpt_u_phase_pcoll_nop_outputall_CFLAGS= -O2 -DENABLE_CSP_ADPT_U -DOUTPUT_ALL_PHASES -mkl=sequential
csp_adpt_u_phase_pcoll_nop_outputall_LDFLAGS= -Wl,-rpath -Wl,$(libdir) -mkl=sequential
csp_adpt_u_phase_pcoll_nop_outputall_LDADD= -L$(libdir) -lcasper 

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

csp_adpt_thresh_all2n_nop_freq_put_SOURCES= adpt_thresh_all2n_nop_freq.c
csp_adpt_thresh_all2n_nop_freq_put_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_PUT
csp_adpt_thresh_all2n_nop_freq_put_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2n_nop_freq_put_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2n_nop_freq_get_SOURCES= adpt_thresh_all2n_nop_freq.c
csp_adpt_thresh_all2n_nop_freq_get_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GET
csp_adpt_thresh_all2n_nop_freq_get_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2n_nop_freq_get_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2n_nop_freq_acc_SOURCES= adpt_thresh_all2n_nop_freq.c
csp_adpt_thresh_all2n_nop_freq_acc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_ACC
csp_adpt_thresh_all2n_nop_freq_acc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2n_nop_freq_acc_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2n_nop_flushall_freq_put_SOURCES= adpt_thresh_all2n_nop_freq.c
csp_adpt_thresh_all2n_nop_flushall_freq_put_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_PUT -DTEST_RMA_FLUSH_ALL
csp_adpt_thresh_all2n_nop_flushall_freq_put_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2n_nop_flushall_freq_put_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2n_nop_flushall_freq_get_SOURCES= adpt_thresh_all2n_nop_freq.c
csp_adpt_thresh_all2n_nop_flushall_freq_get_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GET -DTEST_RMA_FLUSH_ALL
csp_adpt_thresh_all2n_nop_flushall_freq_get_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2n_nop_flushall_freq_get_LDADD= -L$(libdir) -lcasper 

csp_adpt_thresh_all2n_nop_flushall_freq_acc_SOURCES= adpt_thresh_all2n_nop_freq.c
csp_adpt_thresh_all2n_nop_flushall_freq_acc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_ACC -DTEST_RMA_FLUSH_ALL
csp_adpt_thresh_all2n_nop_flushall_freq_acc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_thresh_all2n_nop_flushall_freq_acc_LDADD= -L$(libdir) -lcasper 

static_single_phase_SOURCES= static_single_phase.c
static_single_phase_CFLAGS= -O2 -mkl=sequential
static_single_phase_LDFLAGS= -mkl=sequential

static_single_phase_disable_shm_SOURCES= static_single_phase.c
static_single_phase_disable_shm_CFLAGS= -O2 -mkl=sequential -DDISABLE_SHM
static_single_phase_disable_shm_LDFLAGS= -mkl=sequential

csp_static_single_phase_SOURCES= static_single_phase.c
csp_static_single_phase_CFLAGS= -O2 -DENABLE_CSP -mkl=sequential
csp_static_single_phase_LDFLAGS= -Wl,-rpath -Wl,$(libdir) -mkl=sequential
csp_static_single_phase_LDADD= -L$(libdir) -lcasper 