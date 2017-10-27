#
# Copyright (C) 2016. See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS = -I$(top_srcdir)/include
AM_LDFLAGS = -Wl,-rpath -Wl,$(libdir)
LDADD = -L$(libdir) -lcasper 

noinst_PROGRAMS = \
    csp_adpt_all2n_bench_nop_blk_3d_put    \
    csp_adpt_all2n_bench_nop_blk_3d_get    \
    csp_adpt_all2n_bench_nop_blk_3d_acc    \
    csp_adpt_all2n_bench_nop_blk_3d_gacc   \
    csp_adpt_all2n_bench_nop_blk_contig_put    \
    csp_adpt_all2n_bench_nop_blk_contig_get    \
    csp_adpt_all2n_bench_nop_blk_contig_acc    \
    csp_adpt_all2n_bench_nop_blk_contig_gacc   \
    csp_adpt_all2n_bench_nop_blk_contig_fop    \
    csp_adpt_all2n_bench_nop_blk_contig_cas    \
    csp_adpt_all2n_bench_nop_nonblk_3d_put    \
    csp_adpt_all2n_bench_nop_nonblk_3d_get    \
    csp_adpt_all2n_bench_nop_nonblk_3d_acc    \
    csp_adpt_all2n_bench_nop_nonblk_3d_gacc   \
    csp_adpt_all2n_bench_nop_nonblk_contig_put    \
    csp_adpt_all2n_bench_nop_nonblk_contig_get    \
    csp_adpt_all2n_bench_nop_nonblk_contig_acc    \
    csp_adpt_all2n_bench_nop_nonblk_contig_gacc   \
    csp_adpt_all2n_bench_nop_nonblk_contig_fop    \
    csp_adpt_all2n_bench_nop_nonblk_contig_cas

csp_adpt_all2n_bench_nop_blk_3d_put_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_blk_3d_put_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_PUT
csp_adpt_all2n_bench_nop_blk_3d_put_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_blk_3d_put_LDADD= -L$(libdir) -lcasper 

csp_adpt_all2n_bench_nop_blk_3d_get_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_blk_3d_get_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GET
csp_adpt_all2n_bench_nop_blk_3d_get_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_blk_3d_get_LDADD= -L$(libdir) -lcasper 

csp_adpt_all2n_bench_nop_blk_3d_acc_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_blk_3d_acc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_ACC
csp_adpt_all2n_bench_nop_blk_3d_acc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_blk_3d_acc_LDADD= -L$(libdir) -lcasper 

csp_adpt_all2n_bench_nop_blk_3d_gacc_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_blk_3d_gacc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GETACC
csp_adpt_all2n_bench_nop_blk_3d_gacc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_blk_3d_gacc_LDADD= -L$(libdir) -lcasper

csp_adpt_all2n_bench_nop_blk_contig_fop_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_blk_contig_fop_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_FOP -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_blk_contig_fop_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_blk_contig_fop_LDADD= -L$(libdir) -lcasper

csp_adpt_all2n_bench_nop_blk_contig_cas_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_blk_contig_cas_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_CAS -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_blk_contig_cas_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_blk_contig_cas_LDADD= -L$(libdir) -lcasper

csp_adpt_all2n_bench_nop_blk_contig_put_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_blk_contig_put_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_PUT -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_blk_contig_put_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_blk_contig_put_LDADD= -L$(libdir) -lcasper 

csp_adpt_all2n_bench_nop_blk_contig_get_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_blk_contig_get_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GET -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_blk_contig_get_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_blk_contig_get_LDADD= -L$(libdir) -lcasper 

csp_adpt_all2n_bench_nop_blk_contig_acc_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_blk_contig_acc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_ACC -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_blk_contig_acc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_blk_contig_acc_LDADD= -L$(libdir) -lcasper 

csp_adpt_all2n_bench_nop_blk_contig_gacc_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_blk_contig_gacc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GETACC -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_blk_contig_gacc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_blk_contig_gacc_LDADD= -L$(libdir) -lcasper 


csp_adpt_all2n_bench_nop_nonblk_3d_put_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_nonblk_3d_put_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_PUT -DTEST_RMA_FLUSH_ALL
csp_adpt_all2n_bench_nop_nonblk_3d_put_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_nonblk_3d_put_LDADD= -L$(libdir) -lcasper 

csp_adpt_all2n_bench_nop_nonblk_3d_get_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_nonblk_3d_get_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GET -DTEST_RMA_FLUSH_ALL
csp_adpt_all2n_bench_nop_nonblk_3d_get_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_nonblk_3d_get_LDADD= -L$(libdir) -lcasper 

csp_adpt_all2n_bench_nop_nonblk_3d_acc_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_nonblk_3d_acc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_ACC -DTEST_RMA_FLUSH_ALL
csp_adpt_all2n_bench_nop_nonblk_3d_acc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_nonblk_3d_acc_LDADD= -L$(libdir) -lcasper

csp_adpt_all2n_bench_nop_nonblk_3d_gacc_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_nonblk_3d_gacc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GETACC -DTEST_RMA_FLUSH_ALL
csp_adpt_all2n_bench_nop_nonblk_3d_gacc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_nonblk_3d_gacc_LDADD= -L$(libdir) -lcasper


csp_adpt_all2n_bench_nop_nonblk_contig_put_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_nonblk_contig_put_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_PUT -DTEST_RMA_FLUSH_ALL -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_nonblk_contig_put_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_nonblk_contig_put_LDADD= -L$(libdir) -lcasper 

csp_adpt_all2n_bench_nop_nonblk_contig_get_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_nonblk_contig_get_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GET -DTEST_RMA_FLUSH_ALL -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_nonblk_contig_get_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_nonblk_contig_get_LDADD= -L$(libdir) -lcasper 

csp_adpt_all2n_bench_nop_nonblk_contig_acc_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_nonblk_contig_acc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_ACC -DTEST_RMA_FLUSH_ALL -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_nonblk_contig_acc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_nonblk_contig_acc_LDADD= -L$(libdir) -lcasper

csp_adpt_all2n_bench_nop_nonblk_contig_gacc_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_nonblk_contig_gacc_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_GETACC -DTEST_RMA_FLUSH_ALL -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_nonblk_contig_gacc_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_nonblk_contig_gacc_LDADD= -L$(libdir) -lcasper

csp_adpt_all2n_bench_nop_nonblk_contig_fop_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_nonblk_contig_fop_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_FOP -DTEST_RMA_FLUSH_ALL -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_nonblk_contig_fop_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_nonblk_contig_fop_LDADD= -L$(libdir) -lcasper

csp_adpt_all2n_bench_nop_nonblk_contig_cas_SOURCES= adpt_all2n_bench_nop.c
csp_adpt_all2n_bench_nop_nonblk_contig_cas_CFLAGS= -O2 -DENABLE_CSP -DTEST_RMA_OP_CAS -DTEST_RMA_FLUSH_ALL -DTEST_RMA_CONTIG
csp_adpt_all2n_bench_nop_nonblk_contig_cas_LDFLAGS= -Wl,-rpath -Wl,$(libdir) 
csp_adpt_all2n_bench_nop_nonblk_contig_cas_LDADD= -L$(libdir) -lcasper