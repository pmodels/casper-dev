#
# Copyright (C) 2014. See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS = -I$(top_srcdir)/include
AM_LDFLAGS = -Wl,-rpath -Wl,$(libdir)
LDADD = -L$(libdir) -lcasper 

noinst_PROGRAMS = \
	get	\
	put	\
	put_l_seg	\
	acc	\
	acc_flush_local	\
	acclock	\
	getacc	\
	self_acclock	\
	no_loadstore	\
	no_check	\
	no_accumulate_ordering	\
	acc_pscw	\
	put_fence	\
	acc_get_fence	\
	fetch_and_op	\
	win_allocate	\
	win_create_acc	\
	init_thread_acc	\
	epoch_type	\
	epoch_type_assert	\
	win_allocate_info	\
	async_adapt	\
	async_adapt_local_check	\
	epoch_type_static_async_off	\
	epoch_type_static_async_reset	\
	epoch_type_static_async_off_reset

async_adapt_local_check_SOURCES= async_adapt.c
async_adapt_local_check_CFLAGS = -DUSE_LOCAL_CHECK

# check win_allocate(async_config = off)
epoch_type_static_async_off_SOURCES= epoch_type.c
epoch_type_static_async_off_CFLAGS = -DCHECK_STATIC_ASYNC_OFF

# check win_set_info
epoch_type_static_async_reset_SOURCES= epoch_type.c
epoch_type_static_async_reset_CFLAGS = -DCHECK_STATIC_ASYNC_RESET

# check win_allocate(async_config = off) with win_set_info
epoch_type_static_async_off_reset_SOURCES= epoch_type.c
epoch_type_static_async_off_reset_CFLAGS = -DCHECK_STATIC_ASYNC_OFF -DCHECK_STATIC_ASYNC_RESET

MPIEXEC=mpiexec

testing:
	./runtest 
	