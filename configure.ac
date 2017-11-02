#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([casper], [1.0])

AC_CONFIG_AUX_DIR(confdb)
AC_CONFIG_MACRO_DIR(confdb)
AM_INIT_AUTOMAKE([-Wall -Werror -Wno-portability-recursive silent-rules foreign 1.12.3 subdir-objects])

# Bug in libtool adds -O2 and -g by default
PAC_PUSH_FLAG([CFLAGS])
AC_PROG_CC(mpicc)
AM_PROG_CC_C_O
PAC_POP_FLAG([CFLAGS])

AM_PROG_AR

LT_PREREQ([2.2.6])

AC_CONFIG_HEADER([include/casperconf.h])

# Bug in libtool adds -O2 and -g by default
PAC_PUSH_FLAG([CFLAGS])
LT_INIT()
PAC_POP_FLAG([CFLAGS])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h])

# Non-verbose make
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_RESTRICT
AC_C_INLINE

# Check for enabling strict
PAC_ARG_STRICT

# Runtime load balancing optimization
AC_ARG_ENABLE(runtime-load, AC_HELP_STRING([--enable-runtime-load],[Enable Runtime Load Balancing]),
                 [ enable_runtime_load=$enableval ],
                 [ enable_runtime_load=no ])
AC_MSG_CHECKING(runtime Load balancing support)
AC_MSG_RESULT($enable_runtime_load)
if test "$enable_runtime_load" = "yes"; then
   AC_DEFINE(CSP_ENABLE_RUNTIME_LOAD_OPT,1,[Define if enable runtime load balancing])
fi

# Dynamic adaptation
AC_ARG_ENABLE(dynamic-adapt, AC_HELP_STRING([--enable-dynamic-adapt],[Enable Dynamic Adaptation]),
                 [ enable_dynamic_adapt=$enableval ],
                 [ enable_dynamic_adapt=no ])
AC_MSG_CHECKING(dynamic adaptation support)
AC_MSG_RESULT($enable_dynamic_adapt)
if test "$enable_dynamic_adapt" = "yes"; then
   AC_DEFINE(CSP_ENABLE_RUNTIME_ASYNC_SCHED,1,[Define if enable dynamic adaptation])
   AC_DEFINE(CSP_ENABLE_RUNTIME_MONITOR,1,[Define if enable runtime monitoring])
   # pass to test configure.
   export CSP_ENABLE_RUNTIME_ASYNC_SCHED=1
fi

# Epoch statuc check
AC_ARG_ENABLE(epochstat-check, AC_HELP_STRING([--disable-epochstat-check],[Disable epoch status check for better performance]),
                 [ enable_epochstat_check=$enableval ],
                 [ enable_epochstat_check=yes ])
AC_MSG_CHECKING(epoch status checking support)
AC_MSG_RESULT($enable_epochstat_check)
if test "$enable_epochstat_check" = "yes"; then
   AC_DEFINE(CSP_ENABLE_EPOCH_STAT_CHECK,1,[Define if disable epoch status check])
fi

# Fast implementation
AC_ARG_ENABLE(fast, AC_HELP_STRING([--enable-fast],
                 [Enable fast implementation. This disables some verbose code used for 
                 info output or strict error checking.]),
                 [ enable_fast=$enableval ],
                 [ enable_fast=no ])
AC_MSG_CHECKING(fast implementation support)
AC_MSG_RESULT($enable_fast)
if test "$enable_fast" = "yes"; then
   AC_DEFINE(CSP_ENABLE_FAST,1,[Define if enable the fast implementation])
fi

# External MPL
AC_ARG_WITH(mpl, [AC_HELP_STRING([--with-mpl[=DIR]],
                [Use the selected MPL installation; Header file mpl_timer.h should be in 
                dir/include, and libmpl should be in dir/lib. Also see option
                --with-timer])],,with_mpl=no)
# Timer method
AC_ARG_WITH([timer],
[  --with-timer=mpl|mpi
                Select the timer to use for internal timestamps.
                mpi - MPI_Wtime (default)
                mpl - MPL timer, user may select faster timer implementation at 
                      MPL installing time. (see option --with-mpl).
],,with_timer=mpi)
AC_MSG_CHECKING(Timer type used for internal timestamps)
AC_MSG_RESULT($with_timer)

case $with_timer in
    mpl)
        if test "$with_mpl" = "no" ; then
            AC_MSG_ERROR([unable to find mpl library.])
        fi

        CFLAGS="$CFLAGS -I$with_mpl/include"
        LDFLAGS="$LDFLAGS -L$with_mpl/lib"
        LIBS="$LIBS -lmpl"
        AC_CHECK_HEADERS(mpl.h)
        AC_CHECK_LIB([mpl],[MPL_wtime_init],have_mpl=yes,AC_MSG_ERROR([unable to find mpl library.]))

        AC_DEFINE([CSP_MPL_TIMER],[1],[defined if the MPL timer is enabled])
    ;;
    *)
        # do nothing
    ;;
esac

# check for attribute support
PAC_C_GNU_ATTRIBUTE

# check for compiler support for the __typeof() extension
AC_CACHE_CHECK([whether the compiler supports __typeof(variable)],
               [pac_cv_have___typeof],
[AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],[[double foo = 0.0; __typeof(foo) bar = 1.0;]])],
                  [pac_cv_have___typeof=yes],
                  [pac_cv_have___typeof=no])]
)
if test "$pac_cv_have___typeof" = "yes" ; then
    AC_DEFINE([HAVE___TYPEOF],[1],[defined if the C compiler supports __typeof(variable)])
fi

#AC_SUBST(abs_top_builddir)

AC_CONFIG_SUBDIRS([test])
AC_CONFIG_FILES([preprocess/Makefile])
AC_CONFIG_FILES([preprocess/gen_batch])
AC_OUTPUT_COMMANDS([chmod a+x preprocess/gen_batch])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
