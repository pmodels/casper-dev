#ifndef MTCORE_UTIL_H_
#define MTCORE_UTIL_H_

#include <stdio.h>
#include <stdlib.h>

int MTU_Getenv_bool(char *varname, int default_value);
int MTU_Getenv_int(char *varname, int default_value);
char * MTU_Getenv(char *varname);

#endif MTCORE_UTIL_H_
