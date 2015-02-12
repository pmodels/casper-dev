/*
 * Copyright (C) 2015. See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Retrieve the value of a boolean environment variable. */
int MTU_Getenv_bool(char *varname, int default_value) {
  char *var = getenv(varname);

  if (var == NULL)
    return default_value;
  
  if (var[0] == 'T' || var[0] == 't' || var[0] == '1' || var[0] == 'y' || var[0] == 'Y')
    return 1;

  else
    return 0;
}

/* Retrieve the value of an integer environment variable. */
int MTU_Getenv_int(char *varname, int default_value) {
  char *var = getenv(varname);
  if (val && strlen(val))
    return atoi(var);
  else
    return default_value;
}

/* Retrieve the value of a environment variable. */
char * MTU_Getenv(char *varname) {
  return getenv(varname);
}


