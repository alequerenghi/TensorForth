#ifndef _TENSOR_H
#define _TENSOR_H

#include <sys/types.h>
#include <stdbool.h>
#include <stdio.h>

#define MAX_DIM 2

typedef struct storage {
	int ref_counter;
	off_t 	offset;
	FILE *fd;
	bool on_disk;
	float *data;
} storage_t;

typedef struct tensor {
	int shape[MAX_DIM];
	storage_t *store;
} tensor_t;

tensor_t *build_tensor_from_memory(float *data, int l);

void destroy_tensor(tensor_t *t);

#endif

