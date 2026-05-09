#ifndef _TENSOR_H
#define _TENSOR_H

#include <sys/types.h>
#include <stdbool.h>
#include <stdio.h>

#define MAX_DIM 2

struct tensor {
	int shape[MAX_DIM];
	int ndim;
	int ref_counter;
	off_t 	offset;
	FILE *fd;
	bool on_disk;
	float *data;
};

struct tensor *build_tensor_from_memory(float *data, int l);

int destroy_tensor(struct tensor *t);

#endif

