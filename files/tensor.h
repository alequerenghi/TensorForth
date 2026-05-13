#ifndef _TENSOR_H
#define _TENSOR_H

#include <sys/types.h>
#include <stdbool.h>
#include <stdio.h>

#define MAX_DIM 2

typedef struct storage {
	int ref_counter;
	off_t 	offset;
	size_t mmap_size;
	bool on_disk;
	float *data;
} storage_t;

typedef struct tensor {
	int shape[MAX_DIM];
	storage_t *store;
} tensor_t;

struct on_disk_tensor {
	int32_t shape[MAX_DIM];
	int32_t dim;
	off_t offset;
};

tensor_t *build_tensor_from_memory(float *data, int l);

tensor_t *build_empty_tensor(int rows, int columns);

tensor_t *build_zero_tensor(int rows, int columns);

tensor_t *build_from_netpbm(const char *filename);

tensor_t *build_on_disk_tensor(const char *filename);

void destroy_tensor(tensor_t *t);

#endif

