#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ctype.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <omp.h>

#include "tensor.h"

/**
 * @file: tensor.c
 * @author: ALESSANDRO QUERENGHI
 *
 * This files contains the implementation of the header functions specified in
 * tensor.h
 */

tensor_t *build_empty_tensor(int rows, int columns)
{
	float *data = (float *)malloc((rows * columns) * sizeof(float));
	if (NULL == data) {
		return NULL;
	}
	storage_t *s = (storage_t *)malloc(sizeof(storage_t));
	if (NULL == s) {
		perror("build_empty_tensor: failed to allocate memory");
		free(data);
		return NULL;
	}
	tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
	if (NULL == t) {
		perror("build_empty_tensor: failed to create tensor");
		free(s);
		free(data);
		return NULL;
	}
	s->data = data;
	s->on_disk = false;
	s->ref_counter = 1;
	t->store = s;
	t->shape[0] = rows;
	t->shape[1] = columns;
	return t;
}

tensor_t *build_zero_tensor(int rows, int columns)
{
	// allocates memory and fills it with 0s
	float *data = (float *)calloc(rows * columns, sizeof(float));
	if (NULL == data) {
		return NULL;
	}
	storage_t *s = (storage_t *)malloc(sizeof(storage_t));
	if (NULL == s) {
		perror("build_zero_tensor: failed to allocate memory");
		free(data);
		return NULL;
	}
	tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
	if (NULL == t) {
		perror("build_zero_tensor: failed to create tensor");
		free(s);
		free(data);
		return NULL;
	}
	s->data = data;
	s->on_disk = false;
	s->ref_counter = 1;
	t->store = s;
	t->shape[0] = rows;
	t->shape[1] = columns;
	return t;
}

tensor_t *build_on_disk_tensor(const char *filename)
{
	// create storage and tensor structs
	storage_t *s = (storage_t *)malloc(sizeof(storage_t));
	if (NULL == s) {
		perror("build_on_disk_tensor: tensor creation failed");
		return NULL;
	}
	tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
	if (NULL == t) {
		perror("build_on_disk_tensor: tensor creation failed");
		free(s);
		return NULL;
	}
	// open file
	FILE *fd = fopen(filename, "r");
	if (NULL == fd) {
		perror("build_on_disk_tensor: failed to open file");
		destroy_tensor(t);
		return NULL;
	}
	// read file info
	struct stat sbuf;
	if (0 != fstat(fileno(fd), &sbuf)) {
		perror("build_on_disk_tensor: failed to read file info");
		destroy_tensor(t);
		fclose(fd);
		return NULL;
	}
	// map memory to file
	void *map = mmap((void *)0, sbuf.st_size, PROT_READ, MAP_SHARED, fileno(fd), 0);
	if (MAP_FAILED == map) {
		perror("build_on_disk_tensor: failed to mmap data");
		destroy_tensor(t);
		fclose(fd);
		return NULL;
	}
	// cast mmap-ped memory to on_disk_tensor to read automatically the parameters
	struct on_disk_tensor *header = (struct on_disk_tensor*)map;
	// initialize tensor fields
	t->shape[0] = header->shape[0];
	t->shape[1] = header->shape[1];
	t->store = s;
	s->on_disk = true;
	s->ref_counter = 1;
	// point data to file region mapped with mmap, add offset to skip header (so
	// it isn't necessary later) and cast to float
	s->data = (float *)((char *)map + header->offset);
	s->offset = header->offset;
	// save for munmap
	s->mmap_size = sbuf.st_size;
	fclose(fd);
	return t;
}

tensor_t *build_from_netpbm(const char *filename)
{
	FILE *fd = fopen(filename, "rb");
	if (NULL == fd) {
		perror("build_from_netpbm: failed to open file");
		return NULL;
	}
	// read header and make sure this is actually a pgm file
	int n, m;
	if (2 != fscanf(fd, "P5\n%d %d\n255\n", &m, &n)) {
		fprintf(stderr, "build_from_netpbm: invalid file format\n");
		fclose(fd);
		return NULL;
	}
	tensor_t *t = build_empty_tensor(n, m);
	if (NULL == t) {
		perror("build_from_netpbm: failed to create tensor");
		fclose(fd);
		return NULL;
	}
	int size = n * m;
	uint8_t *data_raw = (uint8_t *)malloc(size);
	if (NULL == data_raw) {
		perror("build_from_netpbm: failed to allocate memory");
		destroy_tensor(t);
		fclose(fd);
	}
	if ((size_t)size != fread(data_raw, 1, size, fd)) {
		fprintf(stderr, "build_from_netpbm: failed to read pixel data\n");
		free(data_raw);
		destroy_tensor(t);
		fclose(fd);
		return NULL;
	}
#pragma omp parallel for default(none) shared(t, data_raw, size) schedule(static)
	// fills data array with pixel data, converts bytes to float by casting and
	// dividing by 255
	for (int i = 0; i < size; i++) {
		t->store->data[i] = (float)data_raw[i] / 255.0f;
	}
	free(data_raw);
	fclose(fd);
	return t;
}

void destroy_tensor(tensor_t *t)
{
	// nothing to do
	if (NULL == t) {
		return;
	}
	storage_t *s = t->store;
	// if there are still copies of this tensor just decrement the ref_counter
	if (1 < s->ref_counter) {
		s->ref_counter--;
	} else {
		if (!s->on_disk) {
			// tensor is in-memory
			free(s->data);
		} else {
			// tensor is on disk
			void *map = (char *)s->data - s->offset;
			munmap(map, s->mmap_size);
		}
		free(s);
	}
	free(t);
}
