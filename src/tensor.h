#ifndef _TENSOR_H
#define _TENSOR_H

#include <sys/types.h>
#include <stdbool.h>
#include <stdio.h>

/**
 * @file: tensor.h
 * @author: ALESSANDRO QUERENGHI
 * @id: IN2300001
 *
 * This file contains the structs and signatures to implement a tensor
 */

#define MAX_DIM 2

/**
 * Holds the data of a tensor
 */
typedef struct storage {
	int			ref_counter;
	off_t		offset;
	size_t	mmap_size;
	bool		on_disk;
	float		*data;
} storage_t;

/**
 * Holds a reference to the storage struct, useful when reshaping and
 * duplicating vectors.
 */
typedef struct tensor {
	int				shape[MAX_DIM];
	storage_t	*store;
} tensor_t;

/**
 * Used when writing a tensor to file.
 */
struct on_disk_tensor {
	int32_t shape[MAX_DIM];
	int32_t dim;
	off_t		offset;
};

/**
 * Builds a tensor in memory with the specified dimensions and doesn't
 * initialize the memory
 *
 * @param[in] rows Number of rows
 * @param[in] collumns Number of columns
 * @return The pointer to the tensor or NULL if fails
 */
tensor_t *build_empty_tensor(int rows, int columns);

/**
 * Builds a tensor in memory with the specified dimensions and initializes the
 * data array to zeros
 *
 * @param[in] rows Number of rows
 * @param[in] columns Number of columns
 * @return The pointer to the tensor or NULL if fails
 */
tensor_t *build_zero_tensor(int rows, int columns);

/**
 * Builds a tensor from the specified netpbm image
 *
 * @param[in] filename The file name of pgm image
 * @return The pointer to the tensor or NULL if fails
 */
tensor_t *build_from_netpbm(const char *filename);

/**
 * Builds a tensor from the specified file
 *
 * @param[in] filename The file name of raw tensor on disk
 * @return The pointer to the tensor or NULL if fails
 */
tensor_t *build_on_disk_tensor(const char *filename);

/**
 * Destroys a tensor and frees the memory
 *
 * @param[in] t The tensor to destroy
 */
void destroy_tensor(tensor_t *t);

#endif

