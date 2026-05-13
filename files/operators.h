#ifndef _OPERATORS_H
#define _OPERATORS_H

#include "stack.h"

typedef enum {
	// UTILITY
	OP_PRINT,
	// MANIPULATION
	OP_DUPLICATE,
	OP_SWAP,
	OP_OVER,
	OP_DROP,
	// GENERATION
	OP_FILL,
	OP_RAND,
	// SHAPING
	OP_RESHAPE,
	OP_RAVEL,
	OP_SHAPE,
	// ELEMENT BY ELEMENT
	OP_ADD,
	OP_SUB,
	OP_MUL,

	OP_LT,
	OP_GT,
	OP_EQ,

	OP_AND,
	OP_OR,
	OP_NOT,

	OP_RELU,
	OP_MIN,
	OP_MAX,
	// REDUCTION
	OP_SUM,
	// TENSOR OPS
	OP_MATMUL,
	OP_DOT,
	OP_CONV,
	// FILES
	LOAD_PGM,
	LOAD_TENSOR,
	WRITE_PGM,
	WRITE_TENSOR,

	OP_UNKNOWN
} operation_t;

typedef void (*math_op)(float *res, const float *left, const float *right, int size);

operation_t get_operation_from_char(char c);

int execute_operation(tf_stack_t *s, operation_t op);

#endif
