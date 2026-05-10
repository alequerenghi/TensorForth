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

	OP_UNKNOWN
} operation_t;

operation_t get_operation_from_char(char c);

int execute_operation(tf_stack_t *s, operation_t op);

#endif
