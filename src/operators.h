#ifndef _OPERATORS_H
#define _OPERATORS_H

#include "stack.h"

/**
 * @file:		operators.h
 * @author: ALESSANDRO QUERENGHI
 * @id:			IN2300001
 *
 * This file contains the enumerations that define the operations allowed and
 * the signatures of the functions that are implemented
 */

/**
 * This enumeration represents all the available operations in TensorForth
 */
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

	OP_TERNARY,

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

/**
 * Function pointer for operators that make use of three float elements of the
 * same size
 *
 * @param[in] math_op Pointer to the function to be used
 */
typedef void (*math_op)(float *res, const float *left, const float *right, int size);

/**
 * Function to map a character to the operation enum defined before
 *
 * @param[in] c The character
 * @return The corresponding operation or OP_UNKNOWN if no operation is defined
 * for that character
 */
operation_t get_operation_from_char(char c);

/**
 * Executes the operation on the stack by popping the required elements and
 * pushing the result (if any) onto the stack
 *
 * @param[in,out] s The pointer to the stack
 * @parm[in] op The operation to perform
 * @return 0 if success or a negative number if the operation failed
 */
int execute_operation(tf_stack_t *s, operation_t op);

#endif
