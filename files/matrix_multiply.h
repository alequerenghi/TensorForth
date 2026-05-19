#ifndef _MATRIX_MULTIPLY_H
#define _MATRIX_MULTIPLY_H
						
/**
 * @file:		matrix_multiply.h
 * @author: ALESSANDRO QUERENGHI
 * @id			IN2300001
 *
 * This file contains headers for utility functions regarding matrix
 * multiplication
 */

/**
 * Simple implementation of matrix multiplication using row major order.
 *
 * @param[in] A Left operand
 * @param[in] B Right operand
 * @param[out] C Where the result is stored
 * @param[in] n First dimension of A and C
 * @param[in] m Second dimension of B and C
 * @param[in] p Second dimension of A and first dimension of B
 */
void simple_multiply(float * A, float * B, float * C, int n, int m, int p);

void transposed_multiply(float * A, float * B, float * C, int n, int m, int p);

/**
 * Blocked matrix multiplication to make better use of memory locality
 *
 * @param[in] A Left operand
 * @param[in] B Right operand
 * @param[out] C Where the result is stored
 * @param[in] n First dimension of A and C
 * @param[in] m Second dimension of B and C
 * @param[in] p Second dimension of A and first dimension of B
 */
void blocked_multiply(float * A, float * B, float * C, int n, int m, int p);

#endif
