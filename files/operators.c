#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "stack.h"
#include "tensor.h"
#include "operators.h"
#include "matrix_multiply.c"

/**
 * @file:		operators.c
 * @author: ALESSANDRO QUERENGHI
 * @id:			IN2300001
 *
 * This files implements the operations defined in operators.h
 */

/**
 * Utility function to make sure that the stack contains at least count
 * elements and that each is a tensor
 *
 * @param[in,out] s The pointer to the stack
 * @param[in] count The number of tensors that are required
 * @param[in] func_name The name of the function that called require_tensors
 * @return 0 if the stack contains count tensors or a negative integer
 * othwerise
 */
int require_tensors(tf_stack_t *s, int count, char *func_name)
{
	if (count > s->count) {
		fprintf(stderr, "%s: requires %d elements at least\n", func_name, count);
		return -1;
	}
	for (int i = 0; i < count; i++) {
		if (TYPE_TENSOR != s->items[s->count - i - 1].type) {
			fprintf(stderr, "%s: %d tensors required\n", func_name, count);
			return -2;
		}
	}
	return 0;
}

/**
 * Utility function to make sure that the shape tensor has the structure
 * defined by TensorForth
 *
 * @param[in,out] shape The pointer to the shape tensor
 * @param[in] func_name The name of the function that called verify_shape_tensor
 * @return 0 if the shape tensor has the correct structure or a negative
 * integer otherwise
 */
int verify_shape_tensor(tensor_t *shape, char *func_name)
{
	if (1 != shape->shape[0] || 2 < shape->shape[1]) {
		fprintf(stderr, "%s: shape tensors have shape either [ 1 1 ] or [ 1 2 ], this tensor: [%d %d]\n", func_name, shape->shape[0], shape->shape[1]);
		return -1;
	}
	int size = shape->shape[0] * shape->shape[1];
	for (int i = 0; i < size; i++) {
		if (0 > shape->store->data[i]) {
			fprintf(stderr, "%s: shape data should be positive\n", func_name);
			return -2;
		}
	}
	if (1 > size) {
		fprintf(stderr, "%s: each dimension should be at least >= 1\n", func_name);
		return -3;
	}
	return 0;
}

operation_t get_operation_from_char(char c)
{
	switch (c) {
		// UTILITY
		case 'p': return OP_PRINT;
		// MANIPULATION
		case 'd': return OP_DUPLICATE;
		case 's': return OP_SWAP;
		case 'o': return OP_OVER;
		case 'D': return OP_DROP;
		// GENERATION
		case 'f': return OP_FILL;
		case '?': return OP_RAND;
		// SHAPING
		case 'r': return OP_RESHAPE;
		case '_': return OP_RAVEL;
		case '#': return OP_SHAPE;
		// ELEMENT BY ELEMENT
		case '+': return OP_ADD;
		case '-': return OP_SUB;
		case '*': return OP_MUL;

		case '<': return OP_LT;
		case '>': return OP_GT;
		case '=': return OP_EQ;

		case '&': return OP_AND;
		case '|': return OP_OR;
		case '!': return OP_NOT;

		case 'R': return OP_RELU;
		case 'm': return OP_MIN;
		case 'M': return OP_MAX;
		// REDUCTION
		case 'S': return OP_SUM;
		// TENSOR OPS
		case '@': return OP_MATMUL;
		case '.': return OP_DOT;
		case 'c': return OP_CONV;
		// FILES
		case '(': return LOAD_PGM;
		case '{': return LOAD_TENSOR;
		case ')': return WRITE_PGM;
		case '}': return WRITE_TENSOR;

		default: return OP_UNKNOWN;
	}
}

/**
 * Print the tensor on top of the stack and consume it
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int print_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "print_tensor")) {
		return -1;
	}
	// consume element
	s->count--;
	tensor_t *t = s->items[s->count].as.t;
	printf("Tensor(shape=[%d %d], data=[", t->shape[0], t->shape[1]);
	int stored = (t->shape[0] * t->shape[1]);
	if (stored) {
		printf("%.3f", t->store->data[0]);
		for (int i = 1; i < stored; i++) {
			printf(" %.3f", t->store->data[i]);
		}
	}
	printf("])\n");
	destroy_tensor(t);
	return 0;
}

/**
 * Duplicate the tensor on top of the stack
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int duplicate_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "duplicate_tensor")) {
		return -1;
	}
	// The tensor to be duplicated
	tensor_t *t = s->items[s->count - 1].as.t;
	tensor_t *duplicate = (tensor_t *)malloc(sizeof(tensor_t));
	if (NULL == duplicate) {
		perror("duplicate_tensor: memory allocation error");
		return -2;
	}
	// Copy the values
	duplicate->shape[0] = t->shape[0];
	duplicate->shape[1] = t->shape[1];
	// Point to the same storage_t element
	duplicate->store = t->store;
	// try to push it on top of the stack
	if (0 != push_tensor(s, duplicate)) {
		free(duplicate);
		return -3;
	}
	// increase the ref_counter so that the program has a copy of the tensor
	// without allocating more memory
	duplicate->store->ref_counter++;
	return 0;
}

/**
 * Swap the two tensors on top of the stack
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int swap_tensors(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "swap_tensors")) {
		return -1;
	}
	// temporary copy
	stack_item_t temp = s->items[s->count - 1];
	s->items[s->count - 1] = s->items[s->count - 2];
	s->items[s->count - 2] = temp;
	return 0;
}

/**
 * Duplicate the second tensor from the top of the stack and put it on top
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int overput_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "overput_tensor")) {
		return -1;
	}
	tensor_t *t = s->items[s->count - 2].as.t;
	tensor_t *duplicate = (tensor_t *)malloc(sizeof(tensor_t));
	if (NULL == duplicate) {
		perror("overput_tensor: memory allocation error");
		return -2;
	}
	duplicate->shape[0] = t->shape[0];
	duplicate->shape[1] = t->shape[1];
	duplicate->store = t->store;
	if (0 != push_tensor(s, duplicate)) {
		free(duplicate);
		return -3;
	}
	duplicate->store->ref_counter++;
	return 0;
}

/**
 * Pop the tensor from the top of the stack and destroy it
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int drop_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "drop_tensor")) {
		return -1;
	}
	destroy_tensor(s->items[s->count - 1].as.t);
	s->count--;
	return 0;
}

/**
 * Pop f and s from the top of the stack and push on top a new tensor with
 * shape s and fill f (eventually repeated)
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int fill_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "fill_tensor")) {
		return -1;
	}
	tensor_t *fill = s->items[s->count - 1].as.t;
	tensor_t *shape = s->items[s->count - 2].as.t;
	int data_size = (int)(shape->store->data[0] * (2 == shape->shape[1] ? shape->store->data[1] : 1));
	int fill_size = fill->shape[0] * fill->shape[1];
	if (0 != verify_shape_tensor(shape, "fill_tensor")) {
		return -2;
	}
	if (1 > fill_size) {
		fprintf(stderr, "fill_tensor: fill tensor should contain something\n");
		return -3;
	}
	// make space for new tensor
	float *data = (float *)malloc(data_size * sizeof(float));
	if (NULL == data) {
		perror("fill_tensor: memory allocation error");
		return -4;
	}
	for (int i = 0; i < data_size; i++) {
		// fill with values, they can repeat
		data[i] = fill->store->data[i % fill_size];
	}
	// since shape tensors are either [ 1 1 ] or [ 1 2 ]
	int n = 1 == shape->shape[1] ? 1 : (int)shape->store->data[0];
	int m = (int)shape->store->data[1 == shape->shape[1] ? 0 : 1];
	tensor_t *t = build_empty_tensor(n, m);
	if (NULL == t) {
		free(data);
		return -5;
	}
	t->store->data = data;
	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -6;
	}
	return 0;
}

/**
 * Pop a shape tensor from the stack and push a new tensor with shape s and content random values in the range [0, 1]
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int fill_random(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "fill_random")) {
		return -1;
	}
	tensor_t *shape = s->items[s->count - 1].as.t;
	if (0 != verify_shape_tensor(shape, "fill_random")) {
		return -2;
	}
	// since shape tensors are either [ 1 1 ] or [ 1 2 ]
	int n = 1 == shape->shape[1] ? 1 : (int)shape->store->data[0];
	int m = (int)shape->store->data[1 == shape->shape[1] ? 0 : 1];
	int data_size = n * m;
	float *data = (float *)malloc(data_size * sizeof(float));
	if (NULL == data) {
		perror("fill_random: memory allocation error");
		return -3;
	}
	for (int i = 0; i < data_size; i++) {
		// maybe use rand_r or something else for parallel filling
		data[i] = (float)rand() / RAND_MAX;
	}
	tensor_t *t = build_empty_tensor(n, m);
	if (NULL == t) {
		free(data);
		return -4;
	}
	t->store->data = data;
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -5;
	}
	return 0;
}

/**
 * Pop a shape tensor s and reshape the tensor on top of the stack
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int reshape_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "reshape_tensor")) {
		return -1;
	}
	tensor_t *shape = s->items[s->count - 1].as.t;
	tensor_t *target = s->items[s->count - 2].as.t;
	if (0 != verify_shape_tensor(shape, "reshape_tensor")) {
		return -2;
	}
	// since shape tensors are either [ 1 1 ] or [ 1 2 ]
	int n = 1 == shape->shape[1] ? 1 : (int)shape->store->data[0];
	int m = (int)shape->store->data[1 == shape->shape[1] ? 0 : 1];
	int new_size = n * m;
	int old_size = target->shape[0] * target->shape[1];
	// check that sizes match
	if (old_size != new_size) {
		fprintf(stderr, "reshape_tensor: new size doesn't match old size\n");
		return -3;
	}
	// push new parameters into reshaped tensor
	target->shape[0] = n;
	target->shape[1] = m;
	drop_tensor(s);
	return 0;
}

/**
 * Flatten the tensor on top of the stack
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int ravel_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "ravel_tensor")) {
		return -1;
	}
	tensor_t *target = s->items[s->count - 1].as.t;
	// tensor has shape [ 1 n * m ]
	int size = target->shape[0] * target->shape[1];
	// push values into reshaped tensor
	target->shape[0] = 1;
	target->shape[1] = size;
	return 0;
}

/**
 * Pop the tensor on top and push a tensor with its shape
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int shape_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "shape_tensor")) {
		return -1;
	}
	tensor_t *target = s->items[s->count - 1].as.t;
	// create shape tensor
	tensor_t *t = build_empty_tensor(1, 2);
	if (NULL == t) {
		return -2;
	}
	// fill with shape of target
	t->store->data[0] = (float)target->shape[0];
	t->store->data[1] = (float)target->shape[1];
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -4;
	}
	return 0;
}
		
/**
 * Pop two tensors from the stack and perform an operation specified by func on
 * each element of them. func has signature void (*func)(float *, float *,
 * float *)
 *
 * @param[in,out] s The pointer to the stack
 * @param[in] op_name The name of the function performing the operation
 * @param[in] func The function thath performs the operation
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int op_elem_by_elem(tf_stack_t *s, char *op_name, math_op func)
{
	if (0 != require_tensors(s, 2, op_name)) {
		return -1;
	}
	// the operands
	tensor_t *left = s->items[s->count - 1].as.t;
	tensor_t *right = s->items[s->count - 2].as.t;
	if (right->shape[0] != left->shape[0] ||
			right->shape[1] != left->shape[1]) {
		fprintf(stderr, "%s: incompatible shapes [%d, %d] != [%d, %d]\n", op_name, left->shape[0], left->shape[1], right->shape[0], right->shape[1]);
		return -2;
	}
	// result tensor
	tensor_t *t = build_empty_tensor(left->shape[0], left->shape[1]);
	if (NULL == t) {
		return -3;
	}
	int size = left->shape[0] * left->shape[1];

	// pass only data arrays and size
	func(t->store->data, left->store->data, right->store->data, size);

	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -4;
	}
	return 0;
}

/**
 * Sums the values from left with right and puts the result in res
 *
 * @param[out] res The array containing the result
 * @param[in] left First operand
 * @param[in] right Second operand
 */
void add_floats(float *res, const float *left, const float *right, int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = left[i] + right[i];
	}
}

void sub_floats(float *res, const float *left, const float *right, int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = left[i] - right[i];
	}
}

void mult_floats(float *res, const float *left, const float *right, int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = left[i] * right[i];
	}
}

void compare_lt(float *res, const float *left, const float *right, int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = left[i] < right[i];
	}
}

void compare_gt(float *res, const float *left, const float *right, int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = left[i] > right[i];
	}
}

void compare_eq(float *res, const float *left, const float *right, int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = left[i] == right[i];
	}
}

/**
 * Pop the mask, left and right operand from the stack and allocate a new
 * tensor filled with elements from left if the mask contains 0.0 and from
 * right if it contains 1.0
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int op_ternary(tf_stack_t *s)
{
	if (0 != require_tensors(s, 3, "op_ternary")) {
		return -1;
	}
	tensor_t *mask = s->items[s->count - 1].as.t;
	tensor_t *left = s->items[s->count - 2].as.t;
	tensor_t *right = s->items[s->count - 3].as.t;
	// check that sizes match
	if (right->shape[0] != left->shape[0] || right->shape[0] != mask->shape[0] ||
			right->shape[1] != left->shape[1] || right->shape[1] != mask->shape[1]) {
		fprintf(stderr, "op_ternary: incompatible shapes\n");
		return -2;
	}
	int size = left->shape[0] * left->shape[1];
	// allocate space for new data
	float *data = (float *)malloc(size * sizeof(float));
	if (NULL == data) {
		perror("op_ternary: memory allocation error");
		return -3;
	}
	for (int i = 0; i < size; i++) {
		int p = mask->store->data[i] == 1;
		// insert element from right if mask[i] == 1.0, else from left
		data[i] = p * right->store->data[i] + (1 - p) * left->store->data[i];
	}
	tensor_t *t = build_empty_tensor(left->shape[0], left->shape[1]);
	if (NULL == t) {
		free(data);
		return -4;
	}
	t->store->data = data;
	drop_tensor(s);
	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -5;
	}
	return 0;
}

void compare_and(float *res, const float *left, const float *right, int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = (float)(((int)left[i]) & ((int)right[i]));
	}
}

void compare_or(float *res, const float *left, const float *right, int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = (float)(((int)left[i]) | ((int)right[i]));
	}
}

/**
 * Pop the binary mask on top of the stack and replace each element its bit
 * negation. If the content of the mask is not binary the outcome is not
 * specified.
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int negate_floats(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "negate_floats")) {
		return -1;
	}
	tensor_t *t = s->items[s->count - 1].as.t;
	tensor_t *target = build_empty_tensor(t->shape[0], t->shape[1]);
	if (NULL == target) {
		return -2;
	}
	int size = t->shape[0] * t->shape[1];
	for (int i = 0; i < size; i++) {
		// put the negation of each value in the result vector
		target->store->data[i] = (float)(!t->store->data[i]);
	}
	drop_tensor(s);
	if (0 != push_tensor(s, target)) {
		destroy_tensor(target);
		return -3;
	}
	return 0;
}

/**
 * Normalize the tensor on top of the stack by zeroing negative values
 * 
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
// see if branchless is better than branched
int get_relu(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "get_relu")) {
		return -1;
	}
	tensor_t *t = s->items[s->count - 1].as.t;
	tensor_t *target = build_empty_tensor(t->shape[0], t->shape[1]);
	if (NULL == target) {
		return -2;
	}
	int size = t->shape[0] * t->shape[1];
	for (int i = 0; i < size; i++) {
		int p = (0 < t->store->data[i]);
		target->store->data[i] = p * t->store->data[i];
	}
	drop_tensor(s);
	if (0 != push_tensor(s, target)) {
		destroy_tensor(target);
		return -3;
	}
	return 0;
}

// check this branchless
void get_min(float *res, const float *left, const float *right, int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = (left[i] + right[i] - fabsf(left[i] - right[i])) / 2;
	}
}

// check this branchless
void get_max(float *res, const float *left, const float *right, int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = (left[i] + right[i] + fabsf(left[i] - right[i])) / 2;
	}
}

/**
 * Perform a reduction on the vector on top of the stack, return a single
 * element tensor containing the sum of all elements of the other tensor
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int sum_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "sum_tensor")) {
		return -1;
	}
	tensor_t *t = s->items[s->count - 1].as.t;
	tensor_t *target = build_empty_tensor(1, 1);
	if (NULL == target) {
		return -2;
	}
	float total_sum = 0.0f;
	int size = t->shape[0] * t->shape[1];
	// #pragma omp parallel for reduction(+: total_sum)
	for (int i = 0; i < size; i++) {
		// sum all elements in the tensor
		total_sum += t->store->data[i];
	}
	target->store->data[0] = total_sum;
	drop_tensor(s);
	if (0 != push_tensor(s, target)) {
		destroy_tensor(target);
		return -3;
	}
	return 0;
}

/**
 * Pop two compatible 2D tensors from the stack and perform matrix
 * multiplication, pushes the result on top of the stack
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int matmul(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "matmul")) {
		return -1;
	}
	// (b a -- a@b)
	// a => left
	// b => right
	//
	// a => sopra
	// b => sotto
	tensor_t *left = s->items[s->count - 1].as.t;
	tensor_t *right = s->items[s->count - 2].as.t;
	if (1 == left->shape[0] || 1 == right->shape[0]) {
		fprintf(stderr, "matmul: matmul requires 2D elements for 1D use the dot operation '.' instead\n");
		return -2;
	}
	if (left->shape[1] != right->shape[0]) {
		fprintf(stderr, "matmul: incompatible sizes: [%d, %d] @ [%d, %d]",
						left->shape[0], left->shape[1],
						right->shape[0], right->shape[1]);
		return -3;
	}
	int n = left->shape[0];
	int p = left->shape[1];
	int m = right->shape[1];
	tensor_t *target = build_zero_tensor(n, m);
	if (NULL == target) {
		return -4;
	}
	// imported from matrix_multiply.h
	blocked_multiply(left->store->data, right->store->data, target->store->data, n, m, p);
	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, target)) {
		destroy_tensor(target);
		return -5;
	}
	return 0;
}

/**
 * Scalar vector of two 1D tensors
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int dot(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "dot")) {
		return -1;
	}
	tensor_t *left = s->items[s->count - 1].as.t;
	tensor_t *right = s->items[s->count - 2].as.t;
	if (1 != left->shape[0] || 1 != right->shape[0]) {
		fprintf(stderr, "dot: dot product only allowed with 1D vectors\n");
		return -2;
	}
	if (left->shape[1] != right->shape[1]) {
		fprintf(stderr, "dot: incompatible sizes: [1, %d] != [1, %d]\n", left->shape[1], right->shape[1]);
		return -3;
	}
	float dot_sum = 0.0f;
	int size = left->shape[1];
	for (int i = 0; i < size; i++) {
		// perform scalar product
		dot_sum += left->store->data[i] * right->store->data[i];
	}
	tensor_t *target = build_empty_tensor(1, 1);
	if (NULL == target) {
		return -4;
	}
	target->store->data[0] = dot_sum;
	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, target)) {
		destroy_tensor(target);
		return -5;
	}
	return 0;
}

/**
 * Pop a convolution kernel and a tensor from the stack and perform 0-padded
 * convolution with stride 1
 *
 * @param[in,out] s The pointer to the stack
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int convolute_tensors(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "convolute_tensors")) {
		return -1;
	}
	tensor_t *kernel = s->items[s->count - 1].as.t;
	tensor_t *left = s->items[s->count - 2].as.t;
	if (1 == left->shape[0]) {
		fprintf(stderr, "convolute_tensors: can only convolute 2D matrices\n");
		return -2;
	}
	if (kernel->shape[0] != kernel->shape[1] || 0 == (kernel->shape[0] % 2)) {
		fprintf(stderr, "convolute_tensors: kernels are square-shaped and odd side length\n");
		return -3;
	}
	int n = left->shape[0];
	int m = left->shape[1];
	int k = kernel->shape[0];
	// the result of the convolution shares the dimensions with the original vector
	tensor_t *target = build_empty_tensor(n, m);
	if (NULL == target) {
		return -3;
	}
	// number of padding layers
	int offset = (k - 1) / 2;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			// simulate padding: p for rows and q for columns, start avoids memory
			// overflow when close to the top and left sides, while stop avoids
			// memory overflow on the right and bottom sides
			int start_p = offset > i ? offset - i : 0;
			int start_q = offset > j ? offset - j : 0;
			int stop_p = offset > n - i - 1 ? n - i + offset : k;
			int stop_q = offset > m - j - 1 ? m - j + offset : k;
			float sum = 0;
			for (int p = start_p; p < stop_p; p++) {
				for (int q = start_q; q < stop_q; q++) {
					sum += left->store->data[(i + p - offset) * m + j + q - offset] * kernel->store->data[p * k + q];
				}
			}
			target->store->data[i * m + j] = sum;
		}
	}
	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, target)) {
		destroy_tensor(target);
		return -4;
	}
	return 0;
}

/**
 * Helper function to load a tensor from a file
 *
 * @param[in,out] s The pointer to the stack
 * @param[in] func_name The name of the function bein called
 * @param[in] func The function that loads a tensor from disk
 */
int load_from_file(tf_stack_t *s, const char *func_name, tensor_t * (*func)(const char *))
{
	if (1 > s->count) {
		fprintf(stderr, "load_pgm: the stack doesn't contain enough arguments\n");
		return -1;
	}
	if (TYPE_STRING != s->items[s->count - 1].type) {
		fprintf(stderr, "load_pgm: filename required\n");
		return -2;
	}
	char *filename = s->items[s->count - 1].as.filename;
	// load the tensor
	tensor_t *t = func(filename);
	if (NULL == t) {
		return -3;
	}
	// pop the file name from the stack
	s->count--;
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -4;
	}
	return 0;
}

/**
 * Helper function to save a tensor to file
 *
 * @param[in,out] s The pointer to the stack
 * @param[in] func_name The name of the function bein called
 * @param[in] func The function that writes a tensor to disk
 */
int save_to_file(tf_stack_t *s, const char * func_name, int (*func)(FILE *, tensor_t *))
{
	if (2 > s->count) {
		fprintf(stderr, "%s: stack requires at least 2 elements\n", func_name);
		return -1;
	}
	if (TYPE_STRING != s->items[s->count - 1].type ||
			TYPE_TENSOR != s->items[s->count - 2].type) {
		fprintf(stderr, "%s: stack requires one tensor and one filename\n", func_name);
		return -2;
	}
	const char *filename = s->items[s->count - 1].as.filename;
	tensor_t *t = s->items[s->count - 2].as.t;
	FILE *fd = fopen(filename, "wb");
	if (NULL == fd) {
		perror("save_to_file: failed to open file");
		return -3;
	}
	if (0 != func(fd, t)) {
		fprintf(stderr, "%s: failed to write to file\n", func_name);
		fclose(fd);
		return -4;
	}
	fclose(fd);
	s->count--;
	drop_tensor(s);
	return 0;
}

/**
 * Function to write a tensor to pgm format, each element in data is normalized
 * as a byte value between 0 and 255
 *
 * @param[in,out] fd File stream on which the tensor is written
 * @param[in] t Tensor to be written
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int write_pgm(FILE *fd, tensor_t *t)
{
	// write header to file
	if (0 > fprintf(fd, "P5\n%d %d\n255\n", t->shape[1], t->shape[0])) {
		perror("write_pgm: failed to write header");
		return -1;
	}
	int size = t->shape[0] * t->shape[1];
	uint8_t *data = (uint8_t *)malloc(size);
	if (NULL == data) {
		perror("write_pgm: memory allocation failed");
		return -2;
	}
	for (int i = 0; i < size; i ++) {
		// normalize data and copy it to data array
		data[i] = (uint8_t)(((0 < t->store->data[i]) * t->store->data[i] - ((1 < t->store->data[i]) * (t->store->data[i] - 1))) * 255);
	}
	// write data and make sure that everything is written to file
	if (size != fwrite(data, 1, size, fd)) {
		fprintf(stderr, "write_pgm: failed to write data\n");
		free(data);
		return -2;
	}
	free(data);
	return 0;
}

/**
 * Function to write a raw tensor to file
 *
 * @param[in,out] fd File stream on which the tensor is written
 * @param[in] t Tensor to be written
 * @return 0 if the operation is successful or a negative integer otherwise
 */
int write_tensor(FILE *fd, tensor_t *t)
{
	// create the header of the file
	struct on_disk_tensor header;
	// zeroing the memory
	memset(&header, 0, sizeof(header));
	header.shape[0] = t->shape[0];
	header.shape[1] = t->shape[1];
	header.dim = 1 == header.shape[0] ? 1 : 2;
	// since the shape and dim are 32 bits each the offset only requires to be 64 bytes
	header.offset = 64;
	size_t header_size = sizeof(struct on_disk_tensor);
	// write header to file
	if (1 != fwrite(&header, header_size, 1, fd)) {
		fprintf(stderr, "write_tensor: failed to write %ld bytes to disk\n", header_size);
		return -1;
	}
	// compute the offset so that data is page aligned
	if (header.offset > header_size) {
		size_t padding = header.offset - header_size;
		char pad[64] = {0};
		// write padding
		if (padding != fwrite(pad, 1, padding, fd)) {
			fprintf(stderr, "write_tensor: failed to write padding\n");
			return -2;
		}
	}
	size_t size = (size_t)(t->shape[0] * t->shape[1]);
	if (size != fwrite(t->store->data, sizeof(float), size, fd)) {
		fprintf(stderr, "write_tensor: failed to write data\n");
		return -1;
	}
	return 0;
}

int execute_operation(tf_stack_t *s, operation_t op)
{
	switch (op) {
		// UTILITY
		case OP_PRINT:			return print_tensor(s);
		// MANIPULATION
		case OP_DUPLICATE:	return duplicate_tensor(s);
		case OP_SWAP:				return swap_tensors(s);
		case OP_OVER:				return overput_tensor(s);
		case OP_DROP:				return drop_tensor(s);
		// GENERATION
		case OP_FILL:				return fill_tensor(s);
		case OP_RAND:				return fill_random(s);
		// SHAPING
		case OP_RESHAPE:		return reshape_tensor(s);
		case OP_RAVEL:			return ravel_tensor(s);
		case OP_SHAPE:			return shape_tensor(s);
		// ELEMENT BY ELEMENT
		case OP_ADD:				return op_elem_by_elem(s, "add_floats", add_floats);
		case OP_SUB:				return op_elem_by_elem(s, "sub_floats", sub_floats);
		case OP_MUL:				return op_elem_by_elem(s, "mult_floats", mult_floats);

		case OP_LT:					return op_elem_by_elem(s, "compare_lt", compare_lt);
		case OP_GT:					return op_elem_by_elem(s, "compare_gt", compare_gt);
		case OP_EQ:					return op_elem_by_elem(s, "compare_eq", compare_eq);

		case OP_TERNARY:		return op_ternary(s);

		case OP_AND:				return op_elem_by_elem(s, "compare_and", compare_and);
		case OP_OR:					return op_elem_by_elem(s, "compare_or", compare_or);
		case OP_NOT:				return negate_floats(s);

		case OP_RELU:				return get_relu(s);
		case OP_MIN:				return op_elem_by_elem(s, "get_min", get_min);
		case OP_MAX:				return op_elem_by_elem(s, "get_max", get_max);
		// REDUCTION
		case OP_SUM:				return sum_tensor(s);
		// TENSOR OPS
		case OP_MATMUL:			return matmul(s);
		case OP_DOT:				return dot(s);
		case OP_CONV:				return convolute_tensors(s);
		// FILES
		case LOAD_PGM:			return load_from_file(s,	"load_pgm", build_from_netpbm);
		case LOAD_TENSOR:		return load_from_file(s,	"load_tensor", build_on_disk_tensor);
		case WRITE_PGM:			return save_to_file(s,		"write_pgm", write_pgm);
		case WRITE_TENSOR:	return save_to_file(s,		"write_tensor", write_tensor);

		case OP_UNKNOWN:
			fprintf(stderr, "execute_operation: unknown command\n");
			return -1;
	}
	return 0;
}

