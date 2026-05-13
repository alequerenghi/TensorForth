#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "stack.h"
#include "tensor.h"
#include "operators.h"
#include "matrix_multiply.c"

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

int verify_shape_tensor(tensor_t *shape, char *func_name)
{
	if (1 != shape->shape[0] || 2 != shape->shape[1]) {
		fprintf(stderr, "%s: shape tensors have shape [ 1 2 ]\n", func_name);
		return -1;
	}
	if (1 > (shape->store->data[0] * shape->store->data[1])) {
		fprintf(stderr, "%s: each dimension should be >= 1\n", func_name);
		return -2;
	}
	return 0;
}


operation_t get_operation_from_char(char c)
{
	printf("%c\n", c);
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

int print_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "print_tensor")) {
		return -1;
	}
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

int duplicate_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "duplicate_tensor")) {
		return -1;
	}
	tensor_t *t = s->items[s->count - 1].as.t;
	tensor_t *duplicate = (tensor_t *)malloc(sizeof(tensor_t));
	if (NULL == duplicate) {
		perror("duplicate_tensor: memory allocation error");
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

int swap_tensors(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "swap_tensors")) {
		return -1;
	}
	stack_item_t temp = s->items[s->count - 1];
	s->items[s->count - 1] = s->items[s->count - 2];
	s->items[s->count - 2] = temp;
	return 0;
}

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

int drop_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "drop_tensor")) {
		return -1;
	}
	destroy_tensor(s->items[s->count - 1].as.t);
	s->count--;
	return 0;
}

// forse rifare la creazione del tensore
int fill_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "fill_tensor")) {
		return -1;
	}
	tensor_t *fill = s->items[s->count - 1].as.t;
	tensor_t *shape = s->items[s->count - 2].as.t;
	int data_size = (int)(shape->store->data[0] * shape->store->data[1]);
	int fill_size = fill->shape[0] * fill->shape[1];
	if (0 != verify_shape_tensor(shape, "fill_tensor")) {
		return -2;
	}
	if (1 > fill_size) {
		fprintf(stderr, "fill_tensor: fill tensor should contain something\n");
		return -3;
	}
	float *data = (float *)malloc(data_size * sizeof(float));
	if (NULL == data) {
		perror("fill_tensor: memory allocation error");
		return -4;
	}
	for (int i = 0; i < data_size; i++) {
		data[i] = fill->store->data[i % fill_size];
	}
	tensor_t *t = build_tensor_from_memory(data, data_size);
	if (NULL == t) {
		perror("fill_tensor: tensor creation failed");
	}
	t->shape[0] = (int)shape->store->data[0];
	t->shape[1] = (int)shape->store->data[1];
	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -4;
	}
	return 0;
}

int fill_random(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "fill_random")) {
		return -1;
	}
	tensor_t *shape = s->items[s->count - 1].as.t;
	if (0 != verify_shape_tensor(shape, "fill_random")) {
		return -2;
	}
	int data_size = (int)(shape->store->data[0] * shape->store->data[1]);
	float *data = (float *)malloc(data_size * sizeof(float));
	if (NULL == data) {
		perror("fill_random: memory allocation error");
		return -3;
	}
	for (int i = 0; i < data_size; i++) {
		data[i] = (float)(rand() / RAND_MAX);
	}
	tensor_t *t = build_tensor_from_memory(data, data_size);
	if (NULL == t) {
		perror("fill_random: tensor creation failed");
		return -4;
	}
	t->shape[0] = (int)shape->store->data[0];
	t->shape[1] = (int)shape->store->data[1];
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -5;
	}
	return 0;
}

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
	int new_size = (int)(shape->store->data[0] * shape->store->data[1]);
	int old_size = target->shape[0] * target->shape[1];
	if (old_size != new_size) {
		fprintf(stderr, "reshape_tensor: new size doesn't match old size\n");
		return -3;
	}
	tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
	if (NULL == t) {
		perror("reshape_tensor: memory allocation error");
		return -4;
	}
	t->shape[1] = (int)shape->store->data[1];
	t->shape[0] = (int)shape->store->data[0];
	t->store = target->store;
	t->store->ref_counter++;
	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		return -5;
	}
	return 0;
}

int ravel_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "ravel_tensor")) {
		return -1;
	}
	tensor_t *target = s->items[s->count - 1].as.t;
	int size = target->shape[0] * target->shape[1];
	tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
	if (NULL == t) {
		perror("ravel_tensor: memory allocation error");
		return -2;
	}
	t->shape[0] = 1;
	t->shape[1] = size;
	t->store = target->store;
	t->store->ref_counter++;
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -3;
	}
	return 0;
}

int shape_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "shape_tensor")) {
		return -1;
	}
	tensor_t *target = s->items[s->count - 1].as.t;
	storage_t *store = (storage_t *)malloc(sizeof(storage_t));
	if (NULL == store) {
		perror("shape_tensor: tensor creation failed");
		return -2;
	}
	store->ref_counter = 1;
	store->on_disk = false;
	store->data = (float *)malloc(MAX_DIM * sizeof(float));
	store->data[0] = (float)target->shape[0];
	store->data[1] = (float)target->shape[1];
	tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
	if (NULL == t) {
		free(store);
		perror("shape_tensor: tensor creation failed");
		return -3;
	}
	t->shape[0] = 1;
	t->shape[1] = 2;
	t->store = store;
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -4;
	}
	return 0;
}
		
int op_elem_by_elem(tf_stack_t *s, char *op_name, math_op func)
{
	if (0 != require_tensors(s, 2, op_name)) {
		return -1;
	}
	tensor_t *right = s->items[s->count - 1].as.t;
	tensor_t *left = s->items[s->count - 2].as.t;
	if (right->shape[0] != left->shape[0] ||
			right->shape[1] != left->shape[1]) {
		fprintf(stderr, "%s: incompatible shapes [%d, %d] != [%d, %d]\n", op_name, left->shape[0], left->shape[1], right->shape[0], right->shape[1]);
		return -2;
	}
	tensor_t *t = build_empty_tensor(left->shape[0], left->shape[1]);
	if (NULL == t) {
		perror("op_elem_by_elem: tensor creation failed");
		return -3;
	}
	int size = left->shape[0] * left->shape[1];

	func(t->store->data, left->store->data, right->store->data, size);

	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		destroy_tensor(t);
		return -4;
	}
	return 0;
}

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

int negate_floats(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "negate_floats")) {
		return -1;
	}
	tensor_t *t = s->items[s->count - 1].as.t;
	tensor_t *target = build_empty_tensor(t->shape[0], t->shape[1]);
	if (NULL == target) {
		perror("negate_floats: tensor creation failed");
		return -2;
	}
	int size = t->shape[0] * t->shape[1];
	for (int i = 0; i < size; i++) {
		target->store->data[i] = (float)(!t->store->data[i]);
	}
	drop_tensor(s);
	if (0 != push_tensor(s, target)) {
		destroy_tensor(target);
		return -3;
	}
	return 0;
}

// see if branchless is better than branched
int get_relu(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "get_relu")) {
		return -1;
	}
	tensor_t *t = s->items[s->count - 1].as.t;
	tensor_t *target = build_empty_tensor(t->shape[0], t->shape[1]);
	if (NULL == target) {
		perror("get_relu: tensor creation failed");
		return -2;
	}
	int size = t->shape[0] * t->shape[1];
	for (int i = 0; i < size; i++) {
		target->store->data[i] = (0 < t->store->data[i]) * t->store->data[i];
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

int sum_tensor(tf_stack_t *s)
{
	if (0 != require_tensors(s, 1, "sum_tensor")) {
		return -1;
	}
	tensor_t *t = s->items[s->count - 1].as.t;
	tensor_t *target = build_empty_tensor(1, 1);
	if (NULL == target) {
		perror("sum_tensor: tensor creation failed");
		return -2;
	}
	float total_sum = 0.0f;
	int size = t->shape[0] * t->shape[1];
	// #pragman omp parallel for reduction(+: total_sum)
	for (int i = 0; i < size; i++) {
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

int matmul(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "matmul")) {
		return -1;
	}
	tensor_t *right = s->items[s->count - 1].as.t;
	tensor_t *left = s->items[s->count - 2].as.t;
	if (left->shape[1] != right->shape[0]) {
		fprintf(stderr, "matmul: incompatible sizes: [%d, %d] @ [%d, %d]",
						left->shape[0], left->shape[1],
						right->shape[0], right->shape[1]);
		return -2;
	}
	int n = left->shape[0];
	int p = left->shape[1];
	int m = right->shape[1];
	tensor_t *target = build_zero_tensor(n, m);
	if (NULL == target) {
		perror("matmul: tensor creation failed\n");
		return -3;
	}
	blocked_multiply(left->store->data, right->store->data, target->store->data, n, m, p);
	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, target)) {
		destroy_tensor(target);
		return -4;
	}
	return 0;
}

int dot(tf_stack_t *s)
{
	if (0 != require_tensors(s, 2, "dot")) {
		return -1;
	}
	tensor_t *right = s->items[s->count - 1].as.t;
	tensor_t *left = s->items[s->count - 2].as.t;
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
		dot_sum += left->store->data[i] * right->store->data[i];
	}
	tensor_t *target = build_empty_tensor(1, 1);
	if (NULL == target) {
		perror("dot: tensor creation failed");
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
	// E QUI DOVREI METTERE LA FAMIGERATA convoluzione
	int n = left->shape[0];
	int m = left->shape[1];
	int k = kernel->shape[0];
	tensor_t *target = build_empty_tensor(n, m);
	if (NULL == target) {
		perror("convolute_tensors: tensor creation failed");
		return -3;
	}
	int offset = (k - 1) / 2;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
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
	tensor_t *t = func(filename);
	if (NULL == t) {
		return -3;
	}
	s->count--;
	if (0 != push_tensor(s, t)) {
		fprintf(stderr, "load_pgm: failed to push tensor on stack\n");
		destroy_tensor(t);
		return -4;
	}
	return 0;
}

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
		return -4;
	}
	s->count--;
	drop_tensor(s);
	return 0;
}

int write_pgm(FILE *fd, tensor_t *t)
{
	if (0 > fprintf(fd, "P5\n%d %d\n255\n", t->shape[1], t->shape[0])) {
		perror("write_pgm: failed to write header");
		return -1;
	}
	int size = t->shape[0] * t->shape[1];
	uint8_t *data = (uint8_t *)malloc(size);
	for (int i = 0; i < size; i ++) {
		data[i] = (uint8_t)(((0 < t->store->data[i]) * t->store->data[i] - ((1 > t->store->data[i]) * t->store->data[i] - 1)) * 255);
	}
	if (size != fwrite(data, 1, size, fd)) {
		fprintf(stderr, "write_pgm: failed to write data\n");
		return -2;
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
		case LOAD_PGM:			return load_from_file(s, "load_pgm", build_from_netpbm);
		case LOAD_TENSOR:		return load_from_file(s, "load_tensor", build_on_disk_tensor);
		case WRITE_PGM:			return save_to_file(s, "write_pgm", write_pgm);
		case WRITE_TENSOR:	return save_to_file(s, "write_tensor", write_tensor);

		case OP_UNKNOWN:
			fprintf(stderr, "execute_operation: unknown command\n");
			return -1;
	}
	return 0;
}

