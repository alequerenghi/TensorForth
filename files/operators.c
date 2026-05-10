#include <stdlib.h>
#include <stdio.h>

#include "stack.h"
#include "tensor.h"
#include "operators.h"

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
		case '-': return OP_RAVEL;
		case '#': return OP_SHAPE;

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
	if (0 != push_tensor(s, t)) {
		return -2;
	}
	t->store->ref_counter++;
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
	if (0 != push_tensor(s, t)) {
		return -2;
	}
	t->store->ref_counter++;
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
	t->shape[0] = (int)shape->store->data[0];
	t->shape[1] = (int)shape->store->data[1];
	drop_tensor(s);
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
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
	t->shape[0] = (int)shape->store->data[0];
	t->shape[1] = (int)shape->store->data[1];
	drop_tensor(s);
	if (0 != push_tensor(s, t)) {
		return -4;
	}
	return 0;
}

// Casin
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

int execute_operation(tf_stack_t *s, operation_t op)
{
	switch (op) {
		// UTILITY
		case OP_PRINT: return print_tensor(s);
		// MANIPULATION
		case OP_DUPLICATE: return duplicate_tensor(s);
		case OP_SWAP: return swap_tensors(s);
		case OP_OVER: return overput_tensor(s);
		case OP_DROP: return drop_tensor(s);
		// GENERATION
		case OP_FILL: return fill_tensor(s);
		case OP_RAND: return fill_random(s);
		// SHAPING
		case OP_RESHAPE: return reshape_tensor(s);
		case '-': return OP_RAVEL;
		case '#': return OP_SHAPE;

		case OP_UNKNOWN:
			fprintf(stderr, "execute_operation: unknown command\n");
			return -1;
	}
	return 0;
}

