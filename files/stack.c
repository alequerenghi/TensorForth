#include <stdlib.h>
#include <stdio.h>

#include "stack.h"
#include "tensor.h"

/**
 * @file: stack.c
 * @author: ALESSANDRO QUERENGHI
 *
 * This files contains the implementation of the functions defined in stack.h
 */

tf_stack_t *create_stack(int capacity)
{
	tf_stack_t *s = (tf_stack_t *)malloc(sizeof(tf_stack_t));
	if (NULL == s) {
		return NULL;
	}
	s->items = (stack_item_t *)malloc(capacity * sizeof(stack_item_t));
	if (NULL == s->items) {
		free(s);
		return NULL;
	}
	s->count = 0;
	s->capacity = capacity;
	return s;
}

void destroy_stack(tf_stack_t *s)
{
	// iterates on each element in the stack and removes it
	for (int i = 0; i < s->count; i++) {
			item_type_t type = s->items[i].type;
		switch (type) {
			case TYPE_TENSOR:
				destroy_tensor(s->items[i].as.t);
				break;
			case TYPE_STRING:
				// since strings are not allocated dynamically there is nothing to do
				break;
		}
	}
	free(s->items);
	free(s);
}

/**
 * Function to generalize pushing an element onto the stack, manages memory
 * allocation (if more space is needed)
 *
 * @param[in] s The pointer to the stack on which the push is performed
 * @param[in] item The item being pushed onto the stack, can be either a tensor
 * or a string
 * @return 0 if the item has been pushed successfully or a negative integer if
 * it fails
 */
int push_generic(tf_stack_t *s, stack_item_t item)
{
	// check if there is room left
	if (s->capacity - 1 <= s->count) {
		// double capacity
		int new_capacity = s->capacity * 2;
		stack_item_t *temp = (stack_item_t *)realloc(s->items, new_capacity * sizeof(stack_item_t));
		if (NULL == temp) {
			perror("push_tensor: failed to increase stack memory");
			return -1;
		}
		s->items = temp;
		s->capacity = new_capacity;
	}
	// put the element on the stack
	s->items[s->count++] = item;
	return 0;
}

int push_tensor(tf_stack_t *s, tensor_t *t)
{
	stack_item_t item;
	item.type = TYPE_TENSOR;
	item.as.t = t;
	// use push_generic to put the element onto the stack to centralize logic
	return push_generic(s, item);
}

int push_string(tf_stack_t *s, char *fn)
{
	stack_item_t item;
	item.type = TYPE_STRING;
	item.as.filename = fn;
	// use push_generic to put the element onto the stack to centralize logic
	return push_generic(s, item);
}

