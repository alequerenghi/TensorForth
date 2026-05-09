#include <stdlib.h>
#include <stdio.h>

#include "stack.h"
#include "tensor.h"

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
	for (int i = 0; i < s->count; i++) {
			item_type_t type = s->items[i].type;
		switch (type) {
			case TYPE_TENSOR:
				destroy_tensor(s->items[i].as.t);
				break;
			case TYPE_STRING:
				break;
		}
	}
	free(s->items);
	free(s);
}

int push_generic(tf_stack_t *s, stack_item_t item)
{
	if (s->capacity-1 <= s->count) {
		int new_capacity = s->capacity * 2;
		stack_item_t *temp = (stack_item_t *)realloc(s->items, new_capacity * sizeof(stack_item_t));
		if (NULL == temp) {
			perror("push_tensor: failed to increase stack memory");
			return -1;
		}
		s->items = temp;
		s->capacity = new_capacity;
	}
	s->items[s->count++] = item;
	return 0;
}

int push_tensor(tf_stack_t *s, struct tensor *t)
{
	stack_item_t item;
	item.type = TYPE_TENSOR;
	item.as.t = t;
	return push_generic(s, item);
}

int push_string(tf_stack_t *s, char *fn)
{
	stack_item_t item;
	item.type = TYPE_STRING;
	item.as.filename = fn;
	return push_generic(s, item);
}
