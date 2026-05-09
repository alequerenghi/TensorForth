#ifndef _STACK_H
#define _STACK_H

typedef enum {
	TYPE_STRING,
	TYPE_TENSOR//,
	// TYPE_OP
} itemp_type_t;

typedef struct {
	itemp_type_t type;
	union {
		char *filename;
		struct tensor *t;
		//enum *operator
	} as;
} stack_item_t;

typedef struct {
	stack_item_t *items;
	int count;
	int capacity;
} tf_stack_t;

tf_stack_t *create_stack(int capacity);

void destroy_stack(tf_stack_t *s);

int push_tensor(tf_stack_t *s, struct tensor *t);

int push_generic(tf_stack_t *s, stack_item_t item);

#endif
