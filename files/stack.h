#ifndef _STACK_H
#define _STACK_H

#include "tensor.h"

/**
 * @file: stack.h
 * @author: ALESSANDRO QUERENGHI
 *
 * This file contains the structs and function headers to manage the stack for
 * the program TensorForth
 */

/**
 * This enum is required to know what is the type of a stack element
 */
typedef enum item_type {
	TYPE_STRING,
	TYPE_TENSOR
} item_type_t;

/**
 * This struct models the element inside the stack. The union is used so that
 * both tensors and file names can be stored in the stack and the type element
 * is used to know which is which
 */
typedef struct stack_item {
	item_type_t type;
	union {
		char *filename;
		tensor_t *t;
	} as;
} stack_item_t;

/**
 * The actual stack struct, it is basically an array of stack_item-s 
 */
typedef struct {
	stack_item_t *items;
	int count;
	int capacity;
} tf_stack_t;

/**
 * Initializes the empty stack
 *
 * @param[in] capacity The initial capacity
 * @return The pointer to the stack or NULL if fails
 */
tf_stack_t *create_stack(int capacity);

/**
 * Destroys the stack by destroying all elements inside and freeing the memory
 * allocated for the stack itself
 *
 * @param[in] s The pointer to the stack to destroy
 */
void destroy_stack(tf_stack_t *s);

/**
 * Pushes a tensor onto the stack, allocating more memory if necessary
 *
 * @param[in] s The pointer to the stack
 * @param[in] t The tensor to be pushed
 * @return 0 if the tensor has been pushed successfully or a negative integer
 * if fails
 */
int push_tensor(tf_stack_t *s, tensor_t *t);

/**
 * Pushes a string onto the stack, allocating more memory if necessary
 *
 * @param[in] s The pointer to the stack
 * @param[in] fn The string to be pushed
 * @return 0 if the string has been pushed successfully or a negative integer
 * if fails
 */
int push_string(tf_stack_t *s, char *fn);

#endif
