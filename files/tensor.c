#include <stdlib.h>
#include <stdio.h>

#include "tensor.h"

tensor_t *build_tensor_from_memory(float *data, int l)
{
	storage_t *s = (storage_t *)malloc(sizeof(storage_t));
	if (NULL == s) {
		return NULL;
	}
	s->ref_counter = 1;
	s->on_disk = false;
	s->data = data;
	tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
	if (NULL == t) {
		return NULL;
	}
	t->shape[0] = 1;
	t->shape[1] = l;
	t->store = s;
	return t;
}

void destroy_tensor(tensor_t *t)
{
	if (NULL == t) {
		return;
	}
	storage_t *s = t->store;
	if (1 < s->ref_counter) {
		s->ref_counter--;
	} else {
		if (!s->on_disk) {
			free(s->data);
		}
		free(s);
	}
	free(t);
}
