#include <stdlib.h>
#include <stdio.h>

#include "tensor.h"

struct tensor *build_tensor_from_memory(float *data, int l)
{
	struct tensor *t = (struct tensor *)malloc(sizeof(struct tensor));
	if (NULL == t) {
		return NULL;
	}
	t->shape[0] = l;
	t->ndim = 1;
	t->ref_counter = 1;
	t->on_disk = false;
	t->data = data;
	return t;
}

int destroy_tensor(struct tensor *t)
{
	if (t->ref_counter > 1) {
		t->ref_counter--;
	} else {
		if (t->on_disk) {
			free(t->data);
		}
		free(t);
	}
	return 0;
}
