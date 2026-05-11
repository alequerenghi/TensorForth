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
		free(s);
		return NULL;
	}
	t->shape[0] = 1;
	t->shape[1] = l;
	t->store = s;
	return t;
}

tensor_t *build_empty_tensor(int rows, int columns)
{
	float *data = (float *)malloc((rows * columns) * sizeof(float));
	if (NULL == data) {
		return NULL;
	}
	tensor_t *t = build_tensor_from_memory(data, rows * columns);
	if (NULL == t) {
		free(data);
		return NULL;
	}
	t->shape[0] = rows;
	t->shape[1] = columns;
	return t;
}

tensor_t *build_zero_tensor(int rows, int columns)
{
	float *data = (float *)calloc((rows * columns) * sizeof(float));
	if (NULL == data) {
		return NULL;
	}
	tensor_t *t = build_tensor_from_memory(data, rows * columns);
	if (NULL == t) {
		free(data);
		return NULL;
	}
	t->shape[0] = rows;
	t->shape[1] = columns;
	return t;
}

// tensor_t *build_empty_mmap_tensor(int rows, int columns, const char *filepath)
// {
//     size_t num_elements = (size_t)rows * columns;
//     size_t byte_size = num_elements * sizeof(float);
// 
//     // 1. Open a new file (Create it if missing, truncate it if it exists)
//     int fd = open(filepath, O_RDWR | O_CREAT | O_TRUNC, 0666);
//     if (fd == -1) {
//         perror("build_mmap_tensor: failed to open file");
//         return NULL;
//     }
// 
//     // 2. MAGIC STEP: Stretch the file to the exact size we need!
//     if (ftruncate(fd, byte_size) == -1) {
//         perror("build_mmap_tensor: ftruncate failed");
//         close(fd);
//         return NULL;
//     }
// 
//     // 3. Map the stretched file into memory
//     float *data = (float *)mmap(NULL, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//     if (data == MAP_FAILED) {
//         perror("build_mmap_tensor: mmap failed");
//         close(fd);
//         return NULL;
//     }
// 
//     // 4. Manually construct the Storage struct for disk tracking
//     storage_t *store = (storage_t *)malloc(sizeof(storage_t));
//     if (store == NULL) {
//         munmap(data, byte_size);
//         close(fd);
//         return NULL;
//     }
//     
//     store->data = data;
//     store->size = num_elements;
//     store->ref_counter = 1;
//     
//     // Set the mmap-specific tracking variables so destroy_tensor works!
//     store->on_disk = true;
//     store->fd = fd;
//     store->mmap_size = byte_size;
// 
//     // 5. Construct the View (Tensor) struct
//     tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
//     if (t == NULL) {
//         free(store);
//         munmap(data, byte_size);
//         close(fd);
//         return NULL;
//     }
//     
//     t->shape[0] = rows;
//     t->shape[1] = columns;
//     t->store = store;
// 
//     return t;
// }

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
