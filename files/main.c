#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "tensor.h"
#include "stack.h"
#include "operators.h"

// sintassi tensore va corretta
int read_tensor(FILE *fd, float **tensor, int *count)
{
	int capacity = 10;
	int inserted = 0;
	float *arr = (float *)malloc(capacity * sizeof(float));
	int c;
	while ((c = fgetc(fd)) != EOF) {
		while (isspace(c)) {
			c = fgetc(fd);	
		}
		if (']' == c) {
			*tensor = arr;
			*count  = inserted;
			return 0;
		}
		ungetc(c, fd);
		float val;
		if (fscanf(fd, "%f", &val) != 1) {
			printf("read_tensor: Erorr in tensor format");
			free(arr);
			return -2;
		}
		if (capacity > inserted) {
			arr[inserted++] = val;
		} else {
			capacity *= 2;
			float *temp = (float *)realloc(arr, capacity * sizeof(float));
			if (NULL == temp) {
				perror("read_tensor: Failed to allocate memory");
				free(arr);
				return -1;
			}
			arr = temp;
			arr[inserted++] = val;
		}
	}
	printf("read_tensor: EOF reached");
	free(arr);
	return -4;
}

int parse_file(char* fn)
{
	FILE *fd = fopen(fn, "r+");
	if (NULL == fd) {
		printf("parse_file: Cannot open file %s", fn);
		return -1;
	}
	tf_stack_t *s = create_stack(20);
	int c;
	while ((c = fgetc(fd)) != EOF) {
		if (isspace(c)) {
			continue;
		} else if ('[' == c) {
			float *data = NULL;
			int inserted  = 0;
			if (read_tensor(fd, &data, &inserted) != 0) {
				fclose(fd);
			} else {
				tensor_t *t = build_tensor_from_memory(data, inserted);
				if (NULL == t) {
					perror("parse_file: Failed to build tensor from memory");
					return -2;
				}
				push_tensor(s, t);
			}
		} else if ('"' == c) {
			char filename[256];
			if (fscanf(fd, "%255[^\"]\"", filename) != 1) {
				fprintf(stderr, "Filename too long or not double-quote-terminated\n");
			}
			push_string(s, filename);
		} else {
			operation_t op = get_operation_from_char(c);
			if (OP_UNKNOWN == op) {
				fprintf(stderr, "parse_file: unknown command: %c\n", c);
				break;
			} else if (0 != execute_operation(s, op)) {
				fprintf(stderr, "Fatal error at offset %ld. Aborting execution.\n", ftell(fd));
				break;
			}
		}
	}
	destroy_stack(s);
	return 0;
}

int main(int argc, char* argv[])
{
	if (2 > argc) {
		perror("No commands or argument provided");
		return 1;
	} else if (2 == argc) {
		parse_file(argv[1]);
	} else {
		printf("Non ancora implementato");
	}
	return 0;
}



