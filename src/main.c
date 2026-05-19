#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "tensor.h"
#include "stack.h"
#include "operators.h"

/**
 * @file: main.c
 * @author: ALESSANDRO QUERENGHI
 * @id:			IN2300001
 *
 * This file contains the code for the parser and the main() function.
 * Compilation of this code produces the executable `tensorforth`
 */

/**
 * Called after encountering '[', parses the file until a ']' is encountered
 * and generates a float array from it.
 * The program requires at least a whiespace character between each float and
 * at least one whiespace character after '[' and before '['. Any other
 * character will crash the program.
 *
 * @param[in] fd The file stream from which data is read from
 * @param[out] tensor The float array read from the file
 * @param[out] count The actual number of elements contained in the array
 * @return 0 if success or a negative number if the tensor syntax is not
 * followed
 */
int read_tensor(FILE *fd, float **tensor, int *count)
{
	// initial capacity
	int capacity = 10;
	// number of inserted elements
	int inserted = 0;
	float *arr = (float *)malloc(capacity * sizeof(float));
	if (NULL == arr) {
		perror("read_tensor: Failed to allocate memory");
		return -1;
	}
	// make sure that there is one space after '['
	int c = fgetc(fd);
	if (!isspace(c)) {
		fprintf(stderr, "read_tensor: Error in tensor format\n");
		free(arr);
		return -2;
	}
	while ((c = fgetc(fd)) != EOF) {
		// ignore whitespace
		while (isspace(c)) {
			c = fgetc(fd);	
		}
		// array completely read
		if (']' == c) {
			*tensor = arr;
			*count  = inserted;
			return 0;
		}
		// put back a character
		ungetc(c, fd);
		float val;
		if (fscanf(fd, "%f", &val) != 1) {
			printf("read_tensor: Erorr in tensor format");
			free(arr);
			return -3;
		}
		// if reached aray limit, double in size using realloc
		if (capacity <= inserted) {
			capacity *= 2;
			float *temp = (float *)realloc(arr, capacity * sizeof(float));
			if (NULL == temp) {
				perror("read_tensor: Failed to allocate memory");
				free(arr);
				return -4;
			}
			arr = temp;
		}
		// insert element
		arr[inserted++] = val;
		// check that one whiespace character follows the float read
		c = fgetc(fd);
		if (!isspace(c)) {
			fprintf(stderr, "read_tensor: Error in tensor format\n");
			free(arr);
			return -5; 
		}
	}
	// something broke
	fprintf(stderr, "read_tensor: EOF reached");
	free(arr);
	return -6;
}

/**
 * Parses a file containing TensorFort syntax and executes the program. If the
 * syntax is incorrect the program will fail.
 *
 * @param[in] fn File name of the program to execute
 * @return 0 if success or a negative integer if the syntax is not followed or
 * a command failed
 */
int parse_file(char* fn)
{
	FILE *fd = fopen(fn, "r+");
	if (NULL == fd) {
		fprintf(stderr, "parse_file: Cannot open file %s", fn);
		return -1;
	}
	tf_stack_t *s = create_stack(20);
	if (NULL == s) {
		fprintf(stderr, "parse_file: failed to create stack\n");
		fclose(fd);
		return -2;
	}
	int c;
	while ((c = fgetc(fd)) != EOF) {
		// ignore whitespace
		if (isspace(c)) {
			continue;
		} else if ('[' == c) {
			// start parsing float array
			float *data = NULL;
			int inserted = 0;
			if (0 != read_tensor(fd, &data, &inserted)) {
				fclose(fd);
				destroy_stack(s);
				return -3;
			}
			// initialize tensor from parsed data
			tensor_t *t = build_empty_tensor(1, inserted);
			if (NULL == t) {
				perror("parse_file: Failed to build tensor from memory");
				fclose(fd);
				destroy_stack(s);
				free(data);
				return -4;
			}
			t->store->data = data;
			// push tensor
			if (0 != push_tensor(s, t)) {
				fclose(fd);
				destroy_stack(s);
				destroy_tensor(t);
				return -5;
			}
		} else if ('"' == c) {
			// reading a filename of maximum 256 chars
			char filename[256];
			// read a filename '"' terminated
			if (fscanf(fd, "%255[^\"]\"", filename) != 1) {
				fprintf(stderr, "Filename too long or not double-quote-terminated\n");
				fclose(fd);
				destroy_stack(s);
				return -6;
			}
			if (0 != push_string(s, filename)) {
				fclose(fd);
				destroy_stack(s);
				return -7;
			}
		} else {
			// reading a command, probably
			operation_t op = get_operation_from_char(c);
			if (OP_UNKNOWN == op) {
				fprintf(stderr, "parse_file: unknown command: %c\n", c);
				fclose(fd);
				destroy_stack(s);
				return -8;
			}
			// execute the parsed command
			if (0 != execute_operation(s, op)) {
				fprintf(stderr, "Fatal error at offset %ld. Aborting execution.\n", ftell(fd));
				fclose(fd);
				destroy_stack(s);
				return -9;
			}
		}
		// make sure that each command, filename or tensor is separated by at least
		// one whitespace character
		c = fgetc(fd);
		if (EOF != c && !isspace(c)) {
			fprintf(stderr, "parse_file: Syntax error at offset %ld. Tokens must be whitepsace-separated\n", ftell(fd) - 1);
			fclose(fd);
			destroy_stack(s);
			return -10;
		}
	}
	fclose(fd);
	destroy_stack(s);
	return 0;
}

/**
 * Program entrypoint
 *
 */
int main(int argc, char* argv[])
{
	if (2 > argc) {
		fprintf(stderr, "No commands or argument provided\n");
		return 1;
	} else if (2 == argc) {
		parse_file(argv[1]);
	} else {
		printf("Non ancora implementato");
	}
	return 0;
}

