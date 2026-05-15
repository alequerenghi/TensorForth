CC = gcc
CFLAGS = -O3 -march=native -Wall -pedantic -std=gnu18 -fopenmp -g
LIB = lib
CHEADERS = ${wildcard *.}
PROGNAME = tensorforth

all : ${LIB} ${PROGNAME}

${PROGNAME}: main.o ${LIB}/libtensorfort.a
	${CC} ${CFLAGS} -o $@ $< -L${LIB} -ltensorfort

${LIB}:
	mkdir -p $@

${LIB}/libtensorfort.a: tensor.o stack.o operators.o matrix_multiply.o
	ar rs $@ $^

main.o: main.c ${CHEADERS}
	${CC} ${CFLAGS} -c $<

%.o: %.c %.h
	${CC} ${CFLAGS} -c $<

.PHONY: clean clean-all
clean: 
	rm -f *.o

clean-all: clean
	rm -rf ${LIB}
	rm -f ${PROGNAME}
