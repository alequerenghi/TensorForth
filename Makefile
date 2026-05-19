CC				= gcc
CFLAGS 		= -O3 -march=native -Wall -pedantic -std=gnu18 -fopenmp -g

SRC				= src
LIB 			= lib
BUILD			= build
PROGNAME 	= tensorforth

CHEADERS 	= ${wildcard ${SRC}/*.c}
LIB_SRCS	= ${filter-out ${SRC}/main.c, ${CHEADERS}}
LIB_OBJS	= ${patsubst ${SRC}/%.c, ${BUILD}/%.o, ${LIB_SRCS}}

all: ${LIB}/libtensorfort.a ${PROGNAME}

${LIB}/libtensorfort.a: ${LIB_OBJS}
	@mkdir -p ${LIB}
	ar rs $@ $^

${PROGNAME}: ${SRC}/main.c ${LIB}/libtensorfort.a
	${CC} ${CFLAGS} -o $@ $< -L${LIB} -ltensorfort

${BUILD}/%.o: ${SRC}/%.c
	@mkdir -p ${BUILD}
	${CC} ${CFLAGS} -c $< -o $@

.PHONY: clean clean-all
clean: 
	rm -rf ${BUILD}

clean-all: clean
	rm -rf ${LIB}
	rm -f ${PROGNAME}
