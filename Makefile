COMPILER ?= gcc

ifeq ($(COMPILER), gcc)
	CC				= gcc
	CFLAGS		= -O3 -march=native -Wall -pedantic -std=gnu23 -fopenmp

else ifeq ($(COMPILER), nvc)
	CC				= nvc
	CFLAGS		= -O3 -std=gnu18 -mp=gpu -gpu=mem:managed -gopt

else
	$(error Unknown COMPILER option '$(COMPILER)'. Please use COMPILER=nvc or COMPILER=gcc)
endif

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
