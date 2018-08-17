include ../../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -arch compute_60 -code sm_60

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O2
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

all: backprop generator

backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o 
	mkdir -p bin
	$(CC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o -o bin/backprop -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

generator: generator.o backprop.o imagenet.o backprop_cuda.o
	mkdir -p bin
	$(CC) $(CC_FLAGS) backprop.o imagenet.o generator.o backprop_cuda.o -o bin/generator -lm -lcuda -lcudart

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

gnerator.o: generator.c backprop.h
	$(CC) $(CC_FLAGS) generator.c -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f bin/* *.o *~ backprop backprop_cuda.linkinfo
