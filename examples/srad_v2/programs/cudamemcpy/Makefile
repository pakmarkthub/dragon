include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = srad.cu

EXE = bin/srad

all: release generator

release: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

enum: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

debug: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

debugenum: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

generator: generator.cu
	mkdir -p bin
	$(CC) $(KERNEL_DIM) generator.cu -o ./bin/generator -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

clean: $(SRC)
	rm -f bin/*
