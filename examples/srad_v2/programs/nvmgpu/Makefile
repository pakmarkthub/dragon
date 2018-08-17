include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = srad.cu

EXE = bin/srad

all: release

release: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -ldragon

enum: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -ldragon

debug: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -ldragon

debugenum: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -ldragon

clean: $(SRC)
	rm -f bin/*
