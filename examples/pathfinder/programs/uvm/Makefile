include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu

EXE = bin/pathfinder

release:
	mkdir -p bin
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

enum:
	mkdir -p bin
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

debug:
	mkdir -p bin
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

debugenum:
	mkdir -p bin
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

clean:
	rm -f bin/*
