include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu

EXE = bin/pathfinder

release:
	mkdir -p bin
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -ldragon

enum:
	mkdir -p bin
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -ldragon

debug:
	mkdir -p bin
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -ldragon

debugenum:
	mkdir -p bin
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -ldragon

clean:
	rm -f bin/*
