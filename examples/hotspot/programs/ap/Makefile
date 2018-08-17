include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

EXE = bin/hotspot

release: $(SRC)
	mkdir -p bin
	$(CC) $(SRC) -o hotspot.o -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L../../../../gpufs/libgpufs/release/ -lgpufs -arch compute_60 -code sm_60 -std=c++11 -dc
	$(CC) $(KERNEL_DIM) hotspot.o -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L../../../../gpufs/libgpufs/release/ -lgpufs -arch compute_60 -code sm_60 -std=c++11

enum: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

debug: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

debugenum: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt *.o
