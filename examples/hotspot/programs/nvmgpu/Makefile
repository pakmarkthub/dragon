include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

EXE = bin/hotspot

release: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -I../../../library/include -ldragon

enum: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -I../../../library/include -ldragon

debug: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -I../../../library/include -ldragon

debugenum: $(SRC)
	mkdir -p bin
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch compute_60 -code sm_60 -I../../../library/include -ldragon

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
