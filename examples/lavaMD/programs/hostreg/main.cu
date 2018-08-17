//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	UPDATE
//========================================================================================================================================================================================================200

//	14 APR 2011 Lukasz G. Szafaryn

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>					// (in path known to compiler)			needed by printf
#include <stdlib.h>					// (in path known to compiler)			needed by malloc
#include <stdbool.h>				// (in path known to compiler)			needed by true/false
#include <string.h>

#include <cuda.h>
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./util/timer/timer.h"			// (in path specified here)
#include "./util/num/num.h"				// (in path specified here)

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"						// (in the current directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel/kernel_gpu_cuda_wrapper.h"	// (in library path specified here)

#define CUDA_CALL_SAFE(f) \
    do \
    {                                                        \
        cudaError_t _cuda_error = f;                         \
        if (_cuda_error != cudaSuccess)                      \
        {                                                    \
            fprintf(stderr,  \
                "%s, %d, CUDA ERROR: %s %s\n",  \
                __FILE__,   \
                __LINE__,   \
                cudaGetErrorName(_cuda_error),  \
                cudaGetErrorString(_cuda_error) \
            ); \
            abort(); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)        

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

static inline double time_diff(struct timeval tv_start, struct timeval tv_end)
{
    return (double)(tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + (double)(tv_end.tv_usec - tv_start.tv_usec) / 1000.0;
}

int 
main(	int argc, 
		char *argv [])
{

	printf("thread block size of kernel = %d \n", NUMBER_THREADS);
	//======================================================================================================================================================150
	//	CPU/MCPU VARIABLES
	//======================================================================================================================================================150

	// timer
    struct timeval tv_start, tv_end;

    double kernel_time = 0;       // in ms
    double map_time = 0;       // in ms
    double free_time = 0;       // in ms

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR* rv_cpu;
	fp* qv_cpu;
	FOUR_VECTOR* fv_cpu;

    char *folder;
    char *filepath;

	//======================================================================================================================================================150
	//	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================150

	// assing default values
	dim_cpu.boxes1d_arg = 1;

    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <boxes1d> <folder>\n", argv[0]);
        abort();
        exit(EXIT_FAILURE);
    }

    dim_cpu.cur_arg = 1;
	dim_cpu.boxes1d_arg = atoi(argv[1]);

    if (dim_cpu.boxes1d_arg < 0)
    {
        fprintf(stderr, "ERROR: Wrong value to -boxes1d parameter, cannot be <=0\n");
        abort();
        exit(EXIT_FAILURE);
    }

    folder = argv[2];

	// Print configuration
	printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150

	par_cpu.alpha = 0.5;


	//======================================================================================================================================================150
	//	DIMENSIONS
	//======================================================================================================================================================150

	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	//======================================================================================================================================================150
	//	SYSTEM MEMORY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	BOX
	//====================================================================================================100

    filepath = (char *)malloc(sizeof(char) * (strlen(folder) + 128));

    int fd_box, fd_rv, fd_qv, fd_fv;

	// allocate boxes
    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/box.mem", folder);
    if ((fd_box = open(filepath, O_LARGEFILE | O_RDWR)) < 0)
    {
        fprintf(stderr, "Cannot open file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((box_cpu = (box_str *)mmap(NULL, dim_cpu.box_mem, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_box, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(box_cpu, dim_cpu.box_mem, cudaHostRegisterDefault));

	//====================================================================================================100
	//	PARAMETERS, DISTANCE, CHARGE AND FORCE
	//====================================================================================================100

	// input (distances)
    sprintf(filepath, "%s/rv.mem", folder);
    if ((fd_rv = open(filepath, O_LARGEFILE | O_RDWR)) < 0)
    {
        fprintf(stderr, "Cannot open file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((rv_cpu = (FOUR_VECTOR *)mmap(NULL, dim_cpu.space_mem, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_rv, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(rv_cpu, dim_cpu.space_mem, cudaHostRegisterDefault));

	// input (charge)
    sprintf(filepath, "%s/qv.mem", folder);
    if ((fd_qv = open(filepath, O_LARGEFILE | O_RDWR)) < 0)
    {
        fprintf(stderr, "Cannot open file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((qv_cpu = (fp *)mmap(NULL, dim_cpu.space_mem2, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_qv, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(qv_cpu, dim_cpu.space_mem2, cudaHostRegisterDefault));


	// output (forces)
    sprintf(filepath, "%s/fv.hostreg.mem", folder);
    if ((fd_fv = open(filepath, O_LARGEFILE | O_RDWR | O_CREAT)) < 0)
    {
        fprintf(stderr, "Cannot open file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fd_fv, dim_cpu.space_mem) != 0)
    {
        fprintf(stderr, "Cannot truncate file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((fv_cpu = (FOUR_VECTOR *)mmap(NULL, dim_cpu.space_mem, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_fv, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(fv_cpu, dim_cpu.space_mem, cudaHostRegisterDefault));
    gettimeofday(&tv_end, NULL);

    map_time += time_diff(tv_start, tv_end);

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU_CUDA
	//====================================================================================================100

	kernel_gpu_cuda_wrapper(par_cpu,
							dim_cpu,
							box_cpu,
							rv_cpu,
							qv_cpu,
							fv_cpu,
                            &kernel_time);

    CUDA_CALL_SAFE(cudaDeviceSynchronize());

	//======================================================================================================================================================150
	//	SYSTEM MEMORY DEALLOCATION
	//======================================================================================================================================================150

	// dump results

    gettimeofday(&tv_start, NULL);
    CUDA_CALL_SAFE(cudaHostUnregister(box_cpu));
    if (msync(box_cpu, dim_cpu.box_mem, MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync box_cpu\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(box_cpu, dim_cpu.box_mem) != 0)
    {
        fprintf(stderr, "Cannot munmap box_cpu\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_box);
    close(fd_box);

    CUDA_CALL_SAFE(cudaHostUnregister(rv_cpu));
    if (msync(rv_cpu, dim_cpu.space_mem, MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync rv_cpu\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(rv_cpu, dim_cpu.space_mem) != 0)
    {
        fprintf(stderr, "Cannot munmap rv_cpu\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_rv);
    close(fd_rv);

    CUDA_CALL_SAFE(cudaHostUnregister(qv_cpu));
    if (msync(qv_cpu, dim_cpu.space_mem2, MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync qv_cpu\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(qv_cpu, dim_cpu.space_mem2) != 0)
    {
        fprintf(stderr, "Cannot munmap qv_cpu\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_qv);
    close(fd_qv);

    CUDA_CALL_SAFE(cudaHostUnregister(fv_cpu));
    if (msync(fv_cpu, dim_cpu.space_mem, MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync fv_cpu\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(fv_cpu, dim_cpu.space_mem) != 0)
    {
        fprintf(stderr, "Cannot munmap fv_cpu\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_fv);
    close(fd_fv);
    gettimeofday(&tv_end, NULL);

    free_time += time_diff(tv_start, tv_end);

    free(filepath);

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150
    printf("==> header: kernel_time (ms),map_time (ms),free_time (ms)\n");
    printf("==> data: %f,%f,%f\n", kernel_time, map_time, free_time);


	//======================================================================================================================================================150
	//	RETURN
	//======================================================================================================================================================150

	return 0.0;																					// always returns 0.0

}
