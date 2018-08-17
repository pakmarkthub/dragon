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
#include <unistd.h>

#include <cuda.h>
#include <sys/time.h>

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
    double writefile_time = 0;       // in ms
    double readfile_time = 0;       // in ms

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR* rv_cpu;
	fp* qv_cpu;
	FOUR_VECTOR* fv_cpu;

    FILE *f;
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

	// allocate boxes
    CUDA_CALL_SAFE(cudaMallocManaged(&box_cpu, dim_cpu.box_mem));

    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/box.mem", folder);
    if ((f = fopen(filepath, "rb")) == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    if (fread(box_cpu, dim_cpu.box_mem, 1, f) != 1)
    {
        fprintf(stderr, "Cannot read %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    fclose(f);
    gettimeofday(&tv_end, NULL);

    readfile_time += time_diff(tv_start, tv_end);

	//====================================================================================================100
	//	PARAMETERS, DISTANCE, CHARGE AND FORCE
	//====================================================================================================100

	// input (distances)
    CUDA_CALL_SAFE(cudaMallocManaged(&rv_cpu, dim_cpu.space_mem));

    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/rv.mem", folder);
    if ((f = fopen(filepath, "rb")) == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    if (fread(rv_cpu, dim_cpu.space_mem, 1, f) != 1)
    {
        fprintf(stderr, "Cannot read %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    fclose(f);
    gettimeofday(&tv_end, NULL);

    readfile_time += time_diff(tv_start, tv_end);

	// input (charge)
    CUDA_CALL_SAFE(cudaMallocManaged(&qv_cpu, dim_cpu.space_mem2));

    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/qv.mem", folder);
    if ((f = fopen(filepath, "rb")) == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    if (fread(qv_cpu, dim_cpu.space_mem2, 1, f) != 1)
    {
        fprintf(stderr, "Cannot read %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    fclose(f);
    gettimeofday(&tv_end, NULL);

    readfile_time += time_diff(tv_start, tv_end);

	// output (forces)
    CUDA_CALL_SAFE(cudaMallocManaged(&fv_cpu, dim_cpu.space_mem));

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

	//======================================================================================================================================================150
	//	SYSTEM MEMORY DEALLOCATION
	//======================================================================================================================================================150

	// dump results

    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/fv.uvm.mem", folder);
    if ((f = fopen(filepath, "wb")) == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    if (fwrite(fv_cpu, dim_cpu.space_mem, 1, f) != 1)
    {
        fprintf(stderr, "Cannot write %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    gettimeofday(&tv_end, NULL);

    writefile_time += time_diff(tv_start, tv_end);


	CUDA_CALL_SAFE(cudaFree(rv_cpu));
	CUDA_CALL_SAFE(cudaFree(qv_cpu));
	CUDA_CALL_SAFE(cudaFree(fv_cpu));
	CUDA_CALL_SAFE(cudaFree(box_cpu));

    free(filepath);

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150
    printf("==> header: kernel_time (ms),writefile_time (ms),readfile_time (ms)\n");
    printf("==> data: %f,%f,%f\n", kernel_time, writefile_time, readfile_time);


	//======================================================================================================================================================150
	//	RETURN
	//======================================================================================================================================================150

	return 0.0;																					// always returns 0.0

}
