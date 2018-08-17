// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "srad.h"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

#include <dragon.h>

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

static inline double time_diff(struct timeval tv_start, struct timeval tv_end)
{
    return (double)(tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + (double)(tv_end.tv_usec - tv_start.tv_usec) / 1000.0;
}

void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows/cols> <folder>\n", argv[0]);
	
	exit(1);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
    runTest( argc, argv);

    return EXIT_SUCCESS;
}


void
runTest( int argc, char** argv) 
{
    long rows, cols, size_I, size_R, niter = 10, iter;
    float *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

	float *J_cuda;
    float *C_cuda;
	float *E_C, *W_C, *N_C, *S_C;

	unsigned long r1, r2, c1, c2;

    char *folder;
    char *filepath;

    struct timeval tv_start, tv_end;
    double kernel_time = 0;       // in ms
    double map_time = 0;       // in ms
    double free_time = 0;       // in ms
	
	if (argc == 3)
	{
		rows = atol(argv[1]);  //number of rows in the domain
		cols = rows;           //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0))
        {
            fprintf(stderr, "rows and cols must be multiples of 16\n");
            exit(1);
		}
        folder = argv[2];
		r1   = 0;  //y1 position of the speckle
		r2   = 127;  //y2 position of the speckle
		c1   = 0;  //x1 position of the speckle
		c2   = 127;  //x2 position of the speckle
		lambda = 0.5; //Lambda value
		niter = 2; //number of iterations
	}
    else
    {
        usage(argc, argv);
    }

	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

    filepath = (char *)malloc(sizeof(char) * (strlen(folder) + 128));
    if (!filepath)
    {
        fprintf(stderr, "Cannot allocate filepath");
        exit(EXIT_FAILURE);
    }

    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/J.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(float) * size_I, D_F_READ | D_F_WRITE, (void **)&J) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/C_cuda.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(float) * size_I, D_F_READ | D_F_WRITE | D_F_CREATE | D_F_VOLATILE, (void **)&C_cuda) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/E_C.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(float) * size_I, D_F_READ | D_F_WRITE | D_F_CREATE | D_F_VOLATILE, (void **)&E_C) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/W_C.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(float) * size_I, D_F_READ | D_F_WRITE | D_F_CREATE | D_F_VOLATILE, (void **)&W_C) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/S_C.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(float) * size_I, D_F_READ | D_F_WRITE | D_F_CREATE | D_F_VOLATILE, (void **)&S_C) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/N_C.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(float) * size_I, D_F_READ | D_F_WRITE | D_F_CREATE | D_F_VOLATILE, (void **)&N_C) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    gettimeofday(&tv_end, NULL);
    map_time += time_diff(tv_start, tv_end);

    J_cuda = J;

	printf("Start the SRAD main loop\n");
    gettimeofday(&tv_start, NULL);
 for (iter=0; iter< niter; iter++){     
		sum=0; sum2=0;
        for (long i=r1; i<=r2; i++) {
            for (long j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);


	//Currently the input size must be divided by 16 - the block size
	long block_x = cols/(long)BLOCK_SIZE ;
    long block_y = rows/(long)BLOCK_SIZE ;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(block_x , block_y);
    

	//Run kernels
	srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
	srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 

    CUDA_CALL_SAFE(cudaThreadSynchronize());
}

    gettimeofday(&tv_end, NULL);
    kernel_time += time_diff(tv_start, tv_end);

	printf("Computation Done\n");

    free(filepath);

    gettimeofday(&tv_start, NULL);
    if (dragon_unmap(C_cuda) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap C_cuda\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(J_cuda) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap J_cuda\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(E_C) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap E_C\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(W_C) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap W_C\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(N_C) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap N_C\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(S_C) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap S_C\n");
        exit(EXIT_FAILURE);
    }
    gettimeofday(&tv_end, NULL);
    free_time += time_diff(tv_start, tv_end);
  
    printf("==> header: kernel_time (ms),map_time (ms),free_time (ms)\n");
    printf("==> data: %f,%f,%f\n", kernel_time, map_time, free_time);
}

