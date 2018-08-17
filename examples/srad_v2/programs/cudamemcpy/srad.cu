// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include "srad.h"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

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
    FILE *fp;

    struct timeval tv_start, tv_end;
    double kernel_time = 0;       // in ms
    double writefile_time = 0;       // in ms
    double readfile_time = 0;       // in ms
    double d2h_memcpy_time = 0;       // in ms
    double h2d_memcpy_time = 0;       // in ms
	
 
	if (argc == 3)
	{
		rows = atoi(argv[1]);  //number of rows in the domain
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

    J = (float *)malloc( size_I * sizeof(float) );
    filepath = (char *)malloc(sizeof(char) * (strlen(folder) + 128));


	//Allocate device memory
    CUDA_CALL_SAFE(cudaMalloc((void**)& J_cuda, sizeof(float)* size_I));
    CUDA_CALL_SAFE(cudaMalloc((void**)& C_cuda, sizeof(float)* size_I));
	CUDA_CALL_SAFE(cudaMalloc((void**)& E_C, sizeof(float)* size_I));
	CUDA_CALL_SAFE(cudaMalloc((void**)& W_C, sizeof(float)* size_I));
	CUDA_CALL_SAFE(cudaMalloc((void**)& S_C, sizeof(float)* size_I));
	CUDA_CALL_SAFE(cudaMalloc((void**)& N_C, sizeof(float)* size_I));

    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/J.cudamemcpy.mem", folder);
	if ((fp = fopen(filepath, "rb")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }

    if (fread(J, sizeof(float) * size_I, 1, fp) != 1)
    {
        fprintf(stderr, "Cannot read from %s\n", filepath);
        exit(EXIT_FAILURE);
    }
	fclose(fp);	
    gettimeofday(&tv_end, NULL);
    readfile_time += time_diff(tv_start, tv_end);

	printf("Start the SRAD main loop\n");
 for (iter=0; iter< niter; iter++){     
    gettimeofday(&tv_start, NULL);
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

    gettimeofday(&tv_end, NULL);
    kernel_time += time_diff(tv_start, tv_end);
    

	//Copy data from main memory to device memory
    gettimeofday(&tv_start, NULL);
	CUDA_CALL_SAFE(cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice));
    gettimeofday(&tv_end, NULL);
    h2d_memcpy_time += time_diff(tv_start, tv_end);

    CUDA_CALL_SAFE(cudaThreadSynchronize());

	//Run kernels
    gettimeofday(&tv_start, NULL);
	srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
	srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 

    CUDA_CALL_SAFE(cudaThreadSynchronize());
    gettimeofday(&tv_end, NULL);
    kernel_time += time_diff(tv_start, tv_end);

	//Copy data from device memory to main memory
    gettimeofday(&tv_start, NULL);
    CUDA_CALL_SAFE(cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost));
    gettimeofday(&tv_end, NULL);
    d2h_memcpy_time += time_diff(tv_start, tv_end);

}

    CUDA_CALL_SAFE(cudaThreadSynchronize());

    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/J.cudamemcpy.mem", folder);
	if ((fp = fopen(filepath, "wb")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }

    if (fwrite(J, sizeof(float) * size_I, 1, fp) != 1)
    {
        fprintf(stderr, "Cannot write to %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    fflush(fp);
    fsync(fileno(fp));
	fclose(fp);	
    gettimeofday(&tv_end, NULL);
    writefile_time += time_diff(tv_start, tv_end);

	printf("Computation Done\n");

	free(J);
    free(filepath);

    CUDA_CALL_SAFE(cudaFree(C_cuda));
	CUDA_CALL_SAFE(cudaFree(J_cuda));
	CUDA_CALL_SAFE(cudaFree(E_C));
	CUDA_CALL_SAFE(cudaFree(W_C));
	CUDA_CALL_SAFE(cudaFree(N_C));
	CUDA_CALL_SAFE(cudaFree(S_C));
  
    printf("==> header: kernel_time (ms),writefile_time (ms),d2h_memcpy_time (ms),readfile_time (ms),h2d_memcpy_time (ms)\n");
    printf("==> data: %f,%f,%f,%f,%f\n", kernel_time, writefile_time, d2h_memcpy_time, readfile_time, h2d_memcpy_time);
}

