// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
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
    int fd_J, fd_C, fd_E, fd_W, fd_S, fd_N;

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

	//Allocate device memory
    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/J.hostreg.mem", folder);
	if ((fd_J = open(filepath, O_LARGEFILE | O_RDWR)) < 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((J = (float *)mmap(NULL, sizeof(float) * size_I, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_J, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(J, sizeof(float) * size_I, cudaHostRegisterDefault));

    sprintf(filepath, "%s/C_cuda.hostreg.mem", folder);
	if ((fd_C = open(filepath, O_LARGEFILE | O_RDWR | O_CREAT)) < 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fd_C, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot truncate file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((C_cuda = (float *)mmap(NULL, sizeof(float) * size_I, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_C, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(C_cuda, sizeof(float) * size_I, cudaHostRegisterDefault));

    sprintf(filepath, "%s/E_C.hostreg.mem", folder);
	if ((fd_E = open(filepath, O_LARGEFILE | O_RDWR | O_CREAT)) < 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fd_E, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot truncate file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((E_C = (float *)mmap(NULL, sizeof(float) * size_I, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_E, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(E_C, sizeof(float) * size_I, cudaHostRegisterDefault));

    sprintf(filepath, "%s/W_C.hostreg.mem", folder);
	if ((fd_W = open(filepath, O_LARGEFILE | O_RDWR | O_CREAT)) < 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fd_W, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot truncate file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((W_C = (float *)mmap(NULL, sizeof(float) * size_I, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_W, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(W_C, sizeof(float) * size_I, cudaHostRegisterDefault));
    
    sprintf(filepath, "%s/S_C.hostreg.mem", folder);
	if ((fd_S = open(filepath, O_LARGEFILE | O_RDWR | O_CREAT)) < 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fd_S, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot truncate file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((S_C = (float *)mmap(NULL, sizeof(float) * size_I, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_S, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(S_C, sizeof(float) * size_I, cudaHostRegisterDefault));

    sprintf(filepath, "%s/N_C.hostreg.mem", folder);
	if ((fd_N = open(filepath, O_LARGEFILE | O_RDWR | O_CREAT)) < 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fd_N, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot truncate file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((N_C = (float *)mmap(NULL, sizeof(float) * size_I, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_N, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(N_C, sizeof(float) * size_I, cudaHostRegisterDefault));

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

    gettimeofday(&tv_start, NULL);
    CUDA_CALL_SAFE(cudaHostUnregister(J));
    if (msync(J, sizeof(float) * size_I, MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync J\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(J, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot munmap J\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_J);
    close(fd_J);

    CUDA_CALL_SAFE(cudaHostUnregister(C_cuda));
    if (msync(C_cuda, sizeof(float) * size_I, MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync C_cuda\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(C_cuda, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot munmap C_cuda\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_C);
    close(fd_C);

    CUDA_CALL_SAFE(cudaHostUnregister(E_C));
    if (msync(E_C, sizeof(float) * size_I, MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync E_C\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(E_C, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot munmap E_C\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_E);
    close(fd_E);

    CUDA_CALL_SAFE(cudaHostUnregister(W_C));
    if (msync(W_C, sizeof(float) * size_I, MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync W_C\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(W_C, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot munmap W_C\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_W);
    close(fd_W);

    CUDA_CALL_SAFE(cudaHostUnregister(S_C));
    if (msync(S_C, sizeof(float) * size_I, MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync S_C\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(S_C, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot munmap S_C\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_S);
    close(fd_S);

    CUDA_CALL_SAFE(cudaHostUnregister(N_C));
    if (msync(N_C, sizeof(float) * size_I, MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync N_C\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(N_C, sizeof(float) * size_I) != 0)
    {
        fprintf(stderr, "Cannot munmap N_C\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_N);
    close(fd_N);
    gettimeofday(&tv_end, NULL);
    free_time += time_diff(tv_start, tv_end);

	printf("Computation Done\n");

    free(filepath);

    printf("==> header: kernel_time (ms),map_time (ms),free_time (ms)\n");
    printf("==> data: %f,%f,%f\n", kernel_time, map_time, free_time);
}

