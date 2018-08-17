#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>

#include <dragon.h>

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define HALO 1 // halo width along one direction when advancing to the next iteration

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

void run(int argc, char** argv);

long rows, cols;
int *data;
int *result;
int *result_tmp;
long pyramid_height = 1;

char *folder;
char *filepath;

struct timeval tv_start, tv_end;
double kernel_time = 0;       // in ms
double map_time = 0;       // in ms
double free_time = 0;       // in ms

void init(int argc, char** argv)
{
	if (argc == 3)
    {
		cols = atoi(argv[1]);
        rows = cols;
        folder = argv[2];
	}
    else
    {
        printf("Usage: %s <row/col> folder\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    filepath = (char *)malloc(sizeof(char) * (strlen(folder) + 128));
    if (!filepath)
    {
        fprintf(stderr, "Cannot allocate filepath");
        exit(EXIT_FAILURE);
    }

    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/data.mem", folder);
    if (dragon_map(filepath, sizeof(int) * rows * cols, D_F_READ, (void **)&data) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/result.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(int) * cols, D_F_WRITE | D_F_CREATE, (void **)&result) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    gettimeofday(&tv_end, NULL);
    map_time += time_diff(tv_start, tv_end);

    CUDA_CALL_SAFE(cudaMallocManaged(&result_tmp, sizeof(int) * cols));
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void dynproc_kernel(
                long iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                long cols, 
                long rows,
                long startStep,
                long border)
{

        __shared__ int prev[BLOCK_SIZE];
        __shared__ int result[BLOCK_SIZE];

	long bx = (long)blockIdx.x;
	long tx = (long)threadIdx.x;
	
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	long small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        long blkX = small_block_cols*bx-border;
        long blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	long xidx = blkX+tx;
       
        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        long validXmin = (blkX < 0) ? -blkX : 0;
        long validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        long W = tx-1;
        long E = tx+1;
        
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (long i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  long left = prev[W];
                  long up = prev[tx];
                  long right = prev[E];
                  long shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  long index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];
	
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
	    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];		
      }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], long rows, long cols, \
	 long pyramid_height, long blockCols, long borderCols)
{
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(blockCols);  
	
        int src = 1, dst = 0;
	for (long t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows-t-1), 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);
            CUDA_CALL_SAFE(cudaThreadSynchronize());
	}
        return dst;
}

int main(int argc, char** argv)
{
    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    long borderCols = (pyramid_height)*HALO;
    long smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    long blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
	
    int *gpuWall, *gpuResult[2];

    if (rows % 2 == 0)
    {
        gpuResult[0] = result_tmp;
        gpuResult[1] = result;
    }
    else
    {
        gpuResult[0] = result;
        gpuResult[1] = result_tmp;
    }

    gpuWall = data + cols;

    memcpy(gpuResult[0], data, sizeof(int) * cols);

    gettimeofday(&tv_start, NULL);
    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);
    gettimeofday(&tv_end, NULL);
    kernel_time += time_diff(tv_start, tv_end);

    gettimeofday(&tv_start, NULL);
    if (dragon_unmap(data) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap data\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(result) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap result\n");
        exit(EXIT_FAILURE);
    }
    gettimeofday(&tv_end, NULL);
    free_time += time_diff(tv_start, tv_end);


    CUDA_CALL_SAFE(cudaFree(result_tmp));
    free(filepath);

    printf("==> header: kernel_time (ms),map_time (ms),free_time (ms)\n");
    printf("==> data: %f,%f,%f\n", kernel_time, map_time, free_time);
}

