#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>

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
            return EXIT_FAILURE; \
        } \
    } while (0)        

double time_diff(struct timeval tv_start, struct timeval tv_stop)
{
    return (double)(tv_stop.tv_sec - tv_start.tv_sec) * 1000.0 + (double)(tv_stop.tv_usec - tv_start.tv_usec) / 1000.0;
}

__device__ uint32_t d_result;

__global__ void kernel(uint32_t* g_buf, int seed) 
{
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;

    g_buf[idx] = idx * seed;
}

int main(int argc, char *argv[])
{
    uint32_t *g_buf;
    size_t num_tblocks;          
    size_t num_threads;          
    int size_order;
    size_t total_size;
    cudaEvent_t start_event, stop_event;
    float kernel_time = 0;          // in ms
    double free_time = 0;          // in ms
    double map_time = 0;          // in ms
    int seed;

    struct timeval tv_start, tv_stop;

    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s file size_in_GiB threads_per_block seed\n", argv[0]);
        return EXIT_SUCCESS;
    }

    size_order = atoi(argv[2]);
    num_threads = atoi(argv[3]);
    seed = atoi(argv[4]);
    
    //total_size = ((size_t)1 << 30) * size_order;
    total_size = (size_t)1000000000 * (size_t)size_order;
    num_tblocks = total_size / sizeof(uint32_t) / num_threads;
    
    CUDA_CALL_SAFE(cudaEventCreate(&start_event));
    CUDA_CALL_SAFE(cudaEventCreate(&stop_event));

    gettimeofday(&tv_start, NULL);
    if (dragon_map(argv[1], total_size, D_F_WRITE | D_F_CREATE, (void **)(&g_buf)) != D_OK)
        return EXIT_FAILURE;
    gettimeofday(&tv_stop, NULL);

    map_time = time_diff(tv_start, tv_stop);

    CUDA_CALL_SAFE(cudaEventRecord(start_event));
    kernel<<< num_tblocks, num_threads >>>(g_buf, seed);
    CUDA_CALL_SAFE(cudaEventRecord(stop_event));

    CUDA_CALL_SAFE(cudaEventSynchronize(stop_event));
    CUDA_CALL_SAFE(cudaEventElapsedTime(&kernel_time, start_event, stop_event));

    CUDA_CALL_SAFE(cudaDeviceSynchronize());

    gettimeofday(&tv_start, NULL);
    if (dragon_unmap(g_buf) != D_OK)
        return EXIT_FAILURE;
    gettimeofday(&tv_stop, NULL);

    free_time = time_diff(tv_start, tv_stop);

    printf("==> header: kernel_time (ms),free_time (ms),map_time (ms)\n");
    printf("==> data: %f,%f,%f\n", kernel_time, free_time, map_time);

    return EXIT_SUCCESS;
}

