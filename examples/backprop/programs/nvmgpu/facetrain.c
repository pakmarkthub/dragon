#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "backprop.h"
#include "omp.h"

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

extern BPNN *bpnn_internal_create(long n_in, long n_hidden, long n_out);
extern char *strcpy();
extern void exit();

long layer_size = 0;

char *folder;

double time_diff(struct timeval tv_start, struct timeval tv_end)
{
    return (double)(tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + (double)(tv_end.tv_usec - tv_start.tv_usec) / 1000.0;
}

struct timeval tv_start, tv_end;
double kernel_time = 0;       // in ms
double map_time = 0;       // in ms
double free_time = 0;       // in ms

void backprop_face()
{
    BPNN *net;
    long i;
    float out_err, hid_err;
    net = bpnn_internal_create(layer_size, 16, 1);

    char *filepath = (char *)malloc(sizeof(char) * (strlen(folder) + 128));
    if (!filepath)
    {
        fprintf(stderr, "Cannot allocate filepath\n");
        exit(EXIT_FAILURE);
    }

    printf("Input layer size : %d\n", layer_size);
    
    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/input_units.mem", folder);
    if (dragon_map(filepath, sizeof(float) * (net->input_n + 1), D_F_READ | D_F_DONTTRASH, (void **)&net->input_units) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/target.mem", folder);
    if (dragon_map(filepath, sizeof(float) * (net->output_n + 1), D_F_READ | D_F_DONTTRASH, (void **)&net->target) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/input_weights.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(float) * (net->input_n + 1) * (net->hidden_n + 1), D_F_READ | D_F_WRITE, (void **)&net->input_weights) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/hidden_weights.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(float) * (net->hidden_n + 1) * (net->output_n + 1), D_F_READ | D_F_WRITE | D_F_DONTTRASH, (void **)&net->hidden_weights) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/input_prev_weights.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(float) * (net->input_n + 1) * (net->hidden_n + 1), D_F_READ | D_F_WRITE | D_F_CREATE | D_F_VOLATILE, (void **)&net->input_prev_weights) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/hidden_prev_weights.nvmgpu.mem", folder);
    if (dragon_map(filepath, sizeof(float) * (net->hidden_n + 1) * (net->output_n + 1), D_F_READ | D_F_WRITE | D_F_VOLATILE | D_F_CREATE | D_F_DONTTRASH, (void **)&net->hidden_prev_weights) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_map %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    gettimeofday(&tv_end, NULL);
    map_time += time_diff(tv_start, tv_end);
    //
    //entering the training kernel, only one iteration
    printf("Starting training kernel\n");
    bpnn_train_cuda(net, &out_err, &hid_err);

    gettimeofday(&tv_start, NULL);
    if (dragon_unmap(net->input_units) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap input_units\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(net->target) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap target\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(net->input_weights) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap input_weights\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(net->input_prev_weights) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap input_prev_weights\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(net->hidden_weights) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap hidden_weights\n");
        exit(EXIT_FAILURE);
    }

    if (dragon_unmap(net->hidden_prev_weights) != D_OK)
    {
        fprintf(stderr, "Cannot dragon_unmap hidden_weights\n");
        exit(EXIT_FAILURE);
    }
    gettimeofday(&tv_end, NULL);
    free_time += time_diff(tv_start, tv_end);

    bpnn_free(net);
    free(filepath);


    printf("Training done\n");

    printf("==> header: kernel_time (ms),map_time (ms),free_time (ms)\n");
    printf("==> data: %f,%f,%f\n", kernel_time, map_time, free_time);
}

int setup(int argc, char *argv[])
{
  long seed;

  if (argc != 3)
  {
      fprintf(stderr, "usage: %s <num of input elements> <folder>\n", argv[0]);
      exit(0);
  }
  layer_size = atol(argv[1]);
  folder = argv[2];
  if (layer_size % 16 != 0)
  {
      fprintf(stderr, "The number of input points must be divided by 16\n");
      exit(0);
  }
  
  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
