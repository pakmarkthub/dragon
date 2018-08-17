#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "backprop.h"
#include "omp.h"

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

    int fd_input_units, fd_target; 
    int fd_input_weights, fd_hidden_weights;

    char *filepath = (char *)malloc(sizeof(char) * (strlen(folder) + 128));
    if (!filepath)
    {
        fprintf(stderr, "Cannot allocate filepath\n");
        exit(EXIT_FAILURE);
    }

    net = bpnn_internal_create(layer_size, 16, 1);

    printf("Input layer size : %d\n", layer_size);
    
    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/input_units.mem", folder);
    if ((fd_input_units = open(filepath, O_LARGEFILE | O_RDWR)) < 0)
    {
        fprintf(stderr, "Cannot open file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((net->input_units = (float *)mmap(NULL, sizeof(float) * (net->input_n + 1), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_input_units, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(net->input_units, sizeof(float) * (net->input_n + 1), cudaHostRegisterDefault));

    sprintf(filepath, "%s/target.mem", folder);
    if ((fd_target = open(filepath, O_LARGEFILE | O_RDWR)) < 0)
    {
        fprintf(stderr, "Cannot open file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((net->target = (float *)mmap(NULL, sizeof(float) * (net->output_n + 1), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_target, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(net->target, sizeof(float) * (net->output_n + 1), cudaHostRegisterDefault));

    sprintf(filepath, "%s/input_weights.hostreg.mem", folder);
    if ((fd_input_weights = open(filepath, O_LARGEFILE | O_RDWR)) < 0)
    {
        fprintf(stderr, "Cannot open file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((net->input_weights = (float *)mmap(NULL, sizeof(float) * (net->input_n + 1) * (net->hidden_n + 1), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_input_weights, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(net->input_weights, sizeof(float) * (net->input_n + 1) * (net->hidden_n + 1), cudaHostRegisterDefault));

    sprintf(filepath, "%s/hidden_weights.hostreg.mem", folder);
    if ((fd_hidden_weights = open(filepath, O_LARGEFILE | O_RDWR)) < 0)
    {
        fprintf(stderr, "Cannot open file %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    if ((net->hidden_weights = (float *)mmap(NULL, sizeof(float) * (net->hidden_n + 1) * (net->output_n + 1), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_hidden_weights, 0)) == MAP_FAILED)
    {
        fprintf(stderr, "Cannot mmap %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    CUDA_CALL_SAFE(cudaHostRegister(net->hidden_weights, sizeof(float) * (net->hidden_n + 1) * (net->output_n + 1), cudaHostRegisterDefault));
    gettimeofday(&tv_end, NULL);
    map_time += time_diff(tv_start, tv_end);
    //
    //entering the training kernel, only one iteration
    printf("Starting training kernel\n");
    bpnn_train_cuda(net, &out_err, &hid_err);

    gettimeofday(&tv_start, NULL);
    CUDA_CALL_SAFE(cudaHostUnregister(net->input_units));
    if (msync(net->input_units, sizeof(float) * (net->input_n + 1), MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync input_units\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(net->input_units, sizeof(float) * (net->input_n + 1)) != 0)
    {
        fprintf(stderr, "Cannot munmap input_units\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_input_units);
    close(fd_input_units);

    CUDA_CALL_SAFE(cudaHostUnregister(net->target));
    if (msync(net->target, sizeof(float) * (net->output_n + 1), MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync target\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(net->target, sizeof(float) * (net->output_n + 1)) != 0)
    {
        fprintf(stderr, "Cannot munmap target\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_target);
    close(fd_target);

    CUDA_CALL_SAFE(cudaHostUnregister(net->input_weights));
    if (msync(net->input_weights, sizeof(float) * (net->input_n + 1) * (net->hidden_n + 1), MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync input_weights\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(net->input_weights, sizeof(float) * (net->input_n + 1) * (net->hidden_n + 1)) != 0)
    {
        fprintf(stderr, "Cannot munmap input_weights\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_input_weights);
    close(fd_input_weights);

    CUDA_CALL_SAFE(cudaHostUnregister(net->hidden_weights));
    if (msync(net->hidden_weights, sizeof(float) * (net->hidden_n + 1) * (net->output_n + 1), MS_SYNC) != 0)
    {
        fprintf(stderr, "Cannot msync hidden_weights\n");
        perror("msync");
        exit(EXIT_FAILURE);
    }
    if (munmap(net->hidden_weights, sizeof(float) * (net->hidden_n + 1) * (net->output_n + 1)) != 0)
    {
        fprintf(stderr, "Cannot munmap hidden_weights\n");
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    fsync(fd_hidden_weights);
    close(fd_hidden_weights);
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
