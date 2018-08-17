

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

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

////////////////////////////////////////////////////////////////////////////////

extern "C"
void bpnn_layerforward(float *l1, float *l2, float *conn, long n1, long n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, long nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, long nh, float *delta_o, long no, float *who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, long ndelta, float *ly, long nly, float *w, float *oldw);


extern "C"
int setup(int argc, char** argv);

extern "C"
float *alloc_2d_dbl(long m, long n);

extern "C"
float squash(float x);

unsigned long num_threads = 0;
unsigned long num_blocks = 0;

extern "C"
double time_diff(struct timeval tv_start, struct timeval tv_end);

struct timeval tv_start, tv_end;
extern "C" double kernel_time;       // in ms

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  long in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  num_blocks = in / 16;  
  dim3  grid(num_blocks, 1);
  dim3  threads(16 , 16);
  
  CUDA_CALL_SAFE(cudaMallocManaged(&partial_sum, num_blocks * WIDTH * sizeof(float)));
 
  
  CUDA_CALL_SAFE(cudaMallocManaged((void**) &output_hidden_cuda, (hid + 1) * sizeof(float)));
  
  
  printf("Performing GPU computation\n");
  
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);

  input_cuda = net->input_units;
  input_hidden_cuda = net->input_weights;
  hidden_partial_sum = partial_sum;
  
  gettimeofday(&tv_start, NULL);
  bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid);
 
  CUDA_CALL_SAFE(cudaThreadSynchronize());
  gettimeofday(&tv_end, NULL);
  kernel_time += time_diff(tv_start, tv_end);
  
  cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
  

  gettimeofday(&tv_start, NULL);
  for (long j = 1; j <= hid; j++) {
    sum = 0.0;
    for (long k = 0; k < num_blocks; k++) {	
      sum += partial_sum[k * hid + j-1] ;
    }
	sum += net->input_weights[j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
  gettimeofday(&tv_end, NULL);
  kernel_time += time_diff(tv_start, tv_end);


  hidden_delta_cuda = net->hidden_delta;
  input_prev_weights_cuda = net->input_prev_weights;

  gettimeofday(&tv_start, NULL);
  bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda,  
												hid, 
												input_cuda, 
												in,
												input_hidden_cuda, 
												input_prev_weights_cuda
												);

  CUDA_CALL_SAFE(cudaThreadSynchronize());
  gettimeofday(&tv_end, NULL);
  kernel_time += time_diff(tv_start, tv_end);

  CUDA_CALL_SAFE(cudaFree(output_hidden_cuda));
  CUDA_CALL_SAFE(cudaFree(hidden_partial_sum));
}
