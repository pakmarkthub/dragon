/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"
#include <math.h>
#include <string.h>
#include <cuda.h>
//#define OPEN

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register long _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

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

/*** Return random number between 0.0 and 1.0 ***/
float drnd()
{
  return ((float) rand() / (float) BIGRND);
}

/*** Return random number between -1.0 and 1.0 ***/
float dpn1()
{
  return ((drnd() * 2.0) - 1.0);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/

float squash(float x)
{
  //x = -x;
  //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  //return(1.0 / (1.0 + m));
  return (1.0 / (1.0 + exp(-x)));
}


/*** Allocate 1d array of floats ***/

float *alloc_1d_dbl(long n)
{
  float *ptr;

  CUDA_CALL_SAFE(cudaMallocManaged(&ptr, n * sizeof(float)));
  return ptr;
}


/*** Allocate 2d array of floats ***/

float *alloc_2d_dbl(long m, long n)
{
  float *ptr;

  CUDA_CALL_SAFE(cudaMallocManaged(&ptr, m * n * sizeof(float)));
  return ptr;
}


void bpnn_randomize_weights(float *w, long m, long n)
{
  long i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
     w[i * (n + 1) + j] = (float) rand()/RAND_MAX;
    //  w[i][j] = dpn1();
    }
  }
}

void bpnn_randomize_row(float *w, long m)
{
	long i;
	for (i = 0; i <= m; i++) {
     //w[i] = (float) rand()/RAND_MAX;
	 w[i] = 0.1;
    }
}

extern "C"
void bpnn_zero_weights(float *w, long m, long n)
{
    memset(w, 0, sizeof(float) * (m + 1) * (n + 1));
}

extern "C"
void bpnn_initialize(long seed)
{
    printf("Random number generator seed: %d\n", seed);
    srand(seed);
}

extern "C"
BPNN *bpnn_internal_create(long n_in, long n_hidden, long n_out)
{
    BPNN *newnet;

    newnet = (BPNN *)malloc(sizeof(BPNN));
    if (newnet == NULL) 
    {
        printf("BPNN_CREATE: Couldn't allocate neural network\n");
        return (NULL);
    }

    newnet->input_n = n_in;
    newnet->hidden_n = n_hidden;
    newnet->output_n = n_out;
    //newnet->input_units = alloc_1d_dbl(n_in + 1);
    newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
    newnet->output_units = alloc_1d_dbl(n_out + 1);

    newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
    newnet->output_delta = alloc_1d_dbl(n_out + 1);
    //newnet->target = alloc_1d_dbl(n_out + 1);

    //newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
    //newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

    //newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
    //newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

    return (newnet);
}

extern "C"
void bpnn_free(BPNN *net)
{
  //CUDA_CALL_SAFE(cudaFree((char *) net->input_units));
  CUDA_CALL_SAFE(cudaFree((char *) net->hidden_units));
  CUDA_CALL_SAFE(cudaFree((char *) net->output_units));

  CUDA_CALL_SAFE(cudaFree((char *) net->hidden_delta));
  CUDA_CALL_SAFE(cudaFree((char *) net->output_delta));
  //CUDA_CALL_SAFE(cudaFree((char *) net->target));

  //CUDA_CALL_SAFE(cudaFree((char *) net->input_weights));
  //CUDA_CALL_SAFE(cudaFree((char *) net->input_prev_weights));

  //CUDA_CALL_SAFE(cudaFree((char *) net->hidden_weights));
  //CUDA_CALL_SAFE(cudaFree((char *) net->hidden_prev_weights));

  free((char *) net);
}


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

BPNN *bpnn_create(long n_in, long n_hidden, long n_out)
{
    BPNN *newnet;

    newnet = bpnn_internal_create(n_in, n_hidden, n_out);

    bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
    bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
    bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
    bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
    bpnn_randomize_row(newnet->target, n_out);
    return (newnet);
}

extern "C"
void bpnn_layerforward(float *l1, float *l2, float *conn, long n1, long n2)
{
  float sum;
  long j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for shared(conn, n1, n2, l1) private(k, j) reduction(+: sum) schedule(static)
#endif 
  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {

    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k <= n1; k++) {	
      sum += conn[k * (n2 + 1) + j] * l1[k]; 
    }
    l2[j] = squash(sum);
  }
}

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, long nj, float *err)  
{
  long j;
  float o, t, errsum;
  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}

extern "C"
void bpnn_hidden_error(float *delta_h,   
					   long nh, 
					   float *delta_o, 
					   long no, 
					   float *who, 
					   float *hidden, 
					   float *err)
{
  long j, k;
  float h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j * (no + 1) + k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}

extern "C"
void bpnn_adjust_weights(float *delta, long ndelta, float *ly, long nly, float *w, float *oldw)
{
  float new_dw;
  long k, j;
  ly[0] = 1.0;
  //eta = 0.3;
  //momentum = 0.3;

#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for  \
      shared(oldw, w, delta) \
	  private(j, k, new_dw) \
	  firstprivate(ndelta, nly, momentum) 
#endif 
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k * (ndelta + 1) + j]));
	  w[k * (ndelta + 1) + j] += new_dw;
	  oldw[k * (ndelta + 1) + j] = new_dw;
    }
  }
}

extern "C"
void bpnn_feedforward(BPNN *net)
{
  long in, hid, out;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

}

extern "C"
void bpnn_train(BPNN *net, float *eo, float *eh)
{
  long in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

  /*** Compute error on output and hidden units. ***/
  bpnn_output_error(net->output_delta, net->target, net->output_units,
      out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
      net->hidden_weights, net->hidden_units, &hid_err);
  *eo = out_err;
  *eh = hid_err;

  /*** Adjust input and hidden weights. ***/
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
      net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
      net->input_weights, net->input_prev_weights);

}


