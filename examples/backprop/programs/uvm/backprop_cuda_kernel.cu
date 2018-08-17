

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"


__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
	                   float *output_hidden_cuda,
					   float *input_hidden_cuda,
					   float *hidden_partial_sum,
					   long in,
					   long hid) 
{
   long by = (long)blockIdx.x;
   long tx = (long)threadIdx.x;
   long ty = (long)threadIdx.y;

   long index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   long index_in = HEIGHT * by + ty + 1;
   
   __shared__ float input_node[HEIGHT];
   __shared__ float weight_matrix[HEIGHT][WIDTH];


   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in] ;
   
   __syncthreads();

   weight_matrix[ty][tx] = input_hidden_cuda[index];

   __syncthreads();
   
   weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

   __syncthreads();   
   
   for ( long i = 1 ; i <= __log2f(HEIGHT) ; i++){
 
	   long power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

	   __syncthreads();

   }
   
   //__syncthreads();

   //input_hidden_cuda[index] = weight_matrix[ty][tx];
   
/*
   for ( unsigned long i = 2 ; i <= HEIGHT ; i *= 2){
 
	   unsigned long power_two = i - 1;

	   if( (ty & power_two) == 0 ) {
		weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
	   }

   }
   */

   __syncthreads();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
   }

}


__global__ void bpnn_adjust_weights_cuda(float * delta,   
										 long hid,         
										 float * ly,      
										 long in,          
										 float * w,       
										 float * oldw)  									
{
  
  
   long by = (long)blockIdx.x;

   long tx = (long)threadIdx.x;
   long ty = (long)threadIdx.y;
	
   long index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   long index_y = HEIGHT * by + ty + 1;
   long index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));


   __syncthreads();

   if (ty == 0 && by ==0){
   w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }


}
#endif 
