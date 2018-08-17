#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "backprop.h"
#include "omp.h"

extern char *strcpy();
extern void exit();
extern BPNN *bpnn_internal_create(long n_in, long n_hidden, long n_out);

long layer_size = 0;

char *folder;

double kernel_time = 0;       // in ms
double d2h_memcpy_time = 0;       // in ms
double h2d_memcpy_time = 0;       // in ms

double time_diff(struct timeval tv_start, struct timeval tv_end)
{
    return (double)(tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + (double)(tv_end.tv_usec - tv_start.tv_usec) / 1000.0;
}

void backprop_face()
{
    BPNN *net;
    long i;
    float out_err, hid_err;
    long n_in = layer_size;
    long n_hidden = 16;
    long n_out = 1;

	FILE *fp;
    char *filepath = (char *)malloc(sizeof(char) * (strlen(folder) + 128));
    if (!filepath)
    {
        fprintf(stderr, "Cannot allocate filepath");
        exit(EXIT_FAILURE);
    }

    net = bpnn_internal_create(n_in, n_hidden, n_out);

    bpnn_randomize_weights(net->input_weights, n_in, n_hidden);
    bpnn_randomize_weights(net->hidden_weights, n_hidden, n_out);
    bpnn_randomize_row(net->target, n_out);

    printf("Input layer size : %d\n", layer_size);
    load(net);
    
    sprintf(filepath, "%s/input_units.mem", folder);
	if ((fp = fopen(filepath, "wb")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (fwrite(net->input_units, sizeof(float) * (net->input_n + 1), 1, fp) != 1)
    {
        fprintf(stderr, "Cannot write to %s\n", filepath);
        exit(EXIT_FAILURE);
    }
	fclose(fp);	

    sprintf(filepath, "%s/target.mem", folder);
	if ((fp = fopen(filepath, "wb")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (fwrite(net->target, sizeof(float) * (net->output_n + 1), 1, fp) != 1)
    {
        fprintf(stderr, "Cannot write to %s\n", filepath);
        exit(EXIT_FAILURE);
    }
	fclose(fp);	

    sprintf(filepath, "%s/input_weights.mem", folder);
	if ((fp = fopen(filepath, "wb")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (fwrite(net->input_weights, sizeof(float) * (net->input_n + 1) * (net->hidden_n + 1), 1, fp) != 1)
    {
        fprintf(stderr, "Cannot write to %s\n", filepath);
        exit(EXIT_FAILURE);
    }
	fclose(fp);	

    sprintf(filepath, "%s/hidden_weights.mem", folder);
	if ((fp = fopen(filepath, "wb")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (fwrite(net->hidden_weights, sizeof(float) * (net->hidden_n + 1) * (net->output_n + 1), 1, fp) != 1)
    {
        fprintf(stderr, "Cannot write to %s\n", filepath);
        exit(EXIT_FAILURE);
    }
	fclose(fp);	

    free(filepath);
}

int setup(int argc, char *argv[])
{
  long seed;

  if (argc != 3)
  {
      fprintf(stderr, "usage: %s <num of input elements> <folder>\n", argv[0]);
      exit(EXIT_FAILURE);
  }
  layer_size = atol(argv[1]);
  folder = argv[2];
  if (layer_size % 16 != 0)
  {
      fprintf(stderr, "The number of input points must be divided by 16\n");
      exit(EXIT_FAILURE);
  }
  
  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
