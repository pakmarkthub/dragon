#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "backprop.h"
#include "omp.h"

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
double writefile_time = 0;       // in ms
double readfile_time = 0;       // in ms

void backprop_face()
{
    BPNN *net;
    long i;
    float out_err, hid_err;
    net = bpnn_internal_create(layer_size, 16, 1);

	FILE *fp;
    char *filepath = (char *)malloc(sizeof(char) * (strlen(folder) + 128));
    if (!filepath)
    {
        fprintf(stderr, "Cannot allocate filepath\n");
        exit(EXIT_FAILURE);
    }

    printf("Input layer size : %d\n", layer_size);
    
    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/input_units.mem", folder);
	if ((fp = fopen(filepath, "rb")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (fread(net->input_units, sizeof(float) * (net->input_n + 1), 1, fp) != 1)
    {
        fprintf(stderr, "Cannot read from %s\n", filepath);
        exit(EXIT_FAILURE);
    }
	fclose(fp);	

    sprintf(filepath, "%s/target.mem", folder);
	if ((fp = fopen(filepath, "rb")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (fread(net->target, sizeof(float) * (net->output_n + 1), 1, fp) != 1)
    {
        fprintf(stderr, "Cannot read from %s\n", filepath);
        exit(EXIT_FAILURE);
    }
	fclose(fp);	

    sprintf(filepath, "%s/input_weights.uvm.mem", folder);
	if ((fp = fopen(filepath, "rb")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (fread(net->input_weights, sizeof(float) * (net->input_n + 1) * (net->hidden_n + 1), 1, fp) != 1)
    {
        fprintf(stderr, "Cannot read from %s\n", filepath);
        exit(EXIT_FAILURE);
    }
	fclose(fp);	

    sprintf(filepath, "%s/hidden_weights.uvm.mem", folder);
	if ((fp = fopen(filepath, "rb")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (fread(net->hidden_weights, sizeof(float) * (net->hidden_n + 1) * (net->output_n + 1), 1, fp) != 1)
    {
        fprintf(stderr, "Cannot read from %s\n", filepath);
        exit(EXIT_FAILURE);
    }
	fclose(fp);	
    gettimeofday(&tv_end, NULL);
    readfile_time += time_diff(tv_start, tv_end);

    //entering the training kernel, only one iteration
    printf("Starting training kernel\n");
    bpnn_train_cuda(net, &out_err, &hid_err);

    gettimeofday(&tv_start, NULL);
    sprintf(filepath, "%s/input_weights.uvm.mem", folder);
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
    fflush(fp);
    fsync(fileno(fp));
	fclose(fp);	

    sprintf(filepath, "%s/hidden_weights.uvm.mem", folder);
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
    fflush(fp);
    fsync(fileno(fp));
	fclose(fp);	
    gettimeofday(&tv_end, NULL);
    writefile_time += time_diff(tv_start, tv_end);

    bpnn_free(net);
    free(filepath);

    printf("Training done\n");

    printf("==> header: kernel_time (ms),writefile_time (ms),readfile_time (ms)\n");
    printf("==> data: %f,%f,%f\n", kernel_time, writefile_time, readfile_time);
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
