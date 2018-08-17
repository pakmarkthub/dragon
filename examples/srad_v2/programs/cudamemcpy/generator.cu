// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>
#include "srad.h"

// includes, project
#include <cuda.h>

#include "srad_kernel.cu"


void random_matrix(float *I, long rows, long cols);
void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows/cols> <outfile>\n", argv[0]);
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
    long rows, cols;
    size_t size_I;
    float *J;
    char *filepath;
	FILE *fp;

	if (argc == 3)
	{
		rows = atol(argv[1]);  //number of rows in the domain
		cols = rows;           //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0))
        {
            fprintf(stderr, "rows and cols must be multiples of 16\n");
            exit(1);
		}
        filepath = argv[2];
	}
    else
    {
        usage(argc, argv);
    }

	size_I = (size_t)cols * (size_t)rows;

	if ((fp = fopen(filepath, "w+")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }

    if (ftruncate(fileno(fp), size_I * sizeof(float)) != 0)
    {
		fprintf(stderr, "error: can not truncate %s\n", filepath);
        perror("ftruncate");
        exit(EXIT_FAILURE);
    }

    J = (float *)mmap(NULL, size_I * sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fileno(fp), 0);

	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(J, rows, cols);

    fflush(fp);
	fclose(fp);	
}


void random_matrix(float *I, long rows, long cols){
    
	srand(7);
	
	for(long i = 0 ; i < rows ; i++)
    {
		for (long j = 0 ; j < cols ; j++)
        {
            I[i * cols + j] = (float)exp(rand()/(float)RAND_MAX);
		}
	}

}

