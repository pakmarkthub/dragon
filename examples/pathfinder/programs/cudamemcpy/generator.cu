#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define HALO 1 // halo width along one direction when advancing to the next iteration

#define BENCH_PRINT

void run(int argc, char** argv);

long rows, cols;
int* data;
#define M_SEED 9

void init(int argc, char** argv)
{
    char *folder;
    char *filepath;

	if (argc == 4)
    {
		cols = atol(argv[1]);
		rows = atol(argv[2]);
        folder = argv[3];
	}
    else
    {
        printf("Usage: %s row_len col_len folder\n", argv[0]);
        exit(0);
    }

    filepath = (char *)malloc(sizeof(char) * (strlen(folder) + 128));
    if (!filepath)
    {
        fprintf(stderr, "Cannot allocate filepath");
        exit(EXIT_FAILURE);
    }

    sprintf(filepath, "%s/data.mem", folder);

	FILE *fp;

	if ((fp = fopen(filepath, "w+")) == 0)
    {
        fprintf(stderr, "%s was not opened\n", filepath);
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fileno(fp), sizeof(int) * rows * cols) != 0)
    {
		fprintf(stderr, "error: can not truncate %s\n", filepath);
        perror("ftruncate");
        exit(EXIT_FAILURE);
    }
    data = (int *)mmap(NULL, sizeof(int) * rows * cols, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fileno(fp), 0);

    if (!data)
    {
        fprintf(stderr, "Cannot mmap %s.\n", filepath);
        exit(EXIT_FAILURE);
    }

	int seed = M_SEED;
	srand(seed);

	for (long i = 0; i < rows; i++)
    {
        for (long j = 0; j < cols; j++)
        {
            data[i * cols + j] = rand() % 10;
        }
    }

    fflush(fp);
	fclose(fp);	

    free(filepath);
}

int main(int argc, char** argv)
{
    init(argc, argv);

    return EXIT_SUCCESS;
}

