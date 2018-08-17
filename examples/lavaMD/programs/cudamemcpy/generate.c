//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	UPDATE
//========================================================================================================================================================================================================200

//	14 APR 2011 Lukasz G. Szafaryn

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>					// (in path known to compiler)			needed by printf
#include <stdlib.h>					// (in path known to compiler)			needed by malloc
#include <stdbool.h>				// (in path known to compiler)			needed by true/false
#include <string.h>

#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"						// (in the current directory)

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

int 
main(	int argc, 
		char *argv [])
{

	printf("thread block size of kernel = %d \n", NUMBER_THREADS);
	//======================================================================================================================================================150
	//	CPU/MCPU VARIABLES
	//======================================================================================================================================================150

	// counters
	long i, j, k, l, m, n;

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR* rv_cpu;
	fp* qv_cpu;
	long nh;

    FILE *f;
    char *folder;
    char *filepath;

	//======================================================================================================================================================150
	//	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================150

    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <boxes1d> <folder>\n", argv[0]);
        abort();
        exit(EXIT_FAILURE);
    }

    dim_cpu.cur_arg = 1;
	dim_cpu.boxes1d_arg = atoi(argv[1]);

    if (dim_cpu.boxes1d_arg < 0)
    {
        fprintf(stderr, "ERROR: Wrong value to -boxes1d parameter, cannot be <=0\n");
        abort();
        exit(EXIT_FAILURE);
    }

    folder = argv[2];

    filepath = malloc(sizeof(char) * (strlen(folder) + 128));

	// Print configuration
	printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150

	par_cpu.alpha = 0.5;

	//======================================================================================================================================================150
	//	DIMENSIONS
	//======================================================================================================================================================150

	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

    //fprintf(stderr, "%ld %ld %ld %ld\n", dim_cpu.box_mem, dim_cpu.space_mem, dim_cpu.space_mem2, dim_cpu.box_mem + dim_cpu.space_mem * 2 + dim_cpu.space_mem2);

	//======================================================================================================================================================150
	//	SYSTEM MEMORY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	BOX
	//====================================================================================================100

	// allocate boxes
    sprintf(filepath, "%s/box.mem", folder);
    if ((f = fopen(filepath, "w+")) == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fileno(f), dim_cpu.box_mem) != 0)
    {
		fprintf(stderr, "error: can not truncate %s %ld\n", filepath, dim_cpu.box_mem);
        perror("ftruncate");
        exit(EXIT_FAILURE);
    }

	box_cpu = (box_str*)mmap(NULL, dim_cpu.box_mem, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fileno(f), 0);
    if (box_cpu == NULL)
    {
        fprintf(stderr, "ERROR: Cannot mmap box_cpu\n");
        exit(EXIT_FAILURE);
    }

	// initialize number of home boxes
	nh = 0;

	// home boxes in z direction
	for(i=0; i<dim_cpu.boxes1d_arg; i++){
		// home boxes in y direction
		for(j=0; j<dim_cpu.boxes1d_arg; j++){
			// home boxes in x direction
			for(k=0; k<dim_cpu.boxes1d_arg; k++){

				// current home box
				box_cpu[nh].x = k;
				box_cpu[nh].y = j;
				box_cpu[nh].z = i;
				box_cpu[nh].number = nh;
				box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

				// initialize number of neighbor boxes
				box_cpu[nh].nn = 0;

				// neighbor boxes in z direction
				for(l=-1; l<2; l++){
					// neighbor boxes in y direction
					for(m=-1; m<2; m++){
						// neighbor boxes in x direction
						for(n=-1; n<2; n++){

							// check if (this neighbor exists) and (it is not the same as home box)
							if(		(((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)	&&
									(l==0 && m==0 && n==0)==false	){

								// current neighbor box
								box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
								box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
								box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
								box_cpu[nh].nei[box_cpu[nh].nn].number =	(box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + 
																			(box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + 
																			 box_cpu[nh].nei[box_cpu[nh].nn].x;
								box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

								// increment neighbor box
								box_cpu[nh].nn = box_cpu[nh].nn + 1;

							}

						} // neighbor boxes in x direction
					} // neighbor boxes in y direction
				} // neighbor boxes in z direction

				// increment home box
				nh = nh + 1;

			} // home boxes in x direction
		} // home boxes in y direction
	} // home boxes in z direction

    fflush(f);
    fclose(f);

	//====================================================================================================100
	//	PARAMETERS, DISTANCE, CHARGE AND FORCE
	//====================================================================================================100

	// random generator seed set to random value - time in this case
	srand(time(NULL));

	// input (distances)
    sprintf(filepath, "%s/rv.mem", folder);
    if ((f = fopen(filepath, "w+")) == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fileno(f), dim_cpu.space_mem) != 0)
    {
		fprintf(stderr, "error: can not truncate %s\n", filepath);
        perror("ftruncate");
        exit(EXIT_FAILURE);
    }

	rv_cpu = (FOUR_VECTOR*)mmap(NULL, dim_cpu.space_mem, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fileno(f), 0);
    if (rv_cpu == NULL)
    {
        fprintf(stderr, "ERROR: Cannot mmap rv_cpu\n");
        exit(EXIT_FAILURE);
    }
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		rv_cpu[i].v = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].x = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].y = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].z = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
	}
    fflush(f);
    fclose(f);

	// input (charge)
    sprintf(filepath, "%s/qv.mem", folder);
    if ((f = fopen(filepath, "w+")) == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", filepath);
        abort();
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fileno(f), dim_cpu.space_mem2) != 0)
    {
		fprintf(stderr, "error: can not truncate %s\n", filepath);
        perror("ftruncate");
        exit(EXIT_FAILURE);
    }

	qv_cpu = (fp*)mmap(NULL, dim_cpu.space_mem2, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fileno(f), 0);
    if (qv_cpu == NULL)
    {
        fprintf(stderr, "ERROR: Cannot mmap qv_cpu\n");
        exit(EXIT_FAILURE);
    }
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		qv_cpu[i] = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
	}
    fflush(f);
    fclose(f);

    free(filepath);

	//======================================================================================================================================================150
	//	RETURN
	//======================================================================================================================================================150

	return 0.0;																					// always returns 0.0

}
