/*********************

Hotspot Expand
by Sam Kauffman - Univeristy of Virginia
Generate larger input files for Hotspot by expanding smaller versions

*/


//#include "64_128.h"
//#include "64_256.h"
//#include "1024_2048.h"
//#include "1024_4096.h"
//#include "1024_8192.h"
//#include "1024_16384.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>

using namespace std;

void expand(char * infName, char * outfName, long in_size, long out_size)
{
	const int x = out_size / in_size;
	float val;
	fstream fs;
	float *outMatr;

    FILE *f = fopen(outfName, "w+");
    if (!f)
    {
        cerr << "Failed to open output file.\n";
        abort();
        exit(EXIT_FAILURE);
    }
    if (ftruncate(fileno(f), out_size * out_size * sizeof(float)) != 0)
    {
		fprintf(stderr, "error: can not truncate %s\n", outfName);
        perror("ftruncate");
        exit(EXIT_FAILURE);
    }
    
	// allocate 2d array of floats
    outMatr = (float *)mmap(NULL, out_size * out_size * sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fileno(f), 0);
    if (!outMatr)
    {
        fprintf(stderr, "Cannot mmap %s.\n", outfName);
        exit(EXIT_FAILURE);
    }

	// copy values into larger array
	fs.open(infName, ios::in);
	if (!fs)
    {
		cerr << "Failed to open input file.\n";
        abort();
        exit(EXIT_FAILURE);
    }
	for (long row = 0; row < in_size; row++)
    {
		for (long col = 0; col < in_size; col++)
		{
			fs >> val;
			for (long rowOff = 0; rowOff < x; rowOff++)
				for (long colOff = 0; colOff < x; colOff++)
					outMatr[(x * row + rowOff) * out_size + x * col + colOff] = val;
		}
    }
	fs.close();

    fflush(f);
    fclose(f);
}

int main(int argc, char* argv[])
{
    char *temp_in, *temp_out;
    char *power_in, *power_out;
    long in_size, out_size;
    if (argc != 7)
    {
        cout << "Usage: " << argv[0]
            << " temp_in power_in temp_out power_out in_size out_size\n";
        return 0;
    }

    temp_in = argv[1];
    power_in = argv[2];
    temp_out = argv[3];
    power_out = argv[4];
    in_size = atol(argv[5]);
    out_size = atol(argv[6]);

	expand(temp_in, temp_out, in_size, out_size);
	expand(power_in, power_out, in_size, out_size);

	cout << "Data written to files " << temp_out << " and " << power_out << ".\n";
}
