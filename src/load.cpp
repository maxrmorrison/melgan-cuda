#include <stdio.h>

#include "load.hpp"


/* Load from 32-bit binary float file into two-dimensional array */
float *load(const std::string file, const unsigned int size)
{
    // Allocate cpu memory
    float *data = (float *) malloc(size * sizeof(float));

    // Open file
    FILE *array_file = fopen(file.c_str(), "rb");
    if (array_file == NULL) {
        fprintf(stderr, "Cannot open %s\n", file.c_str());
        exit(1);
    }

    // Read data into memory
    fread(data, sizeof(float), size, array_file);

    // Close file
    fclose(array_file);

    // User frees
    return data;
}
