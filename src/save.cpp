#include <stdio.h>

#include "save.hpp"


/* Save 32-bit float array to file */
void save(const std::string file, const float * const data, const uint size)
{
    // Open file
    FILE *array_file = fopen(file.c_str(), "wb");
    if (array_file == NULL) {
        fprintf(stderr, "Cannot open %s\n", file.c_str());
        exit(1);
    }

    // Write data
    fwrite(data, sizeof(float), size, array_file);

    // Close file
    fclose(array_file);
}
