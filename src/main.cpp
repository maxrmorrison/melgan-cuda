#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include "melgan.hpp"


/* Run melgan inference */
int main(int argc, char **argv)
{
    char *mels = nullptr;
    char *output = nullptr;
    unsigned int frames = -1;

    // Parse command-line arguments
    int option;
    while ((option = getopt(argc, argv, ":i:o:f:")) != -1) {
        switch (option) {
        case 'i':
            mels = optarg;
            printf("Input file: %s\n", mels);
            break;
        case 'o':
            output = optarg;
            printf("Output file: %s\n", output);
            break;
        case 'f':
            frames = atoi(optarg);
            printf("Frames: %u\n", frames);
            break;
        }
    }

    // Error check arguments
    if (mels == nullptr) {
        printf("No input file provided\n");
        return -1;
    }
    if (output == nullptr) {
        printf("No output file provided\n");
        return -1;
    }
    if (frames == -1) {
        printf("Number of frames not provided");
        return -1;
    }

    // Run inference
    infer_from_file_to_file(std::string(mels), frames, std::string(output));

    return 0;
}
