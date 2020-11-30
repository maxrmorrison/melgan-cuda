#include <cudnn.h>

#include "block.hpp"
#include "convert.hpp"
#include "cuda.hpp"
#include "melgan.hpp"
#include "load.hpp"
#include "save.hpp"


static int
checkCudnnError(cudnnStatus_t code, const char *expr, const char *file, int line)
{
    if (code)
    {
        printf(
            "CUDNN error at %s:%d, code=%d (%s) in '%s'\n",
            file,
            line,
            (int)code,
            cudnnGetErrorString(code),
            expr);
        return 1;
    }
    return 0;
}


#define checkCudnnErr(...)                                  \
    do                                                      \
    {                                                       \
        int err = checkCudnnError(                          \
            __VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err) exit(1);                                   \
    } while (0)


/******************************************************************************
Constants
******************************************************************************/


/* Number of mels */
const unsigned int N_MELS = 80;


/******************************************************************************
Inference
******************************************************************************/


/* Infer waveform from mels on cpu */
float *infer(const float *mels, const unsigned int frames)
{
    // Setup cudnn
    cudnnHandle_t cudnn;
    checkCudnnErr(cudnnCreate(&cudnn));

    // Allocate mels
    unsigned int mel_size = N_MELS * frames * sizeof(float);
    float *mels_d = cuda::allocate(mel_size);

    // Copy cpu memory to gpu
    cuda::copy_to_device(mels_d, mels, mel_size);

    // Infer
    float *audio = forward(mels_d, frames, cudnn);

    // Free GPU memory
    cuda::free(mels_d);

    // Free cudnn
    cudnnDestroy(cudnn);

    // User frees audio
    return audio;
}


/* Infer waveform from mels on disk */
float *infer_from_file(const std::string input, const unsigned int frames)
{
    // Load from disk
    float *mels = load(input, N_MELS * frames);

    // Infer
    float *audio = infer(mels, frames);

    // Free mels
    free(mels);

    // User frees audio
    return audio;
}


/* Infer waveform from mels on disk and unsigned int */
void infer_from_file_to_file(const std::string input,
                             const unsigned int frames,
                             const std::string output)
{
    // Load and infer
    float *audio = infer_from_file(input, frames);

    // Save to disk
    save(output, audio, frames_to_samples(frames));

    // Free audio
    free(audio);
}
