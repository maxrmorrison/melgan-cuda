#include <stdio.h>

#include "block.hpp"
#include "cuda.hpp"
#include "dummy.hpp"
#include "layer.hpp"


/******************************************************************************
Dummy layers for timing purposes
******************************************************************************/


namespace dummy {
    /* convolution */
    void conv(float *input,
              const unsigned int frames,
              const Convolution &convolution,
              cudnnHandle_t cudnn)
    {
        float *output = layer::conv(input, frames, convolution, cudnn, false);
        cuda::free(output);
    }

    /* forward pass */
    void forward(float *mels, unsigned int frames, cudnnHandle_t cudnn)
    {
        float *output = forward(mels, frames, cudnn, false);
        cuda::free(output);
    }

    /* reflection padding */
    void reflection_padding(float *activation,
                            const unsigned int frames,
                            const unsigned int channels,
                            const unsigned int padding)
    {
        float *output = layer::reflection_padding(
            activation, frames, channels, padding, false);
        cuda::free(output);
    }

    /* transpose convolution */
    void transpose_conv(float *input,
                        const unsigned int frames,
                        const Convolution &convolution,
                        cudnnHandle_t cudnn)
    {
        float *output = layer::transpose_conv(
            input, frames, convolution, cudnn, false);
        cuda::free(output);
    }
}
