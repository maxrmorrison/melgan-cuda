#include <cudnn.h>

#include "model.hpp"


/******************************************************************************
Dummy layers for timing purposes
******************************************************************************/


namespace dummy {
    /* convolution */
    void conv(float *input,
              const unsigned int frames,
              const Convolution &convolution,
              cudnnHandle_t cudnn);

    /* forward pass */
    void forward(float *mels, unsigned int frames, cudnnHandle_t cudnn);

    /* reflection padding */
    void reflection_padding(float *activation,
                            const unsigned int frames,
                            const unsigned int channels,
                            const unsigned int padding);

    /* transpose convolution */
    void transpose_conv(float *input,
                        const unsigned int frames,
                        const Convolution &convolution,
                        cudnnHandle_t cudnn);
}
