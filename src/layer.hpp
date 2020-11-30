#ifndef LAYER_HPP
#define LAYER_HPP

#include <cudnn.h>

#include "model.hpp"


/******************************************************************************
Constants
******************************************************************************/


extern const unsigned int THREADS_PER_BLOCK;


/******************************************************************************
Layers
******************************************************************************/


namespace layer {
    /* addition */
    float *add(float *x, float *y, const unsigned int size);

    /* convolution */
    float *conv(float *input,
                const unsigned int frames,
                const Convolution &convolution,
                cudnnHandle_t cudnn);

    /* convolution without freeing activation */
    float *conv_no_free(float *input,
                        const unsigned int frames,
                        const Convolution &convolution,
                        cudnnHandle_t cudnn);

    /* leaky relu activation */
    float *leaky_relu(float *activation, const unsigned int size);

    /* printing utility */
    void print(float *activation, const unsigned int size);

    /* reflection padding */
    float *reflection_padding(float *activation,
                                const unsigned int frames,
                                const unsigned int channels,
                                const unsigned int padding);

    /* tanh activation */
    float *tanh(float *activation, const unsigned int size);

    /* transpose convolution */
    float *transpose_conv(float *input,
                          const unsigned int frames,
                          const Convolution &convolution,
                          cudnnHandle_t cudnn);
}


/******************************************************************************
Utilities
******************************************************************************/


/* retrieve number of output frames */
unsigned int get_num_output_frames_backward(unsigned int input_frames,
                                            const Convolution &convolution);
unsigned int get_num_output_frames_forward(unsigned int input_frames,
                                           const Convolution &convolution);

#endif /* LAYER_HPP */
