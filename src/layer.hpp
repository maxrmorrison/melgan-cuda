#ifndef LAYER_HPP
#define LAYER_HPP

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
                const Convolution &convolution);

    /* convolution without freeing activation */
    float *conv_no_free(float *input,
                        const unsigned int frames,
                        const Convolution &convolution);

    /* leaky relu activation */
    float *leaky_relu(float *activation, const unsigned int size);

    /* tanh activation */
    float *tanh(float *activation, const unsigned int size);

    /* transpose convolution */
    float *transpose_conv(float *input,
                          const unsigned int frames,
                          const Convolution &convolution);
}


#endif /* LAYER_HPP */
