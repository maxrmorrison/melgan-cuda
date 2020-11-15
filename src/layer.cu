#include <stdio.h>

#include "layer.hpp"
#include "kernel.hpp"


/******************************************************************************
Constants
******************************************************************************/


const unsigned int THREADS_PER_BLOCK = 1024;


/******************************************************************************
Layers
******************************************************************************/


namespace layer {
    /* addition */
    float *add(float *x, float *y, const unsigned int size)
    {
        // Add in-place
        const unsigned int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel::add<<<blocks, THREADS_PER_BLOCK>>>(x, y, size);
        cudaDeviceSynchronize();

        // Free y
        cudaFree(y);

        // User frees x
        return x;
    }


    /* convolution */
    float *conv(float *input,
                const unsigned int frames,
                const Convolution &convolution)
    {
        float *output = conv_no_free(input, frames, convolution);
        cudaFree(input);
        return output;
    }


    /* convolution without freeing input */
    float *conv_no_free(float *input,
                        const unsigned int frames,
                        const Convolution &convolution)
    {
        // TODO - allocate output
        // TODO - number of blocks and threads
        // TODO - conv kernel
        // TODO - broadcast_add
        float *output;
        return output;
    }


    /* leaky relu activation */
    float *leaky_relu(float *activation, const unsigned int size)
    {
        const unsigned int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel::leaky_relu<<<blocks, THREADS_PER_BLOCK>>>(activation, size);
        cudaDeviceSynchronize();
        return activation;
    }


    /* tanh activation */
    float *tanh(float *activation, const unsigned int size)
    {
        const unsigned int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel::tanh<<<blocks, THREADS_PER_BLOCK>>>(activation, size);
        cudaDeviceSynchronize();
        return activation;
    }


    /* transpose convolution */
    float *transpose_conv(float *input,
                          const unsigned int frames,
                          const Convolution &convolution)
    {
        // TODO
        float *output;
        return output;
    }
}
