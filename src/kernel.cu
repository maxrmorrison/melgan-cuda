#include <stdio.h>

#include "kernel.hpp"
#include "model.hpp"


/******************************************************************************
Constants
******************************************************************************/


/* leaky_relu activation slope */
const float LEAKY_RELU_SLOPE = .2;

/* amount of reflection padding per side */
const unsigned int REFLECTION_PAD_SIZE = 3;


/******************************************************************************
Kernels
******************************************************************************/


namespace kernel {
    /* addition */
    __global__ void add(float *x,
                        const float * const y,
                        const unsigned int size)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) x[index] += y[index];
    }

    __global__ void broadcast_add(float *x,
                                  const float * const y,
                                  const unsigned int rows,
                                  const unsigned int cols)
    {

    }


    /* convolution */
    __global__ void conv(const float * const input,
                         float *output,
                         const unsigned int frames,
                         const Convolution convolution)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid == 0) {
            printf(
                "%d %d %d %d %d %d %d\n",
                convolution.input_channels,
                convolution.output_channels,
                convolution.dilation,
                convolution.kernel_size,
                convolution.reflection_padding,
                convolution.stride,
                convolution.zero_padding
            );
        }
        __syncthreads();

        // TODO - correct conv with no optimizations
        // Note - assume stride is always 1 and zero_padding is always 0
        // for (unsigned int i = 0; c < convolution.output_channels; ++i) {
        //     float sum = 0.;
        //     for (int k = 0; k < kernel_size; ++k) {
        //         for (unsigned int j = 0; j < convolution.input_channels; ++j) {
        //             float input = input[];
        //             float weight = convolution.weight[];
        //         }
        //     }
        // }
    }


    /* leaky relu activation */
    __global__ void leaky_relu(float *input, const unsigned int size)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            const float value = input[index];
            input[index] = fmaxf(value, LEAKY_RELU_SLOPE * value);
            // OR
            // if (value < 0) input[index] = LEAKY_RELU_SLOPE * value;
        }
    }


    /* tanh activation */
    __global__ void tanh(float *input, const unsigned int size)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) input[index] = tanhf(input[index]);
    }


    /* transpose convolution */
    __global__ void transpose_conv(const float * const input,
                                   const float * const weight,
                                   const float * const bias)
    {
        // TODO
    }
}
