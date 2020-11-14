#include "kernel.hpp"


/******************************************************************************
Constants
******************************************************************************/


/* leaky_relu activation slope */
const unsigned int LEAKY_RELU_SLOPE = .2;

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
                         const float * const weight,
                         const float * const bias)
    {
        // TODO
        // NOTE: use modulus to implement reflection padding here
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
