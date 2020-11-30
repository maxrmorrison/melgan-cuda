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


    /* leaky relu activation */
    __global__ void leaky_relu(float *input, const unsigned int size)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            const float value = input[index];
            input[index] = fmaxf(value, LEAKY_RELU_SLOPE * value);
        }
    }


    /* printing utility */
    __global__ void print(float *input, const unsigned int size)
    {
        if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
            for (unsigned int i = 1000000; i < 1000000 + size; ++i) printf("%f ", input[i]);
            printf("\n");
        }
    }


    /* reflection padding */
    __global__ void reflection_padding(float *input,
                                       float *output,
                                       const unsigned int frames,
                                       const unsigned int channels,
                                       const unsigned int padding)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int output_frames = frames + 2 * padding;
        const unsigned int channel = index / output_frames;
        const int output_frame = index % output_frames;
        int input_frame = (output_frame - padding);

        if (index < output_frames * channels) {
            // Reflect
            if (input_frame < 0)
                input_frame = -input_frame;
            else if (input_frame >= frames)
                input_frame = frames - (input_frame - frames) - 2;

            // Pad
            output[channel * output_frames + output_frame] =
                input[channel * frames + input_frame];
        }
    }


    /* tanh activation */
    __global__ void tanh(float *input, const unsigned int size)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) input[index] = tanhf(input[index]);
    }
}
