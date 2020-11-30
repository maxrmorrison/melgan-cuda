#ifndef KERNELS_HPP
#define KERNELS_HPP

#include "model.hpp"


/******************************************************************************
Constants
******************************************************************************/


/* leaky_relu activation slope */
extern const float LEAKY_RELU_SLOPE;

/* amount of reflection padding per side */
extern const unsigned int REFLECTION_PAD_SIZE;


/******************************************************************************
Kernels
******************************************************************************/


namespace kernel {
    /* addition */
    __global__ void add(float *x,
                        const float * const y,
                        const unsigned int size);

    /* leaky relu activation */
    __global__ void leaky_relu(float *input, const unsigned int size);

    /* reflection padding */
    __global__ void reflection_padding(float *input,
                                       float *output,
                                       const unsigned int frames,
                                       const unsigned int channels,
                                       const unsigned int padding);

    /* tanh activation */
    __global__ void tanh(float *input, const unsigned int size);
}


#endif /* KERNELS_HPP */
