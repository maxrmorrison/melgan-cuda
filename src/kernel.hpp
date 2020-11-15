#ifndef KERNELS_HPP
#define KERNELS_HPP


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

    /* convolution */
    __global__ void conv(const float * const input,
                         const float * const weight,
                         const float * const bias);

    /* leaky relu activation */
    __global__ void leaky_relu(float *input, const unsigned int size);

    /* tanh activation */
    __global__ void tanh(float *input, const unsigned int size);

    /* transpose convolution */
    __global__ void transpose_conv(const float * const input,
                                   const float * const weight,
                                   const float * const bias);
}


#endif /* KERNELS_HPP */
