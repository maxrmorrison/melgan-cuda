#ifndef CUDA_HPP
#define CUDA_HPP

#include "model.hpp"


namespace cuda {
    /* allocate memory */
    float *allocate(const unsigned int size);

    /* copy memory to device */
    void copy_to_device(float *device,
                        const float *host,
                        const unsigned int size);

    /* copy weight and bias to device */
    void copy_to_device(Convolution &convolution, bool transpose);

    /* copy memory to host */
    void copy_to_host(float *host,
                      const float *device,
                      const unsigned int size);

    /* free memory */
    void free(float *data);

    /* free weight and bias */
    void free(Convolution &convolution);
}


#endif /* CUDA_HPP */
