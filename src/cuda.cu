#include "cuda.hpp"


namespace cuda {
    /* allocate memory */
    float *allocate(const unsigned int size)
    {
        float *data;
        cudaMalloc((void **) &data, size);
        return data;
    }


    /* copy memory to device */
    void copy_to_device(float *device,
                        const float *host,
                        const unsigned int size)
    {
        cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
    }


    /* copy memory to host */
    void copy_to_host(float *host,
                      const float *device,
                      const unsigned int size)
    {
        cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
    }


    /* free memory */
    void free(float *data)
    {
        cudaFree(data);
    }
}
