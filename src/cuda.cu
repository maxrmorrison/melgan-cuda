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


    /* copy weight and bias to device */
    void copy_to_device(Convolution &convolution, bool transpose)
    {
        // Allocate weight
        float weight_size = convolution.output_channels *
                            convolution.input_channels *
                            convolution.kernel_size *
                            sizeof(float);
        float *weight = cuda::allocate(weight_size);
        cuda::copy_to_device(weight, convolution.weight, weight_size);
        convolution.weight_d = weight;

        // Allocate bias
        const unsigned int channels =
            transpose ? convolution.input_channels
                      : convolution.output_channels;
        float bias_bytes = channels * sizeof(float); //
        float *bias = cuda::allocate(bias_bytes);
        cuda::copy_to_device(bias, convolution.bias, bias_bytes);
        convolution.bias_d = bias;
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


    /* free weight and bias */
    void free(Convolution &convolution)
    {
        cudaFree(convolution.weight_d);
        cudaFree(convolution.bias_d);
    }
}
