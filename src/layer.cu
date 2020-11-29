#include <cudnn.h>
#include <stdio.h>

#include "cuda.hpp"
#include "layer.hpp"
#include "kernel.hpp"


#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


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
        cudnnHandle_t cudnn;
        checkCUDNN(cudnnCreate(&cudnn));

        // Setup input
        cudnnTensorDescriptor_t input_descriptor;
        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
        checkCUDNN(cudnnSetTensor4dDescriptor(
            input_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/convolution.input_channels,
            /*image_height=*/1,
            /*image_width=*/frames));

        // Setup output
        // TODO - output size
        cudnnTensorDescriptor_t output_descriptor;
        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
        checkCUDNN(cudnnSetTensor4dDescriptor(
            output_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/convolution.output_channels,
            /*image_height=*/1,
            /*image_width=*/image.cols));

        // Setup kernel
        cudnnFilterDescriptor_t kernel_descriptor;
        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
        checkCUDNN(cudnnSetFilter4dDescriptor(
            kernel_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/convolution.output_channels,
            /*in_channels=*/convolution.input_channels,
            /*kernel_height=*/1,
            /*kernel_width=*/convolution.kernel_size));

        // Setup convolution
        // TODO - reflection padding
        cudnnConvolutionDescriptor_t convolution_descriptor;
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        checkCUDNN(cudnnSetConvolution2dDescriptor(
            convolution_descriptor,
            /*pad_height=*/0,
            /*pad_width=*/convolution.zero_padding,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/convolution.stride,
            /*dilation_height=*/1,
            /*dilation_width=*/convolution.dilation,
            /*mode=*/CUDNN_CROSS_CORRELATION,
            /*computeType=*/CUDNN_DATA_FLOAT));

        // Get correct sizes
        int batch_size{0}, channels{0}, height{0}, width{0};
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
            convolution_descriptor,
            input_descriptor,
            kernel_descriptor,
            &batch_size,
            &channels,
            &height,
            &width));

        // Setup conv algorithm
        cudnnConvolutionFwdAlgo_t convolution_algorithm;
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
            cudnn,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            /*memoryLimitInBytes=*/0,
            &convolution_algorithm));

        // Get workspace size
        size_t workspace_bytes = 0;
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            convolution_algorithm,
            &workspace_bytes));
        printf("Workspace size: %f\n", (workspace_bytes / 1048576.0));

        // TODO - fix output allocation
        float* d_output{nullptr};
        cudaMalloc(&d_output, image_bytes);
        cudaMemset(d_output, 0, image_bytes);

        const float alpha = 1.0f, beta = 0.0f;

        checkCUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            input_descriptor,
            d_input,
            kernel_descriptor,
            d_kernel,
            convolution_descriptor,
            convolution_algorithm,
            d_workspace,
            workspace_bytes,
            &beta,
            output_descriptor,
            d_output));

        // Clean up
        cudnnDestroyTensorDescriptor(input_descriptor);
        cudnnDestroyTensorDescriptor(output_descriptor);
        cudnnDestroyFilterDescriptor(kernel_descriptor);
        cudnnDestroyConvolutionDescriptor(convolution_descriptor);
        cudnnDestroy(cudnn);
        // unsigned int size = frames * convolution.output_channels;
        // unsigned int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        // float *output = cuda::allocate(size * sizeof(float));
        // kernel::conv<<<blocks, THREADS_PER_BLOCK>>>(
        //     input, output, frames, convolution);
        // return output;
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
