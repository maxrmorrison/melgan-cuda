#include <cudnn.h>
#include <iostream>
#include <stdio.h>

#include "cuda.hpp"
#include "layer.hpp"
#include "kernel.hpp"


/******************************************************************************
Utilities
******************************************************************************/


static int checkCudnnError(cudnnStatus_t code,
                           const char *expr,
                           const char *file,
                           int line)
{
    if (code)
    {
        printf(
            "CUDNN error at %s:%d, code=%d (%s) in '%s'\n",
            file,
            line,
            (int)code,
            cudnnGetErrorString(code),
            expr);
        return 1;
    }
    return 0;
}


#define checkCudnnErr(...)                                  \
    do                                                      \
    {                                                       \
        int err = checkCudnnError(                          \
            __VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err) exit(1);                                   \
    } while (0)


unsigned int get_num_output_frames_backward(unsigned int input_frames,
                                            const Convolution &convolution)
{
    unsigned int d = convolution.dilation;
    unsigned int k = convolution.kernel_size;
    unsigned int p = convolution.zero_padding;
    unsigned int s = convolution.stride;
    return (input_frames - 1) * s - 2 * p + d * (k - 1) + 1;
}


unsigned int get_num_output_frames_forward(unsigned int input_frames,
                                           const Convolution &convolution)
{
    unsigned int d = convolution.dilation;
    unsigned int k = convolution.kernel_size;
    unsigned int p = convolution.zero_padding;
    unsigned int s = convolution.stride;
    return ((input_frames + 2 * p - d * (k - 1) - 1) / (float) s + 1);
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
                const Convolution &convolution,
                cudnnHandle_t cudnn,
                bool free_input)
    {
        unsigned int output_frames = get_num_output_frames_forward(
            frames, convolution);

        // Setup input
        cudnnTensorDescriptor_t input_descriptor;
        checkCudnnErr(cudnnCreateTensorDescriptor(&input_descriptor));
        checkCudnnErr(cudnnSetTensor4dDescriptor(
            input_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/convolution.input_channels,
            /*image_height=*/1,
            /*image_width=*/frames));

        // Setup output
        cudnnTensorDescriptor_t output_descriptor;
        checkCudnnErr(cudnnCreateTensorDescriptor(&output_descriptor));
        checkCudnnErr(cudnnSetTensor4dDescriptor(
            output_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/convolution.output_channels,
            /*image_height=*/1,
            /*image_width=*/output_frames));

        // Setup kernel
        cudnnFilterDescriptor_t kernel_descriptor;
        checkCudnnErr(cudnnCreateFilterDescriptor(&kernel_descriptor));
        checkCudnnErr(cudnnSetFilter4dDescriptor(
            kernel_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/convolution.output_channels,
            /*in_channels=*/convolution.input_channels,
            /*kernel_height=*/1,
            /*kernel_width=*/convolution.kernel_size));

        // Setup bias
        cudnnTensorDescriptor_t bias_descriptor;
        checkCudnnErr(cudnnCreateTensorDescriptor(&bias_descriptor));
        checkCudnnErr(cudnnSetTensor4dDescriptor(
            bias_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/convolution.output_channels,
            /*image_height=*/1,
            /*image_width=*/1));

        // Setup convolution
        cudnnConvolutionDescriptor_t convolution_descriptor;
        checkCudnnErr(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        checkCudnnErr(cudnnSetConvolution2dDescriptor(
            convolution_descriptor,
            /*pad_height=*/0,
            /*pad_width=*/convolution.zero_padding,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/convolution.stride,
            /*dilation_height=*/1,
            /*dilation_width=*/convolution.dilation,
            /*mode=*/CUDNN_CROSS_CORRELATION,
            /*computeType=*/CUDNN_DATA_FLOAT));
        checkCudnnErr(cudnnSetConvolutionMathType(
            convolution_descriptor,
            CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

        // Setup workspace
        cudnnConvolutionFwdAlgo_t convolution_algorithm =
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        size_t workspace_bytes = 0;
        checkCudnnErr(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            convolution_algorithm,
            &workspace_bytes));
        void *workspace = nullptr;
        cudaMalloc(&workspace, workspace_bytes);

        // Setup activation
        cudnnActivationDescriptor_t activation_descriptor;
        checkCudnnErr(cudnnCreateActivationDescriptor(&activation_descriptor));
        checkCudnnErr(cudnnSetActivationDescriptor(
            activation_descriptor,
            CUDNN_ACTIVATION_IDENTITY,
            CUDNN_PROPAGATE_NAN,
            0.));

        // Allocate output
        float *output = cuda::allocate(
            convolution.output_channels * output_frames * sizeof(float));

        // No blending
        float alpha1 = 1.0f, alpha2 = 0.0f;

        // Perform op
        checkCudnnErr(cudnnConvolutionBiasActivationForward(
            cudnn,
            &alpha1,
            input_descriptor,
            input,
            kernel_descriptor,
            convolution.weight_d,
            convolution_descriptor,
            convolution_algorithm,
            workspace,
            workspace_bytes,
            &alpha2,
            output_descriptor,
            output,
            bias_descriptor,
            convolution.bias_d,
            activation_descriptor,
            output_descriptor,
            output));

        // Clean up
        cudnnDestroyActivationDescriptor(activation_descriptor);
        cudnnDestroyConvolutionDescriptor(convolution_descriptor);
        cudnnDestroyFilterDescriptor(kernel_descriptor);
        cudnnDestroyTensorDescriptor(input_descriptor);
        cudnnDestroyTensorDescriptor(output_descriptor);
        cudnnDestroyTensorDescriptor(bias_descriptor);
        cudaFree(workspace);

        // Optionally free input
        if (free_input) cuda::free(input);

        // User frees output
        return output;
    }


    /* leaky relu activation */
    float *leaky_relu(float *activation, const unsigned int size)
    {
        const unsigned int blocks =
            (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel::leaky_relu<<<blocks, THREADS_PER_BLOCK>>>(activation, size);
        cudaDeviceSynchronize();
        return activation;
    }


    /* printing utility */
    void print(float *activation, const unsigned int size)
    {
        kernel::print<<<1, 1>>>(activation, size);
        cudaDeviceSynchronize();
    }


    float *reflection_padding(float *activation,
                              const unsigned int frames,
                              const unsigned int channels,
                              const unsigned int padding,
                              bool free_input)
    {
        // Allocate output
        const unsigned int output_frames = frames + 2 * padding;
        const unsigned int output_size = output_frames * channels;
        const unsigned int output_bytes = output_size * sizeof(float);
        float *output = cuda::allocate(output_bytes);

        // Perform padding
        const unsigned int blocks =
            (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel::reflection_padding<<<blocks, THREADS_PER_BLOCK>>>(
            activation, output, frames, channels, padding);
        cudaDeviceSynchronize();

        // Optionally free input
        if (free_input) cuda::free(activation);

        // User frees output
        return output;
    }


    /* tanh activation */
    float *tanh(float *activation, const unsigned int size)
    {
        const unsigned int blocks =
            (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel::tanh<<<blocks, THREADS_PER_BLOCK>>>(activation, size);
        cudaDeviceSynchronize();
        return activation;
    }


    /* transpose convolution */
    float *transpose_conv(float *input,
                          const unsigned int frames,
                          const Convolution &convolution,
                          cudnnHandle_t cudnn,
                          bool free_input)
    {
        unsigned int output_frames = get_num_output_frames_backward(
            frames, convolution);

        // Setup input
        cudnnTensorDescriptor_t input_descriptor;
        checkCudnnErr(cudnnCreateTensorDescriptor(&input_descriptor));
        checkCudnnErr(cudnnSetTensor4dDescriptor(
            input_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/convolution.output_channels,
            /*image_height=*/1,
            /*image_width=*/frames));

        // Setup output
        cudnnTensorDescriptor_t output_descriptor;
        checkCudnnErr(cudnnCreateTensorDescriptor(&output_descriptor));
        checkCudnnErr(cudnnSetTensor4dDescriptor(
            output_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/convolution.input_channels,
            /*image_height=*/1,
            /*image_width=*/output_frames));

        // Setup kernel
        cudnnFilterDescriptor_t kernel_descriptor;
        checkCudnnErr(cudnnCreateFilterDescriptor(&kernel_descriptor));
        checkCudnnErr(cudnnSetFilter4dDescriptor(
            kernel_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/convolution.output_channels,
            /*in_channels=*/convolution.input_channels,
            /*kernel_height=*/1,
            /*kernel_width=*/convolution.kernel_size));

        // Setup bias
        cudnnTensorDescriptor_t bias_descriptor;
        checkCudnnErr(cudnnCreateTensorDescriptor(&bias_descriptor));
        checkCudnnErr(cudnnSetTensor4dDescriptor(
            bias_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/convolution.input_channels,
            /*image_height=*/1,
            /*image_width=*/1));

        // Setup convolution
        cudnnConvolutionDescriptor_t convolution_descriptor;
        checkCudnnErr(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        checkCudnnErr(cudnnSetConvolution2dDescriptor(
            convolution_descriptor,
            /*pad_height=*/0,
            /*pad_width=*/convolution.zero_padding,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/convolution.stride,
            /*dilation_height=*/1,
            /*dilation_width=*/convolution.dilation,
            /*mode=*/CUDNN_CROSS_CORRELATION,
            /*computeType=*/CUDNN_DATA_FLOAT));

        // Setup workspace
        cudnnConvolutionBwdDataAlgo_t convolution_algorithm =
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        size_t workspace_bytes = 0;
        checkCudnnErr(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnn,
            kernel_descriptor,
            input_descriptor,
            convolution_descriptor,
            output_descriptor,
            convolution_algorithm,
            &workspace_bytes));
        void *workspace = nullptr;
        cudaMalloc(&workspace, workspace_bytes);

        // Allocate output
        float *output = cuda::allocate(
            convolution.input_channels * output_frames * sizeof(float));

        // No blending
        float alpha = 1.0f, beta = 0.0f;

        // Perform op
        checkCudnnErr(cudnnConvolutionBackwardData(
            cudnn,
            &alpha,
            kernel_descriptor,
            convolution.weight_d,
            input_descriptor,
            input,
            convolution_descriptor,
            convolution_algorithm,
            workspace,
            workspace_bytes,
            &beta,
            output_descriptor,
            output));
        beta = 1.;
        checkCudnnErr(cudnnAddTensor(
            cudnn,
            &alpha,
            bias_descriptor,
            convolution.bias_d,
            &beta,
            output_descriptor,
            output));

        // Clean up
        cudnnDestroyConvolutionDescriptor(convolution_descriptor);
        cudnnDestroyFilterDescriptor(kernel_descriptor);
        cudnnDestroyTensorDescriptor(input_descriptor);
        cudnnDestroyTensorDescriptor(output_descriptor);
        cudnnDestroyTensorDescriptor(bias_descriptor);
        cudaFree(workspace);

        // Optionally free input
        if (free_input) cuda::free(input);

        // User frees output
        return output;
        }
    }
