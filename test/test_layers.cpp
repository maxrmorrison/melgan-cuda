#include <gtest/gtest.h>
#include <string>

#include "convert.hpp"
#include "cuda.hpp"
#include "layer.hpp"
#include "load.hpp"


/******************************************************************************
Constants
******************************************************************************/


const uint TEST_FRAMES = 887;
const std::string TEST_ASSETS_DIR =
    "/home/mrm5248/468-final/melgan-cuda/test/assets/";


/******************************************************************************
Tests
******************************************************************************/


TEST(test, add)
{
    const uint size = 2 * THREADS_PER_BLOCK;
    const uint bytes = size * sizeof(float);

    // Setup x
    float *x = (float *) malloc(bytes);
    for (uint i = 0; i < size; ++i) x[i] = i;
    float *x_d = cuda::allocate(bytes);
    cuda::copy_to_device(x_d, x, bytes);

    // Setup y
    float *y = (float *) malloc(bytes);
    for (uint i = 0; i < size; ++i) y[i] = 2 * i;
    float *y_d = cuda::allocate(bytes);
    cuda::copy_to_device(y_d, y, bytes);

    // Setup answer
    float *answer = (float *) malloc(bytes);
    for (uint i = 0; i < size; ++i) answer[i] = 3 * i;

    // Add x and y
    float *z_d = layer::add(x_d, y_d, size);

    // Overwrite x
    cuda::copy_to_host(x, z_d, bytes);

    // Did we get the right answer?
    for (uint i = 0; i < size; ++i) ASSERT_EQ(answer[i], x[i]);

    // Free memory
    free(x);
    free(y);
    free(answer);
    cuda::free(x_d);
    cuda::free(y_d);
}


TEST(test, conv)
{
    // TODO - load input and output
    // TODO - conv
    // TODO - compare vectors
}


TEST(test, leaky_relu)
{
    const uint size = CONV_0.output_channels * TEST_FRAMES;
    const uint bytes = size * sizeof(float);

    // Load input and expected output
    float *activation = load(TEST_ASSETS_DIR + "conv.f32", size);
    float *answer = load(TEST_ASSETS_DIR + "leaky_relu.f32", size);

    // Copy to device
    float *activation_d = cuda::allocate(bytes);
    cuda::copy_to_device(activation_d, activation, bytes);

    // Perform op
    activation_d = layer::leaky_relu(activation_d, size);

    // Overwrite activation
    cuda::copy_to_host(activation, activation_d, bytes);

    // Did we get the right answer?
    for (uint i = 0; i < size; ++i) ASSERT_EQ(answer[i], activation[i]) << "Index " << i;

    // Free memory
    free(answer);
    free(activation);
    cuda::free(activation_d);
}


TEST(test, tanh)
{
    const uint size = frames_to_samples(TEST_FRAMES);
    const uint bytes = size * sizeof(float);

    // Load input and expected output
    float *activation = load(TEST_ASSETS_DIR + "tanh_input.f32", size);
    float *answer = load(TEST_ASSETS_DIR + "output.f32", size);

    // Copy to device
    float *activation_d = cuda::allocate(bytes);
    cuda::copy_to_device(activation_d, activation, bytes);

    // Perform op
    activation_d = layer::tanh(activation_d, size);

    // Overwrite activation
    cuda::copy_to_host(activation, activation_d, bytes);

    // Did we get the right answer?
    for (uint i = 0; i < size; ++i)
        ASSERT_TRUE(abs(answer[i] - activation[i]) < 1e-4);

    // Free memory
    free(answer);
    free(activation);
    cuda::free(activation_d);
}


TEST(test, transpose_conv)
{
    // TODO - load input and output
    // TODO - transpose_conv
    // TODO - compare vectors
}
