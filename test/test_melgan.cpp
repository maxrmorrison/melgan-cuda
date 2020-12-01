#include <math.h>
#include <string>

#include "acutest.hpp"
#include "block.hpp"
#include "convert.hpp"
#include "cuda.hpp"
#include "dummy.hpp"
#include "engine.hpp"
#include "layer.hpp"
#include "load.hpp"
#include "melgan.hpp"
#include "profile.hpp"
#include "save.hpp"


/******************************************************************************
Constants
******************************************************************************/


const unsigned int ITERATIONS = 100;
const std::string TEST_ASSETS_DIR =
    "/home/mrm5248/468-final/melgan-cuda/test/assets/";
const uint TEST_FRAMES = 887;
const uint TEST_PADDING = 3;


/******************************************************************************
Full system test
******************************************************************************/


void test_melgan()
{
    // Load conv weights on device
    engine::start();

    // Load mels
    float *mels = load(TEST_ASSETS_DIR + "mels.f32", N_MELS * TEST_FRAMES);

    // Allocate mels
    unsigned int mel_size = N_MELS * TEST_FRAMES * sizeof(float);
    float *mels_d = cuda::allocate(mel_size);

    // Copy cpu memory to gpu
    cuda::copy_to_device(mels_d, mels, mel_size);

    // Run inference
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    float *audio = forward(mels_d, TEST_FRAMES, cudnn, false);
    TIME_IT("melgan", ITERATIONS, dummy::forward(mels_d, TEST_FRAMES, cudnn);)
    cudnnDestroy(cudnn);

    // Load answer
    const unsigned int samples = frames_to_samples(TEST_FRAMES);
    float *answer = load(TEST_ASSETS_DIR + "output.f32", samples);

    // Did we match the answer?
    bool correct = true;
    for (uint i = 0; i < samples; ++i) {
        if (answer[i] != audio[i]) correct = false;
    }

    // Save result
    save(TEST_ASSETS_DIR + "result.f32", audio, samples);

    // Free memory
    free(audio);
    free(answer);
    engine::stop();

    TEST_CHECK(correct);
}


/******************************************************************************
Block tests
******************************************************************************/


void test_residual_block()
{
    engine::start();
    const unsigned int frames = 8 * TEST_FRAMES;
    const unsigned int size = frames * CONV_2.input_channels;
    const unsigned int bytes = size * sizeof(float);

    // Load input and expected output
    float *activation = load(TEST_ASSETS_DIR + "transpose_conv.f32", size);
    float *answer = load(TEST_ASSETS_DIR + "residual_block.f32", size);

    // Copy to device
    float *activation_d = cuda::allocate(bytes);
    cuda::copy_to_device(activation_d, activation, bytes);

    // Perform op
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    ResidualBlock block = {CONV_2, CONV_3, CONV_4};
    activation_d = residual_block(activation_d, frames, block, cudnn);
    cudnnDestroy(cudnn);

    // Copy to host
    free(activation);
    activation = (float *) malloc(bytes);
    cuda::copy_to_host(activation, activation_d, bytes);

    // Did we get the right answer?
    bool correct = true;
    for (unsigned int i = 0; i < size; ++i) {
        if (activation[i] != answer[i]) correct = false;
    }

    // Free memory
    free(answer);
    free(activation);
    cuda::free(activation_d);
    engine::stop();

    TEST_CHECK(correct);
}


/******************************************************************************
Layer tests
******************************************************************************/


void test_add()
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
    for (uint i = 0; i < size; ++i) TEST_CHECK(answer[i] == x[i]);

    // Free memory
    free(x);
    free(y);
    free(answer);
    cuda::free(x_d);
    cuda::free(y_d);
}


void test_conv()
{
    engine::start();
    const unsigned int input_frames = TEST_FRAMES + (2 * TEST_PADDING);
    const unsigned int input_size = input_frames * N_MELS;
    const unsigned int input_bytes = input_size * sizeof(float);
    const unsigned int output_frames = get_num_output_frames_forward(
        input_frames, CONV_0);
    const unsigned int output_size = output_frames * N_MELS;
    const unsigned int output_bytes = output_size * sizeof(float);

    // Load input and expected output
    float *activation = load(
        TEST_ASSETS_DIR + "reflection_padding.f32", input_size);
    float *answer = load(TEST_ASSETS_DIR + "conv.f32", output_size);

    // Copy to device
    float *activation_d = cuda::allocate(input_bytes);
    cuda::copy_to_device(activation_d, activation, input_bytes);

    // Perform op
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    float *result_d = layer::conv(
        activation_d, input_frames, CONV_0, cudnn, false);
    TIME_IT(
        "conv",
        ITERATIONS,
        dummy::conv(activation_d, input_frames, CONV_0, cudnn);)
    cudnnDestroy(cudnn);

    // Copy to host
    free(activation);
    activation = (float *) malloc(output_bytes);
    cuda::copy_to_host(activation, result_d, output_bytes);

    // Did we get the right answer?
    for (unsigned int i = 0; i < output_size; ++i) {
        TEST_CHECK(answer[i] == activation[i]);
    }

    // Free memory
    free(answer);
    free(activation);
    cuda::free(result_d);
    cuda::free(activation_d);
    engine::stop();
}


void test_leaky_relu()
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
    for (uint i = 0; i < size; ++i) TEST_CHECK(answer[i] == activation[i]);

    // Profile
    TIME_IT("leaky_relu", ITERATIONS, layer::leaky_relu(activation_d, size);)

    // Free memory
    free(answer);
    free(activation);
    cuda::free(activation_d);
}


void test_reflection_padding()
{
    const unsigned int input_size = N_MELS * TEST_FRAMES;
    const unsigned int input_bytes = input_size * sizeof(float);
    const unsigned int output_size = input_size + 2 * TEST_PADDING * N_MELS;
    const unsigned int output_bytes = output_size * sizeof(float);

    // Load input and expected output
    float *activation = load(TEST_ASSETS_DIR + "mels.f32", input_size);
    float *answer = load(
        TEST_ASSETS_DIR + "reflection_padding.f32", output_size);

    // Copy to device
    float *activation_d = cuda::allocate(input_bytes);
    cuda::copy_to_device(activation_d, activation, input_bytes);

    // Perform op
    float *result_d = layer::reflection_padding(
        activation_d, TEST_FRAMES, N_MELS, TEST_PADDING, false);
    TIME_IT(
        "reflection_padding",
        ITERATIONS,
        dummy::reflection_padding(
            activation_d, TEST_FRAMES, N_MELS, TEST_PADDING);)

    // Copy to host
    free(activation);
    activation = (float *) malloc(output_bytes);
    cuda::copy_to_host(activation, result_d, output_bytes);

    // Did we get the right answer?
    for (unsigned int i = 0; i < output_size; ++i) {
        TEST_CHECK(answer[i] == activation[i]);
    }

    // Free memory
    free(answer);
    free(activation);
    cuda::free(activation_d);
    cuda::free(result_d);
}


void test_tanh()
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
        TEST_CHECK(abs(answer[i] - activation[i]) < 1e-4);

    // Profile
    TIME_IT("tanh", ITERATIONS, layer::tanh(activation_d, size);)

    // Free memory
    free(answer);
    free(activation);
    cuda::free(activation_d);
}


void test_transpose_conv()
{
    engine::start();
    const unsigned int input_frames = TEST_FRAMES;
    const unsigned int input_size = input_frames * CONV_1.output_channels;
    const unsigned int input_bytes = input_size * sizeof(float);
    const unsigned int output_frames = get_num_output_frames_backward(
        input_frames, CONV_1);
    const unsigned int output_size = output_frames * CONV_1.input_channels;
    const unsigned int output_bytes = output_size * sizeof(float);

    // Load input and expected output
    float *activation = load(TEST_ASSETS_DIR + "leaky_relu.f32", input_size);
    float *answer = load(TEST_ASSETS_DIR + "transpose_conv.f32", output_size);

    // Copy to device
    float *activation_d = cuda::allocate(input_bytes);
    cuda::copy_to_device(activation_d, activation, input_bytes);

    // Perform op
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    float *result_d = layer::transpose_conv(
        activation_d, input_frames, CONV_1, cudnn, false);
    TIME_IT(
        "transpose_conv",
        ITERATIONS,
        dummy::transpose_conv(activation_d, input_frames, CONV_1, cudnn);)
    cudnnDestroy(cudnn);

    // Copy to host
    free(activation);
    activation = (float *)malloc(output_bytes);
    cuda::copy_to_host(activation, result_d, output_bytes);

    // Did we get the right answer?
    bool correct = true;
    for (unsigned int i = 0; i < output_size; ++i) {
        if (answer[i] != activation[i]) correct = false;
    }

    // Free memory
    free(answer);
    free(activation);
    cuda::free(result_d);
    cuda::free(activation_d);

    engine::stop();

    TEST_CHECK(correct);
}


/******************************************************************************
Loading tests
******************************************************************************/


void test_load()
{
    const uint size = N_MELS * TEST_FRAMES;

    // Load data
    float *data = load(TEST_ASSETS_DIR + "mels.f32", size);

    // Do the first, middle, and last values match?
    const float epsilon = 1e-5;
    TEST_CHECK(abs(-1.7135957 - data[0]) < epsilon);
    TEST_CHECK(abs(-3.7742786 - data[size / 2]) < epsilon);
    TEST_CHECK(abs(-4.257396 - data[size - 1]) < epsilon);
}


void test_easy_load()
{
    // Load data
    float *data = load(TEST_ASSETS_DIR + "test_load.f32", 4);

    // Do the values match?
    for (unsigned int i = 0; i < 4; ++i) TEST_CHECK((float) i == data[i]);
}


/******************************************************************************
Acutest setup
******************************************************************************/


TEST_LIST = {
    {"test_melgan", test_melgan},
    {"test_residual_block", test_residual_block},
    {"test_add", test_add},
    {"test_conv", test_conv},
    {"test_leaky_relu", test_leaky_relu},
    {"test_reflection_padding", test_reflection_padding},
    {"test_tanh", test_tanh},
    {"test_transpose_conv", test_transpose_conv},
    {"test_load", test_load},
    {"test_easy_load", test_easy_load},
    {NULL, NULL}};
