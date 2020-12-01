#include "block.hpp"
#include "cuda.hpp"
#include "layer.hpp"
#include "melgan.hpp"
#include "model.hpp"


/* Forward pass through all layers */
float *forward(float *mels,
               unsigned int frames,
               cudnnHandle_t cudnn,
               bool free_input)
{
    // Input block
    float *activation;
    const unsigned int padding = CONV_0.reflection_padding;
    activation = layer::reflection_padding(
        mels, frames, N_MELS, padding, free_input);
    activation = layer::conv(
        activation, frames + 2 * padding, CONV_0, cudnn, true);

    // Block 0
    UpsampleResidualBlock block_0 = {
        CONV_1,
        {CONV_2, CONV_3, CONV_4 },
        {CONV_5, CONV_6, CONV_7 },
        {CONV_8, CONV_9, CONV_10}
    };
    activation = upsample_residual_block(
        activation, frames, block_0, 8, cudnn);

    // Block 1
    UpsampleResidualBlock block_1 = {
        CONV_11,
        {CONV_12, CONV_13, CONV_14},
        {CONV_15, CONV_16, CONV_17},
        {CONV_18, CONV_19, CONV_20}
    };
    activation = upsample_residual_block(
        activation, frames, block_1, 8, cudnn);

    // Block 2
    UpsampleResidualBlock block_2 = {
        CONV_21,
        {CONV_22, CONV_23, CONV_24},
        {CONV_25, CONV_26, CONV_27},
        {CONV_28, CONV_29, CONV_30}
    };
    activation = upsample_residual_block(
        activation, frames, block_2, 2, cudnn);

    // Block 3
    UpsampleResidualBlock block_3 = {
        CONV_31,
        {CONV_32, CONV_33, CONV_34},
        {CONV_35, CONV_36, CONV_37},
        {CONV_38, CONV_39, CONV_40}
    };
    activation = upsample_residual_block(
        activation, frames, block_3, 2, cudnn);

    // Output block
    activation = layer::leaky_relu(
        activation, frames * CONV_41.input_channels);
    activation = layer::reflection_padding(
        activation, frames, CONV_41.input_channels, padding, true);
    activation = layer::conv(
        activation, frames + 2 * padding, CONV_41, cudnn, true);
    activation = layer::tanh(activation, frames * CONV_41.output_channels);

    // Copy to host
    float * output = (float *) malloc(frames * sizeof(float));
    cuda::copy_to_host(output, activation, frames * sizeof(float));
    cudaFree(activation);
    return output;
}


/* Forward pass through a residual block */
float *residual_block(float *activation,
                      const unsigned int frames,
                      const ResidualBlock &block,
                      cudnnHandle_t cudnn)
{
    const unsigned int padding = block.conv.reflection_padding;
    float *shortcut = layer::conv(
        activation, frames, block.shortcut, cudnn, false);
    activation = layer::leaky_relu(activation,
                                   frames * block.conv.input_channels);
    activation = layer::reflection_padding(
        activation, frames, block.conv.input_channels, padding, true);
    activation = layer::conv(
        activation, frames + 2 * padding, block.conv, cudnn, true);
    activation = layer::leaky_relu(activation,
                                   frames * block.conv.output_channels);
    activation = layer::conv(activation, frames, block.linear, cudnn, true);
    activation = layer::add(activation, shortcut,
                            frames * block.linear.output_channels);
    cudaFree(shortcut);
    return activation;
}


/* Forward pass through upsampling and three residual blocks */
float *upsample_residual_block(float *activation,
                               unsigned int &frames,
                               const UpsampleResidualBlock &block,
                               const unsigned int upsample_factor,
                               cudnnHandle_t cudnn)
{
    // Upsample
    activation = layer::leaky_relu(
        activation,
        frames * block.transpose_conv.output_channels);
    activation = layer::transpose_conv(activation,
                                       frames,
                                       block.transpose_conv,
                                       cudnn,
                                       true);
    frames *= upsample_factor;

    // Residual blocks
    activation = residual_block(activation, frames, block.block_0, cudnn);
    activation = residual_block(activation, frames, block.block_1, cudnn);
    activation = residual_block(activation, frames, block.block_2, cudnn);
    return activation;
}
