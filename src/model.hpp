#ifndef MODEL_HPP
#define MODEL_HPP


/******************************************************************************
Type definitions
******************************************************************************/


typedef struct {
    const float *weight;
    const float *bias;
    const unsigned int input_channels;
    const unsigned int output_channels;
    const unsigned int dilation;
    const unsigned int kernel_size;
    const unsigned int reflection_padding;
    const unsigned int stride;
    const unsigned int zero_padding;
} Convolution;


typedef struct {
    const Convolution &conv;
    const Convolution &linear;
    const Convolution &shortcut;
} ResidualBlock;


typedef struct {
    const Convolution &transpose_conv;
    const ResidualBlock &block_0;
    const ResidualBlock &block_1;
    const ResidualBlock &block_2;
} UpsampleResidualBlock;


/******************************************************************************
Layers
******************************************************************************/


extern const Convolution CONV_0;
extern const Convolution CONV_1;
extern const Convolution CONV_2;
extern const Convolution CONV_3;
extern const Convolution CONV_4;
extern const Convolution CONV_5;
extern const Convolution CONV_6;
extern const Convolution CONV_7;
extern const Convolution CONV_8;
extern const Convolution CONV_9;
extern const Convolution CONV_10;
extern const Convolution CONV_11;
extern const Convolution CONV_12;
extern const Convolution CONV_13;
extern const Convolution CONV_14;
extern const Convolution CONV_15;
extern const Convolution CONV_16;
extern const Convolution CONV_17;
extern const Convolution CONV_18;
extern const Convolution CONV_19;
extern const Convolution CONV_20;
extern const Convolution CONV_21;
extern const Convolution CONV_22;
extern const Convolution CONV_23;
extern const Convolution CONV_24;
extern const Convolution CONV_25;
extern const Convolution CONV_26;
extern const Convolution CONV_27;
extern const Convolution CONV_28;
extern const Convolution CONV_29;
extern const Convolution CONV_30;
extern const Convolution CONV_31;
extern const Convolution CONV_32;
extern const Convolution CONV_33;
extern const Convolution CONV_34;
extern const Convolution CONV_35;
extern const Convolution CONV_36;
extern const Convolution CONV_37;
extern const Convolution CONV_38;
extern const Convolution CONV_39;
extern const Convolution CONV_40;
extern const Convolution CONV_41;


#endif /* MODEL_HPP */
