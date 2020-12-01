#ifndef MODEL_HPP
#define MODEL_HPP


/******************************************************************************
Type definitions
******************************************************************************/


typedef struct {
    float *weight, *weight_d;
    float *bias, *bias_d;
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


extern Convolution CONV_0;
extern Convolution CONV_1;
extern Convolution CONV_2;
extern Convolution CONV_3;
extern Convolution CONV_4;
extern Convolution CONV_5;
extern Convolution CONV_6;
extern Convolution CONV_7;
extern Convolution CONV_8;
extern Convolution CONV_9;
extern Convolution CONV_10;
extern Convolution CONV_11;
extern Convolution CONV_12;
extern Convolution CONV_13;
extern Convolution CONV_14;
extern Convolution CONV_15;
extern Convolution CONV_16;
extern Convolution CONV_17;
extern Convolution CONV_18;
extern Convolution CONV_19;
extern Convolution CONV_20;
extern Convolution CONV_21;
extern Convolution CONV_22;
extern Convolution CONV_23;
extern Convolution CONV_24;
extern Convolution CONV_25;
extern Convolution CONV_26;
extern Convolution CONV_27;
extern Convolution CONV_28;
extern Convolution CONV_29;
extern Convolution CONV_30;
extern Convolution CONV_31;
extern Convolution CONV_32;
extern Convolution CONV_33;
extern Convolution CONV_34;
extern Convolution CONV_35;
extern Convolution CONV_36;
extern Convolution CONV_37;
extern Convolution CONV_38;
extern Convolution CONV_39;
extern Convolution CONV_40;
extern Convolution CONV_41;


#endif /* MODEL_HPP */
