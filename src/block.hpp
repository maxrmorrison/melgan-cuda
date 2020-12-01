#ifndef BLOCK_HPP
#define BLOCK_HPP

#include <cudnn.h>

#include "model.hpp"


/* Forward pass through all layers */
float *forward(float *mels,
               unsigned int frames,
               cudnnHandle_t cudnn,
               bool free_input);

/* Forward pass through a residual block */
float *residual_block(float *activation,
                      const unsigned int frames,
                      const ResidualBlock &block,
                      cudnnHandle_t cudnn);

/* Forward pass through upsampling and three residual blocks */
float *upsample_residual_block(float *activation,
                               unsigned int &frames,
                               const UpsampleResidualBlock &block,
                               const unsigned int upsample_factor,
                               cudnnHandle_t cudnn);

#endif /* BLOCK_HPP */
