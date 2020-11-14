#ifndef BLOCK_HPP
#define BLOCK_HPP

#include "model.hpp"


/* Forward pass through all layers */
float *forward(float *mels, unsigned int frames);

/* Forward pass through a residual block */
float *residual_block(float *activation,
                      const unsigned int frames,
                      const ResidualBlock &block);

/* Forward pass through upsampling and a residual block */
float *upsample_residual_block(float *activation,
                               unsigned int &frames,
                               const UpsampleResidualBlock &block,
                               const unsigned int upsample_factor);


#endif /* BLOCK_HPP */
