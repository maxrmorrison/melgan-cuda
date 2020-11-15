#include <math.h>
#include <string>

#include <gtest/gtest.h>

#include "load.hpp"
#include "melgan.hpp"


/******************************************************************************
Constants
******************************************************************************/


const uint TEST_FRAMES = 887;
const std::string TEST_ASSETS_DIR =
    "/home/mrm5248/468-final/melgan-cuda/test/assets/";


/******************************************************************************
Tests
******************************************************************************/


TEST(test, load)
{
    const uint size = N_MELS * TEST_FRAMES;

    // Load data
    float *data = load(TEST_ASSETS_DIR + "mels.f32", size);

    // Do the first, middle, and last values match?
    const float epsilon = 1e-5;
    ASSERT_TRUE(abs(-1.7135957 - data[0]) < epsilon) << data[0];
    ASSERT_TRUE(abs(-3.7742786 - data[size / 2]) < epsilon) << data[size / 2];
    ASSERT_TRUE(abs(-4.257396 - data[size - 1]) < epsilon) << data[size - 1];
}


TEST(test, easy_load)
{
    // Load data
    float *data = load(TEST_ASSETS_DIR + "test_load.f32", 4);

    // Do the first, middle, and last values match?
    for (unsigned int i = 0; i < 4; ++i) ASSERT_EQ((float) i, data[i]);
}
