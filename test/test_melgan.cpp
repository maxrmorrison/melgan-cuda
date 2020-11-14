#include <gtest/gtest.h>

#include "convert.hpp"
#include "load.hpp"
#include "melgan.hpp"


/******************************************************************************
Constants
******************************************************************************/


const uint TEST_FRAMES = 887;
const std::string TEST_ASSETS_DIR = "/home/mrm5248/468-final/melgan-cuda/test/assets/";


/******************************************************************************
Tests
******************************************************************************/


TEST(test, melgan)
{
    // Run inference
    float *audio = infer_from_file(TEST_ASSETS_DIR + "mels.f32", TEST_FRAMES);

    // Load answer
    float *answer = load(TEST_ASSETS_DIR + "output.f32", N_MELS * TEST_FRAMES);

    // Did we match the answer?
    const uint samples = frames_to_samples(TEST_FRAMES);
    for (uint i = 0; i < samples; ++i) ASSERT_EQ(answer[i], audio[i]);

    // Free memory
    free(audio);
    free(answer);
}
