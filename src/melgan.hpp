#ifndef INFER_HPP
#define INFER_HPP


#include <string>


/******************************************************************************
Constants
******************************************************************************/


/* Number of mels */
extern const uint N_MELS;


/******************************************************************************
Inference
******************************************************************************/


/* Infer waveform from mels on cpu */
float *infer(const float *mels, const uint frames);

/* Infer waveform from mels on disk */
float *infer_from_file(const std::string input, const uint frames);

/* Infer waveform from mels on disk and save */
void infer_from_file_to_file(const std::string input,
                             const uint frames,
                             const std::string output);


#endif /* INFER_HPP */
