/* Convert number of spectrogram frames to number of audio samples */
int frames_to_samples(const unsigned int frames)
{
    return 256 * frames;
}
