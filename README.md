# melgan-cuda

CUDA implementation of the MelGAN vocoder. The original MelGAN implementation can be found [here](https://github.com/descriptinc/melgan-neurips). This implementation performs inference 3.78x faster than the original implementation on a NVIDIA RTX 2060 Super GPU. Further speed-ups are possible. A write-up describing this implementation and benchmarking can be found [here](https://www.maxrmorrison.com/pdfs/morrison2021improving.pdf/).


### Installation

Install CUDA and CUDNN. We recommend CUDA 11.0 and CUDNN 8.0.5, which were used
for testing. Then, run `make`.


### Usage

```
./build/melgan -i <input_file> -o <output_file> -f <frames>
```

 - `<input_file>` is the name of a 32-bit floating point file containing the
input mel spectrogram.
 - `<output_file>` is the name of a file to write the generated 32-bit floating
point audio signal.
 - `<frames>` is the number of mel spectrogram frames in the input.

`test/assets/mels.f32` can be used as an example input file with 887 frames.
The result should compare equal to `test/assets/output.f32`.


### Tests

`./build/test_melgan`
