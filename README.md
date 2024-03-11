Resemble Denoiser in ONNX
=======================

Denoiser from
https://github.com/resemble-ai/resemble-enhance

Resemble released a nice open speech enhancer model. It is in 2 parts, denoiser and an enhancer. The enhancer is much larger model and the quality is not much better (and sometimes it adds weird artifacts).

All PyTorch dependencies are removed. Allows for leaner distribution.


Installation
===
This project has been tested on Python 3.11. You will also need to install the necessary Python packages.

```pip install -r requirements.txt```

Alternativel install your own ONNX Runtime for running on the GPU.

Usage
===
The run.py script is designed to denoise an audio file. You need to supply a .wav file as input. The script reads the input audio file, processes it through the denoiser model, and saves the denoised audio output to a new file.

To use the script, follow these steps:

1. Place your test file in `test_audio.wav` file in the same directory as the run.py script or update the path variable in the script to point to your file's location.
3. Run the script using Python:
```python run.py```
The script will process the audio file and save the output to `denoiser_output.wav` in the same directory.

