import time
import numpy as np
from denoiser import run
import onnxruntime
import librosa
import scipy

path = './test_audio.wav'

wav, sr = librosa.load(path, mono=True)

opts = onnxruntime.SessionOptions()
opts.inter_op_num_threads = 4
opts.intra_op_num_threads = 4
opts.log_severity_level = 4

session = onnxruntime.InferenceSession(
    'denoiser.onnx',
    providers=["CPUExecutionProvider"],
    #providers=["ROCMExecutionProvider"],
    #providers=["DnnlExecutionProvider"],
    sess_options=opts,
)

start = time.time()
wav_onnx, new_sr = run(session, wav, sr, batch_process_chunks=False)
print(f'Ran in {time.time() - start}s')

scipy.io.wavfile.write('denoiser_output.wav', new_sr, wav_onnx)
