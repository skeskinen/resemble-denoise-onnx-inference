import time
import numpy as np
import scipy
from denoiser import run
import onnxruntime

path = './test_audio.wav'

sr, dwav = scipy.io.wavfile.read(path)

if dwav.dtype == np.int16:
    dwav = dwav / 32768.
if dwav.ndim == 2:
    dwav = np.mean(dwav, axis=1)

dwav = dwav.astype(np.float32)

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
wav_onnx, new_sr = run(session, dwav, sr, batch_process_chunks=True)
print(f'Ran in {time.time() - start}s')

scipy.io.wavfile.write('denoiser_output.wav', new_sr, wav_onnx)
