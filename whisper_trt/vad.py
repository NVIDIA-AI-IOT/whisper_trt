# MIT License
#
# Copyright (c) 2024-present NVIDIA
# Copyright (c) 2020-present Silero Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import onnxruntime
import torch
import numpy as np
import os
from .utils import check_file_md5, download_file
from .cache import get_cache_dir, make_cache_dir

# https://github.com/dusty-nv/jetson-containers/issues/568
SILERO_VAD_ONNX_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/1baf307b35ab3bbb070ab374b43a0a3c3604fa2a/files/silero_vad.onnx"
#SILERO_VAD_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"

SILERO_VAD_ONNX_FILENAME = "silero_vad.onnx"
SILERO_VAD_ONNX_MD5_CHECKSUM = "03da8de2fec4108a089b39f1b4abefef"


languages = ['ru', 'en', 'de', 'es']

__all__ = [
    "SileroVAD",
    "load_vad"
]

class SileroVAD():

    def __init__(self, path, force_onnx_cpu=False):

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and 'CPUExecutionProvider' in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=opts)
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:,::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")

        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._h = np.zeros((2, batch_size, 64)).astype('float32')
        self._c = np.zeros((2, batch_size, 64)).astype('float32')
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        x, sr = self._validate_input(x, sr)
        batch_size = x.shape[0]

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if sr in [8000, 16000]:
            ort_inputs = {'input': x.numpy(), 'h': self._h, 'c': self._c, 'sr': np.array(sr, dtype='int64')}
            ort_outs = self.session.run(None, ort_inputs)
            out, self._h, self._c = ort_outs
        else:
            raise ValueError()

        self._last_sr = sr
        self._last_batch_size = batch_size

        out = torch.tensor(out)
        return out

    def audio_forward(self, x, sr: int, num_samples: int = 512):
        outs = []
        x, sr = self._validate_input(x, sr)

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = torch.nn.functional.pad(x, (0, pad_num), 'constant', value=0.0)

        self.reset_states(x.shape[0])
        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i:i+num_samples]
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        stacked = torch.cat(outs, dim=1)
        return stacked.cpu()


def load_vad(path: str | None = None, download: bool = True):

    if path is None:
        make_cache_dir()
        path = os.path.join(get_cache_dir(), SILERO_VAD_ONNX_FILENAME)

    if not os.path.exists(path):
        if not download:
            raise RuntimeError("VAD model not found at {path}.  Please specified download=True.")
        download_file(SILERO_VAD_ONNX_URL, path)

    if not check_file_md5(path, SILERO_VAD_ONNX_MD5_CHECKSUM):
        raise RuntimeError("The MD5 Checksum for {path} does not match the expected value.")
    
    return SileroVAD(path)
