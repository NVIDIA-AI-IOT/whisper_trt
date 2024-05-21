# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import os
import argparse
from whisper_trt import load_trt_model


if __name__ == "__main__":

    import time
    import psutil
    from multiprocessing import Process

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=["tiny.en", "base.en", "small.en"])
    parser.add_argument("audio", type=str)
    parser.add_argument("--backend", type=str, choices=["whisper", "whisper_trt", "faster_whisper"], default="whisper_trt")
    args = parser.parse_args()

    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss

    def profile_model(model, audio, iters: int = 1, is_faster_whisper: bool = False):
        
        result = model.transcribe(audio)

        if is_faster_whisper:
            result = {"text": "".join(seg.text for seg in result[0])}

        torch.cuda.current_stream().synchronize()
        t0 = time.perf_counter_ns()
        for i in range(iters):
            result = model.transcribe(audio)
            if is_faster_whisper:
                result = {"text": "".join(seg.text for seg in result[0])}

        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter_ns()

        return result, (t1 - t0) / (iters * 1e9)

    def run(model: str, backend: str, audio: str):
        start_mem = process_memory()
        if backend == "whisper":
            from whisper import load_model
            result, latency = profile_model(load_model(model), audio)
        elif backend == "whisper_trt":
            result, latency = profile_model(load_trt_model(model), audio)
        elif backend == "faster_whisper":
            from faster_whisper import WhisperModel
            result, latency = profile_model(WhisperModel(model), audio, is_faster_whisper=True)
        else:
            raise RuntimeError("unsupported backend")

        end_mem = process_memory()
        print(f"Backend: {backend}")
        print(f"\tResult: {result['text']}")
        print(f"\tLatency: {latency} Seconds")
        print(f"\tMemory: {(end_mem - start_mem) >> 20} MB")
    
    run(
        args.model,
        args.backend,
        args.audio
    )