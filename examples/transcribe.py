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

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=["tiny.en", "base.en", "small.en"])
    parser.add_argument("audio", type=str)
    parser.add_argument("--backend", type=str, choices=["whisper", "whisper_trt", "faster_whisper"], default="whisper_trt")
    args = parser.parse_args()

    if args.backend == "whisper":

        from whisper import load_model

        model = load_model(args.model)

        result = model.transcribe(args.audio)
        
    elif args.backend == "whisper_trt":

        from whisper_trt import load_trt_model

        model = load_trt_model(args.model)

        result = model.transcribe(args.audio)

    elif args.backend == "faster_whisper":

        from faster_whisper import WhisperModel

        model = WhisperModel(args.model)

        segs, info = model.transcribe(args.audio)

        text = "".join(seg.text for seg in segs)
        result = {"text": text}

    else:
        raise RuntimeError("Unsupported backend.")
    
    print(f"Result: {result['text']}")
