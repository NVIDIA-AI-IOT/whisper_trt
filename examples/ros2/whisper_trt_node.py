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


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from whisper_trt_pipeline import WhisperTRTPipeline


class WhisperTRTNode(Node):
    def __init__(self):
        super().__init__('WhisperTRTNode')

        self.declare_parameter("model", "small.en")
        self.declare_parameter("backend", "whisper_trt") 

        # TODO: remove placeholder default
        self.declare_parameter("cache_dir", "data")#rclpy.Parameter.Type.STRING)
        self.declare_parameter("vad_window", 5)

        self.declare_parameter("mic_device_index", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("mic_sample_rate", 16000)
        self.declare_parameter("mic_channels", 6)
        self.declare_parameter("mic_bitwidth", 2)
        self.declare_parameter("mic_channel_for_asr", 0)

        self.declare_parameter("speech_topic", "/speech")

        self.speech_publisher = self.create_publisher(
            String, 
            self.get_parameter("speech_topic").value, 
            10
        )

        logger = self.get_logger()

        def handle_vad_start():
            logger.info("vad start")

        def handle_vad_end():
            logger.info("vad end")

        def handle_asr(text):
            msg = String()
            msg.data = text
            self.speech_publisher.publish(msg)
            logger.info("published " + text)

        self.pipeline = WhisperTRTPipeline(
            model=self.get_parameter("model").value,
            vad_window=self.get_parameter("vad_window").value,
            backend=self.get_parameter("backend").value,
            cache_dir=self.get_parameter_or("cache_dir", None).value,
            vad_start_callback=handle_vad_start,
            vad_end_callback=handle_vad_end,
            asr_callback=handle_asr,
            mic_device_index=self.get_parameter_or("mic_device_index", None).value,
            mic_sample_rate=self.get_parameter("mic_sample_rate").value,
            mic_channel_for_asr=self.get_parameter("mic_channel_for_asr").value,
            mic_num_channels=self.get_parameter("mic_channels").value,
            mic_bitwidth=self.get_parameter("mic_bitwidth").value
        )

    def start_asr_pipeline(self):
        self.pipeline.start()


def main(args=None):
    rclpy.init(args=args)
    node = WhisperTRTNode()

    node.start_asr_pipeline()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()