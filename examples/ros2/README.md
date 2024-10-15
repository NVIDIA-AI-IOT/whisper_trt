# WhisperTRT - ROS2 Node

This example includes a ROS2 node for interfacing with WhisperTRT.

It includes the full pipeline, including connecting to a microphone, and outputs recognized speech
segments on the ``/speech`` topic.

| Name | Description | Default |
|------|-------------|---------|
| model | The Whisper model to use. | "small.en" |
| backend | The Whisper backend to use. | "whisper_trt" |
| cache_dir | Directory to cache the built models. | None |
| vad_window | Number of audio chunks to use in max-filter window for voice activity detection. | 5 |
| mic_device_index | The microphone device index. | None |
| mic_sample_rate | The microphone sample rate. | 16000 |
| mic_channels | The microphone number of channels. | 6 |
| mic_bitwidth | The microphone bitwidth. | 2 |
| speech_topic | The topic to publish speech segments to. | "/speech" |