#!/usr/bin/env bash


# Generate word.wav using Piper
echo "ready" | piper --model ./en_US-lessac-medium.onnx --output_file ready.wav
echo "set" | piper --model ./en_US-lessac-medium.onnx --output_file set.wav
echo "walk" | piper --model ./en_US-lessac-medium.onnx --output_file walk.wav
echo "stop" | piper --model ./en_US-lessac-medium.onnx --output_file stop.wav

# Resample to 48000Hz, 2 channel stereo for native play through PCM5102
ffmpeg -i stop_raw.wav -ar 48000 -ac 2 stop.wav
