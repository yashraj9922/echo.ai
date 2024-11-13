# Audio Spoof Detection Dashboard (echo.ai)

## Description
This project detects whether an audio file is spoofed or bonafide using three fine-tuned models: ResNet, LSTM, and HuBERT.
This Project is built on Python3.10.0

## Features
- Upload MP3 files (auto-converted to FLAC).
- Record audio on-the-spot.
- Visualizations: Waveform, MFCC, and Spectrogram.

## Models
- resnet101.pt
- wav2vec2.pt (bi-lstm)
- HuBERT.pt
- wavLM.pt

## Setup

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd echo.ai
