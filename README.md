# Audio Spoof Detection Dashboard (echo.ai)

## Description
This project detects whether an audio file is spoofed or bonafide using fine-tuned model based on CNN.


## Features
- Upload FLAC files.
- Record audio on-the-spot.
- Prediction (spoof or bonafide)
- Visualizations: Waveform, MFCC, and Spectrogram.

## Models
- resnet101
- wav2vec2 + bilstm
- HuBERT
- wavLM
- wavNet


## Setup
This Project is built on Python3.10.0
1. Clone the repository:
   ```bash
   git clone https://github.com/yashraj9922/echo.ai.git
   cd echo.ai
2. Set up Python Environment
3. ```bash
   pip install requirements.txt
   streamlit run main.py
