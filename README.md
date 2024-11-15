# Audio Spoof Detection Dashboard (echo.ai)

## Description
This project detects whether an audio file is spoofed or bonafide using fine-tuned model based on CNN.

## Team Members
- Yashraj Kadam (22bds066)
- Ayush Singh (22bds012)
- Nachiket Apte (22bds041)
- Harsh Raj (22bds027)

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
   streamlit run app/main.py
