import streamlit as st
import librosa
import numpy as np
# import tensorflow as tf
from tensorflow.keras.models import load_model
import sounddevice as sd
import matplotlib.pyplot as plt
import tempfile
import soundfile as sf
import librosa.display

# Set Streamlit to dark theme
st.set_page_config(page_title="echo.ai", page_icon=":microphone:", layout="wide", initial_sidebar_state="collapsed")

# Load the pre-trained model
model_path = "models/model.h5"  # Change this path if needed
model = load_model(model_path)

# Genre mapping
genre_mapping = {0: "spoof", 1: "bonafide"}

# Function to preprocess audio and make predictions
def predict_voice(model, audio_file_path, genre_mapping):
    signal, sample_rate = librosa.load(audio_file_path, sr=22050)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T

    # Resize MFCC to match the expected input shape
    mfcc_resized = np.resize(mfcc, (130, 13, 1))  # Resize MFCC to fit model input shape

    # Reshape for model input (batch_size, time_steps, features)
    mfcc_resized = mfcc_resized[np.newaxis, ...]

    # Make prediction
    prediction = model.predict(mfcc_resized)
    predicted_index = np.argmax(prediction, axis=1)

    genre_label = genre_mapping[predicted_index[0]]
    return genre_label, signal, sample_rate, mfcc

# Function to record audio
def record_audio(duration=5, sr=22050):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten(), sr

# Streamlit UI
st.title("echo.ai")
st.write("Upload a FLAC audio file, or record your voice for prediction.")

# Initialize audio as None (for safety, in case no file is uploaded or recorded)
audio = None
uploaded_file = None

# File uploader (only accepts FLAC files)
uploaded_file = st.file_uploader("Choose a FLAC audio file", type=["flac"])

# Record button
if st.button("Record Audio"):
    audio, sr = record_audio(duration=5)
    st.success("Recording complete!")
    file_type = "recording"
else:
    if uploaded_file:
        file_type = "upload"
    else:
        st.warning("Please upload a FLAC file or record audio.")
        file_type = None

if audio is not None or uploaded_file is not None:
    # Process the audio depending on whether it was uploaded or recorded
    if file_type == "recording":
        # Save the recorded audio to a temporary file using tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_audio_path = temp_file.name
            sf.write(temp_audio_path, audio, sr)
        
        # Predict the voice
        predicted_label, signal, sample_rate, mfcc = predict_voice(model, temp_audio_path, genre_mapping)

    elif file_type == "upload":
        # If FLAC file is uploaded directly, save it to a temporary file using tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as temp_file:
            temp_audio_path = temp_file.name
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

        # Predict the voice
        predicted_label, signal, sample_rate, mfcc = predict_voice(model, temp_audio_path, genre_mapping)

    st.subheader("Model Prediction")
    st.markdown(f" predicted label: <span style='font-size: 30px; color: red;'>{predicted_label}</span>", unsafe_allow_html=True)



    # Visualization (Side-by-side Layout)
    st.subheader("Audio Visualizations")

    # Create 3 columns
    col1, col2, col3 = st.columns(3)

    # Waveform Visualization
    with col1:
        st.write("Waveform")
        plt.figure(figsize=(10, 5.4))
        librosa.display.waveshow(signal, sr=sample_rate)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        st.pyplot(plt)

    # MFCC Visualization
    with col2:
        st.write("MFCC")
        plt.figure(figsize=(10, 4.5))
        librosa.display.specshow(mfcc.T, x_axis='time', sr=sample_rate)
        plt.colorbar()
        plt.title('MFCC')
        st.pyplot(plt)

    # Spectrogram Visualization
    with col3:
        st.write("Spectrogram")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sample_rate)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        st.pyplot(plt)
