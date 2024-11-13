import pyttsx3
from pydub import AudioSegment

# Initialize TTS engine
engine = pyttsx3.init()
text = "India is my country"
engine.save_to_file(text, "spoof_audio.flac")
# engine.save_to_file(text, "spoof_audio.mp3")
engine.runAndWait()
