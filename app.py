import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
from scipy.io.wavfile import write
from transformers import pipeline

# --- Custom Styles ---
st.markdown("""
    <style>
    .stTextArea textarea {font-size: 1.1em;}
    .stButton button {background-color: #4F8BF9; color: white;}
    .stSlider {padding-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

# Initialize Whisper ASR
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

st.markdown("<h1 style='text-align:center;'>🎤 Voice Recorder & Transcriber</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Record your voice and get instant transcription using OpenAI Whisper!</p>", unsafe_allow_html=True)
st.markdown("---")

duration = st.slider("🎚️ Recording Duration (seconds)", 1, 30, 5)
language = st.selectbox("Select transcription language", ["auto", "en", "ur", "hi", "fr", "es"])
record_btn = st.button("🔴 Record")

if record_btn:
    st.info("Recording... Speak now!")
    # Record audio: 16 kHz, mono, float32
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    st.success("✅ Recording finished!")

    # Normalize float32 to int16 PCM for proper WAV
    audio_int16 = np.int16(audio * 32767)

    # Save to temp WAV
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(tmp_file.name, 16000, audio_int16)

    st.audio(tmp_file.name)  # Play recording

    # Transcribe
    st.markdown("#### ⏳ Transcribing...")
    kwargs = {} if language == "auto" else {"language": language}
    transcription = transcriber(tmp_file.name, generate_kwargs=kwargs)["text"]
    st.markdown("#### 📝 Transcription")
    st.text_area("Transcription:", value=transcription, height=200)

st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit & OpenAI Whisper</p>", unsafe_allow_html=True)
