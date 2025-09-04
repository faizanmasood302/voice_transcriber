import streamlit as st
import numpy as np
import tempfile
from scipy.io.wavfile import write
from transformers import pipeline
import os
import io

# Try to import sounddevice, fallback if not available
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False

# Try to import librosa for better audio handling
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

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

language = st.selectbox("Select transcription language", ["auto", "en", "ur", "hi", "fr", "es"])

# Check if we're running locally or on cloud
if SOUNDDEVICE_AVAILABLE:
    st.markdown("### 🎤 Real-time Voice Recording")
    duration = st.slider("🎚️ Recording Duration (seconds)", 1, 30, 5)
    record_btn = st.button("🔴 Record Voice")
    
    if record_btn:
        st.info("Recording... Speak now!")
        try:
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
        except Exception as e:
            st.error(f"Recording failed: {str(e)}")
            st.info("Please check your microphone permissions or try the file upload option below.")
    
    st.markdown("---")
    st.markdown("### 📁 Or Upload Audio File")
else:
    st.markdown("### 📁 Upload Audio File")
    st.info("💡 Real-time recording not available in this environment. Please upload an audio file.")

# File uploader for audio files (always available)
uploaded_file = st.file_uploader(
    "🎤 Upload an audio file (WAV, MP3, M4A, etc.)", 
    type=['wav', 'mp3', 'm4a', 'flac', 'ogg']
)

if uploaded_file is not None:
    st.success("✅ Audio file uploaded!")
    
    # Save uploaded file to temporary location
    tmp_file = tempfile.NamedTemporaryFile(suffix=f".{uploaded_file.name.split('.')[-1]}", delete=False)
    tmp_file.write(uploaded_file.getvalue())
    tmp_file.close()
    
    # Play the uploaded audio
    st.audio(tmp_file.name)
    
    # Transcribe button
    if st.button("🎯 Transcribe Audio"):
        st.markdown("#### ⏳ Transcribing...")
        
        try:
            kwargs = {} if language == "auto" else {"language": language}
            
            # Optimized approach - try direct first, then librosa if needed
            try:
                # Direct approach (fastest)
                transcription = transcriber(tmp_file.name, generate_kwargs=kwargs)["text"]
            except Exception as direct_error:
                # Only use librosa if direct approach fails
                if LIBROSA_AVAILABLE:
                    st.info("🔄 Direct approach failed, using librosa...")
                    audio_data, sr = librosa.load(tmp_file.name, sr=16000)
                    # Save as WAV for Whisper
                    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    write(wav_file.name, 16000, (audio_data * 32767).astype(np.int16))
                    transcription = transcriber(wav_file.name, generate_kwargs=kwargs)["text"]
                else:
                    raise direct_error
            
            st.markdown("#### 📝 Transcription")
            st.text_area("Transcription:", value=transcription, height=200)
            
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            st.info("💡 **Troubleshooting tips:**")
            st.info("• Try uploading a WAV file (best compatibility)")
            st.info("• Make sure the audio file is not corrupted")
            st.info("• Check that the file contains speech audio")
            st.info("• For MP3/M4A files, try converting to WAV first")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ ung Streamlit & OpenAI Whisper</p>", unsafe_allow_html=True)
