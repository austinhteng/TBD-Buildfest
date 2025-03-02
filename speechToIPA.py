import pyaudio
import webrtcvad
import numpy as np
import torch
import librosa
import subprocess
import io
import wave
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from collections import deque

# Load Wav2Vec2 Model for IPA Transcription
model_name = "bookbot/wav2vec2-ljspeech-gruut"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Required for VAD and Wav2Vec2
FRAME_DURATION = 30  # in ms (10, 20, or 30ms)
CHUNK = int(RATE * FRAME_DURATION / 1000)

# Initialize PyAudio and VAD
audio = pyaudio.PyAudio()
vad = webrtcvad.Vad(2)  # Adjust aggressiveness (0-3)

# Open audio stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Listening for speech... (Press Ctrl+C to stop)")

buffer = deque(maxlen=int(0.5 * RATE / CHUNK))  # Small buffer for speech continuity
recording = False
word_frames = []
word_count = 1

def apply_ffmpeg_noise_suppression(audio_bytes):    # Note: Apply before or after vad?
    """Apply FFmpeg noise suppression using the 'afftdn' filter."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-af", "afftdn", "-f", "wav", "pipe:1"],
            input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        return result.stdout
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return audio_bytes  # Return original if FFmpeg fails

def audio_buffer_to_ipa(audio_buffer):
    """Processes an in-memory audio buffer and returns an IPA transcription."""
    # Apply FFmpeg noise suppression
    # denoised_audio = apply_ffmpeg_noise_suppression(audio_buffer)

    # Load denoised audio into numpy array
    with wave.open(io.BytesIO(audio_buffer), 'rb') as wf:
    # with wave.open(io.BytesIO(denoised_audio), 'rb') as wf:
        audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    
    speech = audio_data.astype(np.float32) / 32768.0  # Normalize PCM data
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    ipa_transcription = processor.batch_decode(predicted_ids)[0]
    
    return ipa_transcription

try:
    while True:
        frame = stream.read(CHUNK, exception_on_overflow=False)
        is_speech = vad.is_speech(frame, RATE)

        if is_speech:
            if not recording:
                print("Speech detected, recording...")
                recording = True
            word_frames.append(frame)
        else:
            if recording:
                print("Silence detected, processing word...")
                recording = False

                # Convert buffered frames to a single byte string (WAV format in memory)
                output_buffer = io.BytesIO()
                with wave.open(output_buffer, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(word_frames))


                filename = f"word_{word_count}.wav"
                wf = wave.open(filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(word_frames))
                wf.close()
                word_count += 1

                # Convert audio buffer to IPA transcription
                ipa_output = audio_buffer_to_ipa(output_buffer.getvalue())
                if (ipa_output != ""):
                    print(f"IPA Transcription: {ipa_output}")

                word_frames = []  # Reset word buffer

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
