import numpy as np
import librosa
import scipy.signal as signal

def add_echo(audio, sr, delay_ms=400, decay=0.5):
    delay_samples = int(sr * delay_ms / 1000)
    echo_audio = np.zeros(len(audio) + delay_samples, dtype=np.float32)
    echo_audio[:len(audio)] = audio
    echo_audio[delay_samples:] += decay * audio
    return echo_audio / np.max(np.abs(echo_audio))

def add_reverb(audio, sr, reverb_amount=0.3):
    delay_samples = int(sr * 0.05)
    b = np.zeros(delay_samples + 1)
    b[0], b[-1] = 1, reverb_amount
    reverb_audio = signal.lfilter(b, [1], audio)
    return reverb_audio / np.max(np.abs(reverb_audio))

def pitch_shift(audio, sr, steps):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
