import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal
import noisereduce as nr

# === SETTINGS ===
DURATION = 170  # seconds
FS = 44100
HP_CUTOFF = 100  # Hz
LP_CUTOFF = 8000  # Hz
NOTCH_FREQ = 50   # Hz

# === FILTER FUNCTIONS ===
def butter_highpass(cutoff, fs, order=4):
    return signal.butter(order, cutoff / (0.5 * fs), btype='high', output='sos')

def butter_lowpass(cutoff, fs, order=4):
    return signal.butter(order, cutoff / (0.5 * fs), btype='low', output='sos')

def notch_filter(freq, fs, Q=30):
    b, a = signal.iirnotch(freq, Q, fs)
    return b, a

# === EFFECT FUNCTIONS ===
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

def apply_autotune(audio, sr, scale_notes=["C", "D", "E", "F", "G", "A", "B"]):
    if len(audio) < 4096 or np.max(np.abs(audio)) < 0.01:
        print("⚠️ Skipping autotune: audio too short or silent.")
        return audio
    try:
        f0, _, _ = librosa.pyin(audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    except Exception as e:
        print("⚠️ pyin failed:", e)
        return audio
    if f0 is None or np.all(np.isnan(f0)):
        print("⚠️ No pitch detected, skipping autotune.")
        return audio

    audio_tuned = audio.copy()
    scale_semitones = librosa.note_to_midi(scale_notes)
    frame_length = 2048
    hop_length = 512

    for i, pitch in enumerate(f0):
        if pitch is not None and not np.isnan(pitch):
            current_note = librosa.hz_to_midi(pitch)
            nearest_note = min(scale_semitones, key=lambda x: abs(x - current_note))
            shift_steps = nearest_note - current_note
            start = i * hop_length
            end = start + frame_length
            if end <= len(audio_tuned):
                audio_tuned[start:end] = librosa.effects.pitch_shift(audio_tuned[start:end], sr=sr, n_steps=shift_steps)
    return audio_tuned / np.max(np.abs(audio_tuned))

# === PLOT FUNCTIONS ===
def plot_waveform(wave1, wave2, title, label1, label2, samples=2000):
    plt.figure(figsize=(10, 3))
    plt.plot(wave1[:samples], label=label1, alpha=0.7)
    plt.plot(wave2[:samples], label=label2, alpha=0.7)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio, sr, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_time_domain(audio, sr, title):
    t = np.arange(len(audio)) / sr
    plt.figure(figsize=(10, 3))
    plt.plot(t, audio, alpha=0.7)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_frequency_domain(audio, sr, title):
    n = len(audio)
    f = np.fft.rfftfreq(n, 1/sr)
    spectrum = np.abs(np.fft.rfft(audio)) / n
    plt.figure(figsize=(10, 3))
    plt.semilogx(f, 20 * np.log10(spectrum + 1e-8))  # log scale
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === USER INPUT ===
try:
    volume = float(input("🔊 Volume (0.0 to 3.0, default 1.0): ") or "1.0")
    echo_decay = float(input("🔁 Echo decay (0.0 to 1.0, default 0.4): ") or "0.4")
    reverb_amount = float(input("🌫️ Reverb amount (0.0 to 0.7, default 0.3): ") or "0.3")
    pitch_steps = float(input("🎼 Pitch shift in semitones (-5 to 5, default 0): ") or "0.0")
    autotune_enable = input("🎯 Enable autotune? (yes/no): ").strip().lower() == "yes"
except ValueError:
    print("❌ Invalid input! Using defaults.")
    volume, echo_decay, reverb_amount, pitch_steps = 1.0, 0.4, 0.3, 0.0
    autotune_enable = False

# === RECORD AUDIO ===
print("\n🎙️ Recording...")
audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()
sf.write("raw_recording.wav", audio, FS)
print("✅ Recorded and saved as 'raw_recording.wav'")

# === RAW PLOTS ===
plot_time_domain(audio, FS, "🎙️ Raw Audio - Time Domain")
plot_frequency_domain(audio, FS, "🎙️ Raw Audio - Frequency Domain")
plot_spectrogram(audio, FS, "🎙️ Raw Audio Spectrogram")

# === FILTERING ===
sos_hp = butter_highpass(HP_CUTOFF, FS)
filtered = signal.sosfilt(sos_hp, audio)

sos_lp = butter_lowpass(LP_CUTOFF, FS)
filtered = signal.sosfilt(sos_lp, filtered)

b_notch, a_notch = notch_filter(NOTCH_FREQ, FS)
filtered = signal.filtfilt(b_notch, a_notch, filtered)

# === NOISE REDUCTION ===
noise_clip = filtered[:int(0.5 * FS)]
denoised = nr.reduce_noise(y=filtered, y_noise=noise_clip, sr=FS, prop_decrease=0.92, stationary=False)

# === PRE-GAIN BOOST ===
PRE_GAIN = 2
processed = denoised * PRE_GAIN

# === AUTOTUNE ===
if autotune_enable:
    print("🎯 Applying autotune...")
    before_tune = processed.copy()
    processed = apply_autotune(processed, FS)
    try:
        f0_before, _, _ = librosa.pyin(before_tune, sr=FS, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_after, _, _ = librosa.pyin(processed, sr=FS, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        times = librosa.times_like(f0_before, sr=FS)
        plt.figure(figsize=(10, 4))
        plt.plot(times, f0_before, label='Before Autotune', alpha=0.6)
        plt.plot(times, f0_after, label='After Autotune', alpha=0.7)
        plt.title("🎯 Autotune - Pitch Tracks")
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (Hz)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("⚠️ Autotune plot failed:", e)

# === PITCH SHIFT ===
if pitch_steps != 0.0:
    try:
        before_pitch = processed.copy()
        processed = pitch_shift(processed, FS, pitch_steps)
        plot_waveform(before_pitch, processed, "🎼 Pitch Shift Effect", "Original Pitch", f"Shifted ({pitch_steps} semitones)")
    except Exception as e:
        print("⚠️ Pitch shift error:", e)

# === ECHO ===
if echo_decay > 0:
    before_echo = processed.copy()
    processed = add_echo(processed, FS, decay=echo_decay)
    plot_waveform(before_echo, processed, "🔁 Echo Effect", "Original", "With Echo")

# === REVERB ===
if reverb_amount > 0:
    before_reverb = processed.copy()
    processed = add_reverb(processed, FS, reverb_amount=reverb_amount)
    plot_waveform(before_reverb, processed, "🌫️ Reverb Effect", "Before Reverb", "With Reverb")

# === FINAL VOLUME AND CLIP ===
processed *= volume
processed = np.clip(processed, -1.0, 1.0)

# === FINAL OUTPUT ===
sf.write("processed_output.wav", processed, FS)
print("\n🎧 Playing processed audio...")
sd.play(processed, FS)
sd.wait()
print("✅ Processed file saved as 'processed_output.wav'")

# === PROCESSED PLOTS ===
plot_time_domain(processed, FS, "🎧 Processed Audio - Time Domain")
plot_frequency_domain(processed, FS, "🎧 Processed Audio - Frequency Domain")
plot_spectrogram(processed, FS, "🎧 Final Audio Spectrogram")
