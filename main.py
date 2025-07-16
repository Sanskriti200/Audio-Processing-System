from effects import add_echo, add_reverb, pitch_shift
from filters import butter_highpass, butter_lowpass, notch_filter
from plots import plot_time_domain, plot_frequency_domain, plot_spectrogram, plot_waveform
from autotune import apply_autotune
import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.signal as signal
import noisereduce as nr

DURATION = 170
FS = 44100
HP_CUTOFF = 100
LP_CUTOFF = 8000
NOTCH_FREQ = 50

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

# === PROCESSING CHAIN ===
processed = denoised * 2

if autotune_enable:
    print("🎯 Applying autotune...")
    processed = apply_autotune(processed, FS)

if pitch_steps != 0.0:
    processed = pitch_shift(processed, FS, pitch_steps)

if echo_decay > 0:
    processed = add_echo(processed, FS, decay=echo_decay)

if reverb_amount > 0:
    processed = add_reverb(processed, FS, reverb_amount=reverb_amount)

processed *= volume
processed = np.clip(processed, -1.0, 1.0)

sf.write("processed_output.wav", processed, FS)
print("\n🎧 Playing processed audio...")
sd.play(processed, FS)
sd.wait()
print("✅ Processed file saved as 'processed_output.wav'")

# === FINAL PLOTS ===
plot_time_domain(processed, FS, "🎧 Processed Audio - Time Domain")
plot_frequency_domain(processed, FS, "🎧 Processed Audio - Frequency Domain")
plot_spectrogram(processed, FS, "🎧 Final Audio Spectrogram")
