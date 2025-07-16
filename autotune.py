import numpy as np
import librosa

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
