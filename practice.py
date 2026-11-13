"""Practice-track DSP: slow down without changing pitch, or transpose.

librosa's phase vocoder (time_stretch) and pitch shift -- classic,
free, offline. Quality note: phase-vocoder artifacts grow below
~0.6x speed and drum transients soften; fine for learning parts.
"""

import librosa
import numpy as np

SPEED_RANGE = (0.5, 1.5)
SEMITONE_RANGE = (-12, 12)


def practice_file(path, speed=1.0, semitones=0):
    """Process a file for practice; returns (samples, sample_rate).

    speed < 1 slows the song down (0.75 = 75% speed) at unchanged
    pitch; semitones transposes (−12..+12) at unchanged tempo. Both
    keep stereo and the file's native sample rate.
    """
    y, sr = librosa.load(path, sr=None, mono=False)

    if semitones:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    if speed != 1.0:
        y = librosa.effects.time_stretch(y, rate=speed)

    # (channels, frames) -> (frames, channels) for soundfile.
    return np.ascontiguousarray(y).T, sr
