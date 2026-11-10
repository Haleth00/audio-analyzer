"""Scan a folder of audio files and report their basic properties."""

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# File endings we treat as audio. Everything else is skipped.
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"}

# The 12 pitch classes, starting at C.
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles: how strongly each pitch class tends to
# appear in a major or minor scale, counted upwards from the tonic. For
# example the major profile peaks on the tonic (6.35), then the fifth
# (5.19), then the third (4.38) -- exactly how a major scale is built.
KEY_TEMPLATES = [
    (
        "major",
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    ),
    (
        "minor",
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    ),
]


def find_audio_files(folder):
    """Return a sorted list of audio files in `folder` (not its subfolders)."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def describe(path):
    """Print the audio properties of one file.

    sf.info() only reads the file header, so this is fast even for
    long recordings.
    """
    info = sf.info(path)
    print(path.name)
    print(f"  duration:    {info.duration:.1f} s")
    print(f"  sample rate: {info.samplerate} Hz")
    print(f"  channels:    {info.channels}")


def estimate_tempo(y, sr):
    """Estimate the tempo of a mono signal, in beats per minute (BPM)."""
    # beat_track returns a (tempo, beat_frames) tuple; we only need the
    # tempo, so the frames go into the throwaway name _beat_frames.
    bpm, _beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # Depending on the librosa version, the tempo is either a plain
    # float or wrapped in a one-element NumPy array. .item() unwraps
    # both into a plain Python float.
    return np.asarray(bpm).item()


def estimate_key(y, sr):
    """Estimate the musical key of a mono signal.

    Uses the Krumhansl-Schmuckler method: measure how much energy the
    recording has in each of the 12 pitch classes (a chromagram), average
    that over time, and pick the major or minor scale profile that
    correlates best with the result.

    Returns (key name, correlation), e.g. ("A minor", 0.82).
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Average over time (axis 1 of the 12 x frames matrix) -> one energy
    # value per pitch class.
    profile = chroma.mean(axis=1)

    best_key, best_score = None, -2.0
    for step, tonic in enumerate(PITCH_CLASSES):
        for mode_name, template in KEY_TEMPLATES:
            # Rotate the template so it starts at this tonic, then see
            # how well it matches the song's pitch profile.
            score = np.corrcoef(profile, np.roll(template, step))[0, 1]
            if score > best_score:
                best_score, best_key = score, f"{tonic} {mode_name}"
    return best_key, float(best_score)


def main(folder):
    files = find_audio_files(folder)
    if not files:
        print(f"No audio files found in {folder}")
        return

    for path in files:
        try:
            describe(path)
        except RuntimeError as error:
            # A file with an audio extension that soundfile can't open.
            print(f"{path.name}: could not read ({error})")
            continue

        # Decode once; every analysis below works on the same samples.
        # librosa loads mono and resamples to 22050 Hz by default,
        # which is plenty for tempo and key detection.
        try:
            y, sr = librosa.load(path, mono=True)
        except Exception as error:
            print(f"  could not decode for analysis ({error})")
            print()
            continue

        try:
            bpm = estimate_tempo(y, sr)
            print(f"  tempo:       {bpm:.1f} BPM")
        except Exception as error:
            print(f"  tempo:       could not estimate ({error})")

        try:
            key, confidence = estimate_key(y, sr)
            print(f"  key:         {key} (correlation {confidence:.2f})")
        except Exception as error:
            print(f"  key:         could not estimate ({error})")

        print()


if __name__ == "__main__":
    # Use the folder given on the command line, or audio/ by default.
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("audio")
    main(folder)
