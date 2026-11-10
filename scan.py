"""Scan a folder of audio files and report their basic properties."""

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# File endings we treat as audio. Everything else is skipped.
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"}


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


def estimate_tempo(path):
    """Estimate the tempo of one file, in beats per minute (BPM).

    Unlike describe(), this has to decode and process the whole file,
    so it is slower.
    """
    # Load the audio as mono. librosa also resamples to 22050 Hz by
    # default, which is plenty for tempo detection and quick to process.
    y, sr = librosa.load(path, mono=True)
    # beat_track returns a (tempo, beat_frames) tuple; we only need the
    # tempo, so the frames go into the throwaway name _beat_frames.
    bpm, _beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # Depending on the librosa version, the tempo is either a plain
    # float or wrapped in a one-element NumPy array. .item() unwraps
    # both into a plain Python float.
    return np.asarray(bpm).item()


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

        try:
            bpm = estimate_tempo(path)
            print(f"  tempo:       {bpm:.1f} BPM")
        except Exception as error:
            # Broad catch on purpose: one unanalyzable file should not
            # abort the whole scan.
            print(f"  tempo:       could not estimate ({error})")

        print()


if __name__ == "__main__":
    # Use the folder given on the command line, or audio/ by default.
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("audio")
    main(folder)
