"""Scan a folder of audio files and report their basic properties.

With --separate, additionally splits each file into vocals and
instrumental using the Demucs model, writing FLAC stems to separated/.
"""

import argparse
import functools
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


@functools.cache
def get_separator():
    """Load the Demucs model, at most once per run.

    Importing torch and loading the model weights are both slow and
    memory hungry, so they live here instead of at the top of the file:
    a plain scan never pays for them, and functools.cache makes sure
    they are only paid for once even when separating many files.
    """
    import torch
    from demucs.api import Separator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  loading Demucs on {device} (the first run also downloads "
          f"the model weights)...")
    return Separator(model="htdemucs", device=device)


def separate(path, out_dir):
    """Split one file into vocals and instrumental stems in out_dir.

    Demucs separates into four stems -- vocals, drums, bass, other --
    which sum back into the original mix. So "instrumental" here is
    simply the other three stems added together.
    """
    separator = get_separator()
    # The API returns (original_mix, stems) where stems is a dict like
    # {"drums": tensor, "bass": tensor, "other": tensor, "vocals":
    # tensor}, each of shape (channels, samples).
    _original, stems = separator.separate_audio_file(path)
    instrumental = stems["drums"] + stems["bass"] + stems["other"]

    out_dir.mkdir(parents=True, exist_ok=True)
    # soundfile wants (frames, channels); the stems are (channels,
    # frames), hence the transpose. Note Demucs resamples everything to
    # its own rate (44.1 kHz), even if the input was 48 kHz.
    sr = separator.samplerate
    sf.write(out_dir / "vocals.flac", stems["vocals"].cpu().numpy().T, sr)
    sf.write(out_dir / "no_vocals.flac", instrumental.cpu().numpy().T, sr)


def main(folder, do_separate):
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

        if do_separate:
            out_dir = Path("separated") / path.stem
            try:
                separate(path, out_dir)
                print(f"  separation:  {out_dir}/vocals.flac, no_vocals.flac")
            except Exception as error:
                print(f"  separation:  failed ({error})")

        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan a folder of audio files and report their properties."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="audio",
        type=Path,
        help="folder to scan (default: audio)",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="also split each file into vocals and instrumental (slow)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.folder, args.separate)
