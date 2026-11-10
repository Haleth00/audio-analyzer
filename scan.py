"""Scan a folder of audio files and report their basic properties."""

import sys
from pathlib import Path

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


if __name__ == "__main__":
    # Use the folder given on the command line, or audio/ by default.
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("audio")
    main(folder)
