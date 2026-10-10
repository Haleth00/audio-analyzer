from pathlib import Path
import sys

import soundfile as sf

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


def find_audio_files(folder):
    """Return a sorted list of audio files inside a folder."""
    files = []
    for path in Path(folder).iterdir():
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return sorted(files)


def describe(path):
    """Read a file's header and return its basic properties."""
    info = sf.info(path)
    return {
        "name": path.name,
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
    }


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "audio"

    for path in find_audio_files(folder):
        try:
            d = describe(path)
        except Exception as error:
            print(f"{path.name}: could not read ({error})")
            continue

        minutes, seconds = divmod(d["duration"], 60)
        print(
            f"{d['name']}: {int(minutes)}:{seconds:04.1f}, "
            f"{d['sample_rate']} Hz, {d['channels']} ch"
        )


if __name__ == "__main__":
    main()