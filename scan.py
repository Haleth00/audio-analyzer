"""Command-line interface: scan a folder of audio files and report
their basic properties.

With --separate, also splits each file into vocals and instrumental
using the Demucs model, writing FLAC stems to separated/.

The analyses themselves live in analyze.py, shared with the web
service (server.py).
"""

import argparse
import functools
from pathlib import Path

import soundfile as sf

from analyze import AUDIO_EXTENSIONS, analyze_file, find_audio_files


def print_result(result):
    """Print one analysis result dict in the scanner's format."""
    print(result["file"])
    print(f"  duration:    {result['duration_s']} s")
    print(f"  sample rate: {result['sample_rate_hz']} Hz")
    print(f"  channels:    {result['channels']}")
    if result["tempo_bpm"] is not None:
        print(f"  tempo:       {result['tempo_bpm']} BPM")
    else:
        print("  tempo:       could not estimate")
    if result["key"] is not None:
        print(f"  key:         {result['key']} "
              f"(correlation {result['key_confidence']})")
    else:
        print("  key:         could not estimate")


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
            result = analyze_file(path)
        except RuntimeError as error:
            # A file with an audio extension that soundfile can't open.
            print(f"{path.name}: could not read ({error})")
            print()
            continue
        print_result(result)

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
