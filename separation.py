"""Vocal/instrumental separation with the Demucs model.

Shared by the CLI (scan.py) and the web service (server.py). torch and
demucs are imported lazily inside get_separator() so that importing
this module stays cheap -- the test suite and the slim Docker image
never pay for the model.
"""

import functools
from pathlib import Path

import soundfile as sf


@functools.cache
def get_separator():
    """Load the Demucs model, at most once per process.

    Importing torch and loading the model weights are both slow and
    memory hungry, so they live here instead of at the top of the file,
    and functools.cache makes sure they are only paid for once even
    when separating many files.
    """
    import torch
    from demucs.api import Separator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading Demucs on {device} (the first run also downloads "
          f"the model weights)...")
    return Separator(model="htdemucs", device=device)


def separate_file(path, out_dir):
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
