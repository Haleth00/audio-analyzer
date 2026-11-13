"""Lyrics transcription with OpenAI's Whisper models.

Same pattern as separation.py: the heavy import and model load are lazy
and cached, so importing this module stays cheap for tests and the
slim Docker image.
"""

import functools
import tempfile
from pathlib import Path

import soundfile as sf


@functools.cache
def get_model(name="small"):
    """Load a Whisper model, at most once per process.

    The first run downloads the weights (~460 MB for "small") to
    ~/.cache/whisper; after that it loads from the cache.
    """
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading whisper {name!r} on {device}...")
    return whisper.load_model(name, device=device)


def park():
    """Free the GPU (see separation.park): forget the cached model,
    return the allocator's blocks. Reloads on next use."""
    get_model.cache_clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def transcribe_file(path, model_name="small", denoise=False):
    """Transcribe an audio file; returns dict with text and language.

    With denoise=True the audio is run through DeepFilterNet first --
    built for speech recordings (voice memos, lectures). For songs the
    better route is transcribing the isolated vocal stem, which the
    service's mode=vocals chains together.

    Fair warning baked into the design: Whisper was trained on speech,
    not singing over instruments -- expect usable but imperfect lyrics
    on real songs.
    """
    model = get_model(model_name)
    if denoise:
        import enhance

        samples, sr = enhance.enhance_file(path)
        # Whisper reads files (via ffmpeg), so park the enhanced
        # samples in a temporary flac.
        tmp = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
        try:
            with tmp:
                sf.write(tmp.name, samples, sr)
            result = model.transcribe(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    else:
        result = model.transcribe(str(path))
    return {
        "text": result["text"].strip(),
        "language": result["language"],
    }
