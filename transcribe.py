"""Lyrics transcription with OpenAI's Whisper models.

Same pattern as separation.py: the heavy import and model load are lazy
and cached, so importing this module stays cheap for tests and the
slim Docker image.
"""

import functools


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


def transcribe_file(path, model_name="small"):
    """Transcribe an audio file; returns dict with text and language.

    Fair warning baked into the design: Whisper was trained on speech,
    not singing over instruments -- expect usable but imperfect lyrics
    on real songs.
    """
    model = get_model(model_name)
    result = model.transcribe(str(path))
    return {
        "text": result["text"].strip(),
        "language": result["language"],
    }
