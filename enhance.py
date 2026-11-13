"""Speech denoising with DeepFilterNet (DeepFilterNet3).

Same lazy-and-cached pattern as the other model modules. Audio is fed
to the model as tensors with soundfile doing the IO -- df's own file
loading needs old torchaudio APIs, so we sidestep it entirely.
"""

import functools

import numpy as np
import soundfile as sf


@functools.cache
def get_model():
    """Load DeepFilterNet3, at most once per process.

    The first call downloads the weights (~100 MB) into the package's
    cache directory; after that it loads from disk.
    """
    import torch
    from df.enhance import init_df

    print("loading DeepFilterNet3...")
    # Returns (model, df_state, suffix, epoch); we need the first two.
    model, df_state, _suffix, _epoch = init_df()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, df_state


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


def enhance_file(path):
    """Denoise an audio file; returns (samples, sample_rate).

    DeepFilterNet works at 48 kHz internally, so input at another rate
    is resampled there; the returned samples are always 48 kHz.
    Designed for speech: on music it strips instruments as "noise",
    which is exactly what we want before transcribing a vocal stem.
    """
    import librosa
    import torch
    from df.enhance import enhance

    model, df_state = get_model()
    # always_2d -> (frames, channels) even for mono files.
    data, sr = sf.read(path, dtype="float32", always_2d=True)

    target_sr = df_state.sr()
    if sr != target_sr:
        # librosa.resample takes (channels, frames).
        data = librosa.resample(data.T, orig_sr=sr,
                                target_sr=target_sr).T

    audio = torch.from_numpy(data.T)  # -> (channels, frames)
    with torch.no_grad():
        enhanced = enhance(model, df_state, audio)
    return enhanced.numpy().T, target_sr  # -> (frames, channels)
