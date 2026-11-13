"""Compare a mix against a reference track.

The checks a mastering engineer runs first: how loud, where the
energy sits across the spectrum, how dynamic. Every number is
relative to the reference -- the tool never says "good", it says
"compared to your reference, your mix is X".
"""

import librosa
import numpy as np
import pyloudnorm as pyln

BANDS = [
    ("sub", 20, 60),
    ("bass", 60, 250),
    ("low-mid", 250, 500),
    ("mid", 500, 2000),
    ("high-mid", 2000, 6000),
    ("high", 6000, 20000),
]


def _measure(path):
    """Loudness, true-ish peak and crest factor of one file."""
    y, sr = librosa.load(path, sr=None, mono=True)
    meter = pyln.Meter(sr)
    peak_dbfs = 20 * np.log10(np.max(np.abs(y)) + 1e-9)
    rms_db = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)
    return y, sr, {
        "lufs": round(float(meter.integrated_loudness(y)), 1),
        "peak_dbfs": round(float(peak_dbfs), 1),
        "crest_db": round(float(peak_dbfs - rms_db), 1),
    }


def _band_shape_db(y, sr):
    """Band energies in dB, normalized to the average band level.

    Subtracting the mean removes overall loudness, so what remains is
    the spectral *shape* -- a comparison then says "heavier bass",
    not just "louder".
    """
    spectrum = np.abs(librosa.stft(y)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    bands = []
    for _name, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        energy = float(np.mean(spectrum[mask])) if mask.any() else 1e-12
        bands.append(10 * np.log10(energy + 1e-12))
    bands = np.array(bands)
    return bands - bands.mean()


def compare_files(mix_path, reference_path):
    """Compare a mix against a reference; returns measurements, the
    per-band balance deltas (mix - reference, dB) and plain-language
    notes for anything that stands out."""
    y_mix, sr_mix, mix = _measure(mix_path)
    y_ref, sr_ref, ref = _measure(reference_path)

    shape_delta = _band_shape_db(y_mix, sr_mix) - _band_shape_db(y_ref, sr_ref)
    bands = [
        {"name": name, "from": lo, "to": hi,
         "delta_db": round(float(d), 1)}
        for (name, lo, hi), d in zip(BANDS, shape_delta)
    ]

    notes = []
    delta_lufs = mix["lufs"] - ref["lufs"]
    if abs(delta_lufs) >= 1.0:
        word = "louder" if delta_lufs > 0 else "quieter"
        notes.append(f"your mix is {abs(delta_lufs):.1f} dB {word} "
                     f"than the reference")
    for band in bands:
        if abs(band["delta_db"]) >= 3.0:
            lean = "heavier" if band["delta_db"] > 0 else "lighter"
            notes.append(f"{band['name']} is {abs(band['delta_db']):.1f} dB "
                         f"{lean} than the reference")
    delta_crest = mix["crest_db"] - ref["crest_db"]
    if abs(delta_crest) >= 2.0:
        word = "more dynamic" if delta_crest > 0 else "more compressed"
        notes.append(f"your mix reads {word} than the reference "
                     f"(crest {delta_crest:+.1f} dB)")
    if not notes:
        notes.append("no meaningful differences against the reference")

    return {
        "mix": mix,
        "reference": ref,
        "delta_lufs": round(delta_lufs, 1),
        "bands": bands,
        "notes": notes,
    }
