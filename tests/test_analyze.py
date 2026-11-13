"""Tests for analyze.py.

No audio files are shipped: everything is synthesized. A melody built
from a scale's notes has a known key, and a click track has a known
tempo -- so the tests generate their own ground truth.
"""

import numpy as np
import soundfile as sf

from analyze import (analyze_file, estimate_chords, estimate_key,
                     estimate_loudness, estimate_structure,
                     estimate_tempo)

SR = 22050

# Note names -> frequencies in Hz (octave 4).
NOTES = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
    "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88,
}


def tone(freq, duration=0.6, amplitude=0.5):
    """One steady tone, with short fades so it doesn't click."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    fade = int(SR * 0.01)
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    return wave


def melody(note_names):
    """Concatenate one tone per note name."""
    return np.concatenate([tone(NOTES[name]) for name in note_names])


def triad(root, kind="", duration=2.0):
    """Stacked tones forming one chord: triad("C") is C major,
    triad("A", "m") is A minor. Ground truth for chord detection."""
    offsets = [0, 4, 7] if kind == "" else [0, 3, 7]
    return sum(
        tone(NOTES[root] * 2 ** (offset / 12), duration, amplitude=0.3)
        for offset in offsets
    )


def chord(root, suffix, duration=2.0):
    """A chord from the full vocabulary by its suffix, matching
    CHORD_TYPES: chord("C", "maj7"), chord("G", "7"), ..."""
    from analyze import CHORD_TYPES
    offsets = dict(CHORD_TYPES)[suffix]
    return sum(
        tone(NOTES[root] * 2 ** (offset / 12), duration, amplitude=0.3)
        for offset in offsets
    )


def clicks(bpm, seconds=8.0):
    """A click track: a short 1 kHz burst on every beat."""
    out = np.zeros(int(SR * seconds))
    burst = tone(1000.0, duration=0.02, amplitude=0.8)
    for beat in np.arange(0, seconds - 0.1, 60.0 / bpm):
        start = int(beat * SR)
        out[start:start + len(burst)] += burst
    return out


# Tonic, fifth and third get the most repeats -- that emphasis is what
# the Krumhansl-Kessler profiles encode, so the melodies mirror it.
C_MAJOR = ["C", "G", "E", "C", "G", "C",
           "F", "D", "A", "B", "E", "G", "C", "C"]
A_MINOR = ["A", "E", "C", "A", "E", "A",
           "D", "F", "G", "B", "C", "E", "A", "A"]


def test_key_of_major_melody():
    key, confidence = estimate_key(melody(C_MAJOR), SR)
    assert key == "C major"
    assert confidence > 0.5


def test_key_of_minor_melody():
    key, _confidence = estimate_key(melody(A_MINOR), SR)
    assert key == "A minor"


def test_tempo_of_click_track():
    bpm = estimate_tempo(clicks(120), SR)
    assert abs(bpm - 120) < 3


def test_chord_progression():
    y = np.concatenate([triad("C"), triad("A", "m"),
                        triad("F"), triad("G")])
    result = estimate_chords(y, SR)
    names = [s["chord"] for s in result["segments"]]
    assert names == ["C", "Am", "F", "G"]
    assert result["segments"][0]["confidence"] > 0.8


def test_every_chord_type_is_detected():
    # One representative per chord type in the vocabulary.
    samples = [
        ("C", ""), ("G", "m"), ("D", "dim"), ("A", "aug"),
        ("F", "sus2"), ("E", "sus4"),
        ("C", "6"), ("G", "m6"), ("D", "7"), ("A", "maj7"),
        ("F", "m7"), ("E", "mMaj7"), ("C", "m7b5"), ("G", "dim7"),
    ]
    for root, suffix in samples:
        y = chord(root, suffix)
        result = estimate_chords(y, SR)
        detected = result["segments"][0]["chord"]
        assert detected == f"{root}{suffix}", \
            f"{root}{suffix} detected as {detected}"


def structured_song():
    """A 12 s song with three timbre-distinct sections over a steady
    click track: dark tone (0-4 s), bright tone (4-8 s), white noise
    (8-12 s). True section boundaries sit at exactly 4 s and 8 s."""
    click_track = clicks(120, seconds=12.0)
    dark = tone(220.0, duration=4.0, amplitude=0.3)
    bright = tone(1760.0, duration=4.0, amplitude=0.3)
    noise = 0.1 * np.random.default_rng(42).normal(0, 1.0, 4 * SR)
    return np.concatenate([dark, bright, noise]) + click_track


def test_structure_boundaries():
    sections = estimate_structure(structured_song(), SR)
    assert sections is not None
    # The real transitions at 4 s and 8 s must appear among the
    # detected boundaries (within tolerance).
    starts = [s["start"] for s in sections]
    assert any(abs(t - 4.0) < 1.5 for t in starts)
    assert any(abs(t - 8.0) < 1.5 for t in starts)
    # And the first section is measurably darker than the second.
    assert sections[0]["brightness_hz"] < sections[1]["brightness_hz"]


def test_loudness_tracks_amplitude():
    # LUFS is perceptual, so exact values aren't derivable by hand --
    # but the same tone played 18x louder must measure far louder.
    loud = tone(440.0, duration=3.0, amplitude=0.9)
    quiet = tone(440.0, duration=3.0, amplitude=0.05)
    assert estimate_loudness(loud, SR) > estimate_loudness(quiet, SR) + 10


def test_analyze_file_end_to_end(tmp_path):
    """analyze_file() on a written WAV: header fields + analyses."""
    path = tmp_path / "song.wav"
    y = melody(C_MAJOR)
    sf.write(path, y, SR)

    result = analyze_file(path)
    assert result["file"] == "song.wav"
    assert result["sample_rate_hz"] == SR
    assert result["channels"] == 1
    assert abs(result["duration_s"] - len(y) / SR) < 0.1
    assert result["key"] == "C major"
    assert result["tempo_bpm"] is not None
    assert result["chord_progression"] is not None
    assert result["loudness_lufs"] is not None
