"""Shared analysis functions for the audio analyzer.

Both the command-line scanner (scan.py) and the web service (server.py)
import from here, so there is exactly one implementation of each
analysis.
"""

from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
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

# Chord vocabulary as pitch-class offsets from the root: a major triad
# is root + major third (4 semitones) + fifth (7 semitones); a minor
# triad uses the minor third (3 semitones). "" names C major as "C",
# "m" names A minor as "Am".
CHORD_TYPES = [
    ("", [0, 4, 7]),
    ("m", [0, 3, 7]),
]

# Below this similarity a beat counts as "no chord" (N/C) -- silence,
# percussion-only, or too ambiguous to call.
CHORD_CONFIDENCE_FLOOR = 0.6

# Rough number of sections to split a song into. More would give
# intro/verse1/verse2 granularity but risks splitting true sections.
STRUCTURE_SECTIONS = 6


def find_audio_files(folder):
    """Return a sorted list of audio files in `folder` (not its subfolders)."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


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


def _cosine(a, b):
    """Cosine similarity of two vectors: 1.0 = same direction."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _best_chord(chroma_snapshot):
    """Match one beat's chroma snapshot against all 24 triads.

    Returns (name, similarity); name is "N/C" when nothing clears the
    confidence floor.
    """
    best_name, best_score = "N/C", CHORD_CONFIDENCE_FLOOR
    for step, root in enumerate(PITCH_CLASSES):
        for suffix, offsets in CHORD_TYPES:
            template = np.zeros(12)
            template[[(step + offset) % 12 for offset in offsets]] = 1.0
            score = _cosine(chroma_snapshot, template)
            if score > best_score:
                best_score, best_name = score, f"{root}{suffix}"
    return best_name, round(best_score, 2)


def estimate_chords(y, sr):
    """Estimate the chord progression of a mono signal.

    One chroma snapshot per musical beat (median over the beat, which
    ignores fast wiggles inside it), each matched against major/minor
    triad templates; consecutive equal chords merge into segments.

    Returns {"progression": "Cm - Ab - ...", "segments": [
        {"chord", "start", "end", "confidence"}, ...]}.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    duration = len(y) / sr

    _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_frames = np.asarray(beat_frames)
    if beat_frames.size < 4:
        # No usable pulse (pads, ambient, speech): one window per
        # second instead of one per beat. astype(int) because frame
        # indices must be integers even when the hop gives fractional
        # frames per second.
        frames_per_second = chroma.shape[1] / duration
        beat_frames = np.arange(
            0, chroma.shape[1], frames_per_second
        ).astype(int)

    # Boundaries cover the whole file even if beats start late or end
    # early; librosa.util.sync makes one column per window between
    # consecutive boundaries.
    boundaries = np.concatenate([[0], beat_frames, [chroma.shape[1]]])
    synced = librosa.util.sync(chroma, boundaries, aggregate=np.median)
    times = librosa.frames_to_time(boundaries, sr=sr)

    segments = []
    for i in range(synced.shape[1]):
        chord, confidence = _best_chord(synced[:, i])
        segments.append({
            "chord": chord,
            "start": round(float(times[i]), 1),
            "end": round(float(times[i + 1]), 1),
            "confidence": confidence,
        })

    # Squash runs of the same chord into one segment.
    merged = []
    for segment in segments:
        if merged and merged[-1]["chord"] == segment["chord"]:
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(segment)

    return {
        "progression": " - ".join(s["chord"] for s in merged),
        "segments": merged,
    }


def _frame_mean(feature, frame_from, frame_to):
    """Mean of a frame-wise feature between two frame indices.

    frame_to=None means "through to the end" (the song's last section
    has no next boundary to stop at).
    """
    hi = (feature.shape[0] if frame_to is None
          else min(int(frame_to), feature.shape[0]))
    lo = min(int(frame_from), max(hi - 1, 0))
    return float(feature[lo:hi].mean()) if hi > lo else 0.0


def estimate_structure(y, sr):
    """Split a song into sections (intro/verse/chorus-like blocks).

    Agglomerative clustering over beat-synchronous MFCCs: stretches of
    beats with similar timbre merge together, and the points where the
    cluster label changes become section boundaries. Naming the
    sections is human judgment -- so each one is reported with
    measurements (duration, mean energy, spectral brightness) that let
    you tell the quiet intro from the loud chorus yourself.

    Returns a list of section dicts, or None when there are too few
    beats to segment meaningfully.
    """
    duration = len(y) / sr
    _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_frames = np.asarray(beat_frames)
    if beat_frames.size < STRUCTURE_SECTIONS:
        # Too few beats (pads, speech, very short clips): splitting
        # into sections would be meaningless.
        return None

    # Boundaries include 0 so the first beat window isn't lost.
    beat_bounds = np.concatenate([[0], beat_frames])
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_beat = librosa.util.sync(mfcc, beat_bounds, aggregate=np.median)

    # Boundary indices land on the beat (column) axis; converting back
    # through beat_bounds maps them to real frame indices -> seconds.
    section_bounds = [0] + [int(b) for b in
                            librosa.segment.agglomerative(
                                mfcc_beat, STRUCTURE_SECTIONS)
                            if int(b) > 0]

    # Frame-wise loudness and brightness, to describe each section.
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    sections = []
    for i, start_col in enumerate(section_bounds):
        end_col = (section_bounds[i + 1]
                   if i + 1 < len(section_bounds) else None)
        start_frame = beat_bounds[start_col]
        end_frame = (beat_bounds[end_col]
                     if end_col is not None else None)
        start_t = float(librosa.frames_to_time(start_frame, sr=sr))
        end_t = (float(librosa.frames_to_time(end_frame, sr=sr))
                 if end_frame is not None else duration)
        sections.append({
            "label": f"Section {i + 1}",
            "start": round(start_t, 1),
            "end": round(end_t, 1),
            "duration_s": round(end_t - start_t, 1),
            "mean_energy": round(_frame_mean(rms, start_frame,
                                             end_frame), 4),
            "brightness_hz": round(_frame_mean(centroid, start_frame,
                                               end_frame), 0),
        })

    # Tag the extremes: often already tells you where the chorus is.
    if len(sections) > 1:
        max(sections, key=lambda s: s["mean_energy"])["label"] += " (loudest)"
        min(sections, key=lambda s: s["mean_energy"])["label"] += " (quietest)"
        max(sections, key=lambda s: s["brightness_hz"])["label"] += \
            " (brightest)"
    return sections


def estimate_loudness(y, sr):
    """Perceived loudness in LUFS (EBU R128 integrated measure).

    Unlike plain RMS this is perceptual: the signal is filtered the
    way human hearing weights frequencies before integrating. It's
    the number streaming services normalize to -- Spotify and YouTube
    target roughly -14 LUFS.
    """
    meter = pyln.Meter(sr)
    return round(float(meter.integrated_loudness(y)), 1)


def analyze_file(path):
    """Analyze one audio file and return the results as a dict.

    Header info (duration, sample rate, channels) plus tempo and key.
    If an individual analysis fails, its value is None rather than
    aborting the whole report. Raises if the file can't be read at all.
    """
    info = sf.info(path)
    result = {
        "file": path.name,
        "duration_s": round(info.duration, 1),
        "sample_rate_hz": info.samplerate,
        "channels": info.channels,
    }

    try:
        # Decode once; both analyses below work on the same samples.
        # librosa loads mono and resamples to 22050 Hz by default,
        # which is plenty for tempo and key detection.
        y, sr = librosa.load(path, mono=True)
    except Exception:
        result["tempo_bpm"] = None
        result["key"] = None
        result["key_confidence"] = None
        return result

    try:
        result["tempo_bpm"] = round(estimate_tempo(y, sr), 1)
    except Exception:
        result["tempo_bpm"] = None

    try:
        key, confidence = estimate_key(y, sr)
        result["key"] = key
        result["key_confidence"] = round(confidence, 2)
    except Exception:
        result["key"] = None
        result["key_confidence"] = None

    try:
        chords = estimate_chords(y, sr)
        result["chord_progression"] = chords["progression"]
        result["chord_segments"] = chords["segments"]
    except Exception:
        result["chord_progression"] = None
        result["chord_segments"] = None

    try:
        result["sections"] = estimate_structure(y, sr)
    except Exception:
        result["sections"] = None

    try:
        result["loudness_lufs"] = estimate_loudness(y, sr)
    except Exception:
        result["loudness_lufs"] = None

    return result
