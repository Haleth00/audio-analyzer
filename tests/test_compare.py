"""Tests for reference-mix comparison."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import compare
import server

SR = 22050


def write_tone(path, freq, amplitude=0.5, seconds=3.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    sf.write(path, amplitude * np.sin(2 * np.pi * freq * t), SR)
    return path


def test_bass_heavy_mix_is_flagged(tmp_path):
    mix = write_tone(tmp_path / "mix.flac", freq=60)     # sub/bass
    ref = write_tone(tmp_path / "ref.flac", freq=440)    # mid

    result = compare.compare_files(mix, ref)
    by_name = {b["name"]: b["delta_db"] for b in result["bands"]}

    assert by_name["sub"] > 6        # mix energy lives far below
    assert by_name["mid"] < -6       # ...and the reference's doesn't
    assert any("heavier" in n for n in result["notes"])


def test_loudness_delta_sign(tmp_path):
    loud = write_tone(tmp_path / "loud.flac", 440, amplitude=0.9)
    quiet = write_tone(tmp_path / "quiet.flac", 440, amplitude=0.05)

    result = compare.compare_files(loud, quiet)
    assert result["delta_lufs"] > 10
    assert any("louder" in n for n in result["notes"])


def test_identical_tracks_are_quiet(tmp_path):
    a = write_tone(tmp_path / "a.flac", 440)
    b = write_tone(tmp_path / "b.flac", 440)
    result = compare.compare_files(a, b)
    assert result["notes"] == ["no meaningful differences against "
                               "the reference"]


def test_compare_endpoint(tmp_path):
    client = TestClient(server.app)

    def file_bytes(freq):
        t = np.linspace(0, 2.0, int(SR * 2), endpoint=False)
        buf = io.BytesIO()
        sf.write(buf, 0.4 * np.sin(2 * np.pi * freq * t), SR,
                 format="FLAC")
        buf.seek(0)
        return buf.read()

    response = client.post(
        "/compare",
        files={
            "mix": ("mix.flac", file_bytes(60), "audio/flac"),
            "reference": ("ref.flac", file_bytes(440), "audio/flac"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mix"]["file"] == "mix.flac"
    assert len(body["bands"]) == 6
    assert body["notes"]
