"""Tests for practice-track DSP and its job endpoint."""

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import practice
import server


def make_song(path, seconds=2.0, sr=22050):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
    return path


def test_slower_means_longer_same_pitch(tmp_path):
    path = make_song(tmp_path / "song.flac")
    samples, sr = practice.practice_file(path, speed=0.5)
    # Half speed -> double the duration (within a frame or two).
    assert samples.shape[0] / sr == pytest.approx(4.0, abs=0.05)


def test_transpose_keeps_length(tmp_path):
    path = make_song(tmp_path / "song.flac")
    samples, sr = practice.practice_file(path, semitones=3)
    assert samples.shape[0] / sr == pytest.approx(2.0, abs=0.05)


def test_identity_changes_nothing(tmp_path):
    path = make_song(tmp_path / "song.flac")
    samples, sr = practice.practice_file(path)
    assert samples.shape[0] == int(22050 * 2.0)


def fake_practice_file(path, speed=1.0, semitones=0):
    return np.zeros((2205, 1), dtype=np.float32), 22050


def test_practice_job_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(server.practice, "practice_file",
                        fake_practice_file)
    client = TestClient(server.app)

    wav = make_song(tmp_path / "upload.flac", seconds=1.0)
    with open(wav, "rb") as f:
        response = client.post(
            "/practice?speed=0.75&semitones=-2",
            files={"file": ("song.flac", f, "audio/flac")},
        )
    assert response.status_code == 202
    job = response.json()
    assert job["speed"] == 0.75
    assert job["semitones"] == -2

    state = client.get(f"/practice/{job['job_id']}").json()
    assert state["status"] == "done"
    track = client.get(f"/practice/{job['job_id']}/track")
    assert track.status_code == 200
    assert track.headers["content-type"] == "audio/flac"


def test_practice_rejects_bad_ranges(monkeypatch):
    client = TestClient(server.app)
    import io
    assert client.post(
        "/practice?speed=0.1",
        files={"file": ("s.flac", io.BytesIO(b"x"), "audio/flac")},
    ).status_code == 400
    assert client.post(
        "/practice?semitones=24",
        files={"file": ("s.flac", io.BytesIO(b"x"), "audio/flac")},
    ).status_code == 400
