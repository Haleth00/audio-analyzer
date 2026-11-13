"""Tests for practice-track DSP and its job endpoint."""

from pathlib import Path

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


def practice_client(monkeypatch, calls):
    def fake_separate_file(path, out_dir):
        calls.append(("separate",))
        for name in ("vocals", "drums", "bass", "other", "no_vocals"):
            sf.write(out_dir / f"{name}.flac",
                     np.zeros(2205, dtype=np.float32), 22050)

    def recording_practice_file(path, speed=1.0, semitones=0):
        calls.append(("practice", Path(path).name, speed, semitones))
        return fake_practice_file(path, speed, semitones)

    monkeypatch.setattr(server.practice, "practice_file",
                        recording_practice_file)
    monkeypatch.setattr(server.separation, "separate_file",
                        fake_separate_file)
    return TestClient(server.app)


def test_practice_job_flow(tmp_path, monkeypatch):
    calls = []
    client = practice_client(monkeypatch, calls)

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
    assert job["stem"] == "mix"

    state = client.get(f"/practice/{job['job_id']}").json()
    assert state["status"] == "done"
    track = client.get(f"/practice/{job['job_id']}/track")
    assert track.status_code == 200
    assert track.headers["content-type"] == "audio/flac"
    # Full-mix practice never touches Demucs.
    assert calls == [("practice", "input.flac", 0.75, -2)]


def test_practice_on_a_stem_runs_the_chain(tmp_path, monkeypatch):
    calls = []
    client = practice_client(monkeypatch, calls)

    wav = make_song(tmp_path / "upload.flac", seconds=1.0)
    with open(wav, "rb") as f:
        response = client.post(
            "/practice?speed=0.5&stem=bass",
            files={"file": ("song.flac", f, "audio/flac")},
        )
    assert response.status_code == 202
    state = client.get(f"/practice/{response.json()['job_id']}").json()
    assert state["status"] == "done"
    assert calls == [("separate",), ("practice", "bass.flac", 0.5, 0)]


def test_practice_rejects_bad_ranges(monkeypatch):
    client = practice_client(monkeypatch, [])
    import io
    assert client.post(
        "/practice?speed=0.1",
        files={"file": ("s.flac", io.BytesIO(b"x"), "audio/flac")},
    ).status_code == 400
    assert client.post(
        "/practice?semitones=24",
        files={"file": ("s.flac", io.BytesIO(b"x"), "audio/flac")},
    ).status_code == 400
    assert client.post(
        "/practice?stem=triangle",
        files={"file": ("s.flac", io.BytesIO(b"x"), "audio/flac")},
    ).status_code == 400
