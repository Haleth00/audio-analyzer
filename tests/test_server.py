"""Tests for the /separate job flow.

Demucs is never loaded: separation.separate_file is monkeypatched to
just write two tiny FLAC files, so these tests exercise the job
plumbing (upload -> job id -> poll -> download) in milliseconds.
"""

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import server


def fake_separate_file(path, out_dir):
    """Stand-in for Demucs: write two very short silent stems."""
    silence = np.zeros(22050)
    sf.write(out_dir / "vocals.flac", silence, 22050)
    sf.write(out_dir / "no_vocals.flac", silence, 22050)


def make_client(monkeypatch):
    monkeypatch.setattr(server.separation, "separate_file",
                        fake_separate_file)
    return TestClient(server.app)


def test_separation_job_flow(tmp_path, monkeypatch):
    client = make_client(monkeypatch)

    wav = tmp_path / "song.wav"
    sf.write(wav, np.zeros(22050), 22050)

    # TestClient runs background tasks after building the response
    # body but before post() returns: so the POST's JSON is a snapshot
    # taken *before* the task ran (pending), and by the first poll the
    # task has already finished (done). A real client must never
    # assume "done" from the 202 alone -- that's the point of polling.
    with open(wav, "rb") as f:
        response = client.post(
            "/separate", files={"file": ("song.wav", f, "audio/wav")}
        )
    assert response.status_code == 202
    job = response.json()
    assert job["job_id"]
    assert job["status"] == "pending"

    status = client.get(f"/separate/{job['job_id']}")
    assert status.json()["vocals_url"] == \
        f"/separate/{job['job_id']}/stems/vocals"

    stem = client.get(f"/separate/{job['job_id']}/stems/vocals")
    assert stem.status_code == 200
    assert stem.headers["content-type"] == "audio/flac"


def test_unknown_job_is_404(monkeypatch):
    client = make_client(monkeypatch)
    assert client.get("/separate/nope").status_code == 404


def test_stem_of_unfinished_job_is_409(monkeypatch):
    client = make_client(monkeypatch)
    server.JOBS["fake"] = {"job_id": "fake", "status": "running"}
    response = client.get("/separate/fake/stems/vocals")
    assert response.status_code == 409
