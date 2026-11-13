"""Tests for the /separate job flow.

Demucs is never loaded: separation.separate_file is monkeypatched to
just write two tiny FLAC files, so these tests exercise the job
plumbing (upload -> job id -> poll -> download) in milliseconds.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import server


def fake_separate_file(path, out_dir):
    """Stand-in for Demucs: write the five short silent stems."""
    silence = np.zeros(22050)
    for name in ("vocals", "drums", "bass", "other", "no_vocals"):
        sf.write(out_dir / f"{name}.flac", silence, 22050)


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
    stems = status.json()["stems"]
    assert set(stems) == {"vocals", "instrumental", "drums", "bass",
                          "other"}
    assert stems["drums"] == f"/separate/{job['job_id']}/stems/drums"

    stem = client.get(f"/separate/{job['job_id']}/stems/drums")
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


def test_index_page_is_served(monkeypatch):
    client = make_client(monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    assert "Audio Analyzer" in response.text


def test_library_page_is_served(monkeypatch):
    client = make_client(monkeypatch)
    response = client.get("/library")
    assert response.status_code == 200
    assert "Library" in response.text


def test_library_api_and_detail(monkeypatch):
    client = make_client(monkeypatch)

    def fake_search(**kwargs):
        return [{"id": 1, "file": "song.wav", "tempo_bpm": 120.0,
                 "key": "C major", "loudness_lufs": -14.0,
                 "duration_s": 60.0, "analyzed_at": "2026-11-13 10:00"}]

    def fake_get_track(track_id):
        if track_id != 1:
            return None
        return {"id": track_id, "file": "song.wav",
                "result_json": '{"key": "C major"}'}

    monkeypatch.setattr(server.library, "search", fake_search)
    monkeypatch.setattr(server.library, "get_track", fake_get_track)

    listing = client.get("/api/library")
    assert listing.status_code == 200
    assert listing.json()["count"] == 1

    detail = client.get("/api/library/1")
    assert detail.status_code == 200
    assert "result_json" in detail.json()

    assert client.get("/api/library/999").status_code == 404


def test_capabilities_reports_separation(monkeypatch):
    client = make_client(monkeypatch)
    response = client.get("/capabilities")
    assert response.status_code == 200
    assert isinstance(response.json()["separation"], bool)
    assert isinstance(response.json()["transcription"], bool)


def fake_transcribe_file(path, model_name="small", denoise=False):
    return {"text": "la la la, la la la la", "language": "en"}


def test_transcription_job_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(server.transcribe, "transcribe_file",
                        fake_transcribe_file)
    client = TestClient(server.app)

    wav = tmp_path / "song.wav"
    sf.write(wav, np.zeros(22050), 22050)

    with open(wav, "rb") as f:
        response = client.post(
            "/transcribe", files={"file": ("song.wav", f, "audio/wav")}
        )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "pending"

    state = client.get(f"/transcribe/{job['job_id']}").json()
    assert state["status"] == "done"
    assert state["lyrics"] == "la la la, la la la la"
    assert state["language"] == "en"


def test_unknown_transcription_job_is_404(monkeypatch):
    client = make_client(monkeypatch)
    assert client.get("/transcribe/nope").status_code == 404


def transcription_client(monkeypatch, calls):
    """A client whose model calls just record what got routed where."""
    def fake_transcribe_file(path, model_name="small", denoise=False):
        calls.append(("transcribe", Path(path).name, denoise))
        return {"text": "la", "language": "en"}

    def fake_separate_file(path, out_dir):
        calls.append(("separate",))
        Path(out_dir, "vocals.flac").write_bytes(b"fake")

    monkeypatch.setattr(server.transcribe, "transcribe_file",
                        fake_transcribe_file)
    monkeypatch.setattr(server.separation, "separate_file",
                        fake_separate_file)
    return TestClient(server.app)


def upload(client, mode):
    import io
    wav = io.BytesIO(b"not really audio, the fakes never read it")
    return client.post(
        f"/transcribe?mode={mode}",
        files={"file": ("song.wav", wav, "audio/wav")},
    )


def test_transcription_mode_mix_skips_models(tmp_path, monkeypatch):
    calls = []
    response = upload(transcription_client(monkeypatch, calls), "mix")
    assert response.status_code == 202
    assert calls == [("transcribe", "input.wav", False)]


def test_transcription_mode_vocals_runs_the_chain(tmp_path, monkeypatch):
    calls = []
    response = upload(transcription_client(monkeypatch, calls), "vocals")
    assert response.status_code == 202
    # Separation first, then transcription of the vocal stem,
    # denoised.
    assert calls == [("separate",), ("transcribe", "vocals.flac", True)]


def test_transcription_mode_denoised(tmp_path, monkeypatch):
    calls = []
    response = upload(transcription_client(monkeypatch, calls),
                      "denoised")
    assert response.status_code == 202
    assert calls == [("transcribe", "input.wav", True)]


def test_transcription_bad_mode_is_400(monkeypatch):
    client = make_client(monkeypatch)
    assert upload(client, "sideways").status_code == 400
