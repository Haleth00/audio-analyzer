"""Tests for the SQLite analysis library.

Uses a throwaway database per test (tmp_path), never the real one in
~/.local/share.
"""

import library


def make_result(file="song.wav", **overrides):
    result = {
        "file": file,
        "duration_s": 69.7,
        "sample_rate_hz": 48000,
        "channels": 2,
        "tempo_bpm": 123.0,
        "loudness_lufs": -16.0,
        "key": "C minor",
        "key_confidence": 0.76,
        "chord_progression": "Cm - Fm",
    }
    result.update(overrides)
    return result


def test_record_and_search(tmp_path):
    db = tmp_path / "lib.db"
    library.record(make_result("a.wav"), db)
    library.record(make_result("b.wav", key="A minor", tempo_bpm=90.0,
                               duration_s=100.0), db)

    assert len(library.search(db_path=db)) == 2
    only_c = library.search(key="C minor", db_path=db)
    assert len(only_c) == 1
    assert only_c[0]["file"] == "a.wav"


def test_rerecording_replaces_the_row(tmp_path):
    db = tmp_path / "lib.db"
    library.record(make_result(tempo_bpm=123.0), db)
    # Same file + duration -> replaces, doesn't duplicate.
    library.record(make_result(tempo_bpm=125.0), db)

    rows = library.search(db_path=db)
    assert len(rows) == 1
    assert rows[0]["tempo_bpm"] == 125.0


def test_bpm_range_filter(tmp_path):
    db = tmp_path / "lib.db"
    library.record(make_result("slow.wav", tempo_bpm=70.0,
                               duration_s=50.0), db)
    library.record(make_result("fast.wav", tempo_bpm=140.0,
                               duration_s=60.0), db)

    fast = library.search(bpm_min=100, db_path=db)
    assert [t["file"] for t in fast] == ["fast.wav"]


def test_full_result_survives_as_json(tmp_path):
    db = tmp_path / "lib.db"
    library.record(make_result(), db)
    # The list view drops result_json (it can be large); the detail
    # view fetches it by id.
    listing = library.search(db_path=db)[0]
    assert "result_json" not in listing
    track = library.get_track(listing["id"], db)
    # A column we never made searchable is still retrievable.
    assert '"key_confidence": 0.76' in track["result_json"]
    assert library.get_track(999, db) is None
