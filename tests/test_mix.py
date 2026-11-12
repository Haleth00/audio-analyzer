"""Tests for the harmonic-mix playlist builder."""

import library
import mix


def seed(db, file, bpm, key, duration=60.0):
    library.record(
        {"file": file, "duration_s": duration,
         "tempo_bpm": bpm, "key": key},
        db,
    )


def test_camelot_wheel_mapping():
    assert mix.camelot("C major") == "8B"
    assert mix.camelot("A minor") == "8A"   # relative of C major
    assert mix.camelot("C minor") == "5A"
    assert mix.camelot("G major") == "9B"   # one step clockwise


def test_playlist_picks_compatible_nearest(tmp_path):
    db = tmp_path / "lib.db"
    seed(db, "start.wav", 120.0, "C major")    # 8B
    seed(db, "relative.wav", 120.0, "A minor", duration=61.0)  # 8A
    seed(db, "adjacent.wav", 122.0, "G major", duration=62.0)  # 9B
    seed(db, "far.wav", 120.0, "C# major", duration=63.0)      # 3B
    seed(db, "toofast.wav", 160.0, "C major", duration=64.0)   # 8B

    result = mix.build_playlist("start.wav", length=8, db_path=db)
    names = [t["file"] for t in result["tracks"]]

    assert names[0] == "start.wav"
    # Relative key with zero tempo gap beats the adjacent-wheel pick.
    assert names[1] == "relative.wav"
    assert names[2] == "adjacent.wav"
    # Incompatible key and out-of-range tempo never make the cut.
    assert "far.wav" not in names
    assert "toofast.wav" not in names


def test_unknown_start_is_an_error(tmp_path):
    try:
        mix.build_playlist("ghost.wav", db_path=tmp_path / "lib.db")
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "not in the library" in str(error)
