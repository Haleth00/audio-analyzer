"""Tests for tag embedding (mutagen writes real files in tmp_path)."""

import soundfile as sf
from mutagen import File

from tagging import compact_key, tag_file


def make_song(path):
    import numpy as np
    sf.write(path, np.zeros(22050, dtype=np.float32), 22050)
    return path


def test_compact_key_notation():
    assert compact_key("C major") == "C"
    assert compact_key("C minor") == "Cm"
    assert compact_key("F# minor") == "F#m"


def test_tags_written_to_flac(tmp_path):
    path = make_song(tmp_path / "song.flac")
    tag_file(path, bpm=123.4, key="C minor")

    audio = File(path, easy=True)
    assert audio["bpm"] == ["123"]          # 123.4 rounds, text tag
    assert audio["key"] == ["Cm"]


def test_tags_written_to_mp3(tmp_path):
    path = make_song(tmp_path / "song.mp3")
    tag_file(path, bpm=100.0, key="F# major")

    audio = File(path, easy=True)
    assert audio["bpm"] == ["100"]
    # EasyID3 has no 'key' mapping, so TKEY is verified via the raw
    # frame -- exactly how the writer does it.
    from mutagen.id3 import ID3
    tags = ID3(path)
    assert tags.get("TKEY").text == ["F#"]


def test_none_leaves_tag_untouched(tmp_path):
    path = make_song(tmp_path / "song.flac")
    tag_file(path, bpm=90.0, key=None)
    audio = File(path, easy=True)
    assert audio["bpm"] == ["90"]
    assert "key" not in audio
