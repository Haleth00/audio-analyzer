"""Write analysis results into audio file tags (BPM + key).

DJ software (rekordbox, Traktor, Serato, Mixxx) reads tempo and key
from the files' own tags, so embedding them turns a library scan into
library preparation. Uses mutagen's easy interface, which maps
"BPM"/"key" to the right tag per format (TBPM/TKEY for ID3 mp3s,
Vorbis comments for flac/ogg). Existing tags are preserved; the file
is modified in place.
"""

from mutagen import File

# How the compact key notation maps: we store "C minor", DJs expect
# "Cm". Sharps only -- that's what our pitch classes use.
_COMPACT = {"major": "", "minor": "m"}


def compact_key(key):
    """'C minor' -> 'Cm', 'F# major' -> 'F#'."""
    pitch, mode = key.split()
    return pitch + _COMPACT[mode]


def tag_file(path, bpm, key):
    """Write BPM and key tags into an audio file, in place.

    Either value may be None to leave that tag alone. Raises if the
    format isn't recognized; existing tags other than bpm/key are
    untouched.
    """
    audio = File(path, easy=True)
    if audio is None:
        raise ValueError(f" unrecognized audio format for tagging")

    if bpm is not None:
        # Tag values are strings; BPM as an integer reads cleanest.
        audio["bpm"] = str(int(round(bpm)))
    audio.save()

    if key is not None:
        try:
            audio["key"] = compact_key(key)
            audio.save()
        except KeyError:
            # EasyID3 (mp3) has no easy mapping for the key tag, so
            # write the raw TKEY frame instead. Reopening picks up the
            # bpm we just saved, so nothing is lost.
            from mutagen.id3 import ID3, TKEY

            tags = ID3(path)
            tags.add(TKEY(encoding=3, text=compact_key(key)))
            tags.save(path)
