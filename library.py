"""The analysis library: every analysis ever run, remembered in SQLite.

One database file (default ~/.local/share/audio-analyzer/library.db)
holds a row per track: searchable columns (tempo, key, loudness, ...)
plus the full result dict as JSON so nothing is lost.

Songs are recorded from both the scanner and the API; re-analyzing a
file replaces its row (deduplicated on filename + duration).
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path.home() / ".local" / "share" / "audio-analyzer" / "library.db"
AUDIO_DIR = Path.home() / ".local" / "share" / "audio-analyzer" / "audio"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file TEXT NOT NULL,
    duration_s REAL,
    sample_rate_hz INTEGER,
    channels INTEGER,
    tempo_bpm REAL,
    loudness_lufs REAL,
    key TEXT,
    key_confidence REAL,
    chord_progression TEXT,
    result_json TEXT,
    audio_path TEXT,
    analyzed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS tracks_file_duration
    ON tracks(file, duration_s);
"""

# ORDER BY can't be a SQL parameter (?), so user input only ever picks
# a name from this whitelist -- the raw SQL lives here, not in input.
ORDER_COLUMNS = {
    "file": "file",
    "tempo": "tempo_bpm",
    "loudness": "loudness_lufs",
    "key": "key",
    "recent": "analyzed_at DESC",
}


def connect(db_path=DB_PATH):
    """Open the library, creating the database if it doesn't exist.

    Also upgrades databases from before the audio_path column: the
    schema above only applies to fresh creates, so older tables get
    the column added by hand.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row  # rows behave like dicts
    con.executescript(SCHEMA)
    columns = {row["name"] for row in con.execute("PRAGMA table_info(tracks)")}
    if "audio_path" not in columns:
        con.execute("ALTER TABLE tracks ADD COLUMN audio_path TEXT")
        con.commit()
    return con


def record(result, db_path=DB_PATH, audio_path=None):
    """Save one analyze_file() result; re-analyses replace the row.

    audio_path points at a playable copy of the file (the original
    for CLI scans, a stored copy for uploads), enabling separation
    and practice jobs to run on library tracks without re-uploading.

    db_path stays the second parameter so existing positional
    callers (tests especially) keep meaning what they meant.

    INSERT OR REPLACE works because of the unique index on
    (file, duration_s): a second analysis of the same recording
    matches the index and overwrites instead of duplicating.
    """
    with closing(connect(db_path)) as con, con:  # `con` commits
        con.execute(
            """
            INSERT OR REPLACE INTO tracks
                (file, duration_s, sample_rate_hz, channels,
                 tempo_bpm, loudness_lufs, key, key_confidence,
                 chord_progression, result_json, audio_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (result.get("file"), result.get("duration_s"),
             result.get("sample_rate_hz"), result.get("channels"),
             result.get("tempo_bpm"), result.get("loudness_lufs"),
             result.get("key"), result.get("key_confidence"),
             result.get("chord_progression"),
             json.dumps(result, default=str),
             str(audio_path) if audio_path else None),
        )


def search(key=None, bpm_min=None, bpm_max=None,
           order="file", limit=100, db_path=DB_PATH):
    """Find tracks: exact key match, tempo range, chosen ordering.

    Returns table columns only (no result_json -- that can be large
    and is fetched per-track by get_track). All values travel as ?
    parameters; only the whitelisted ORDER BY expression is
    interpolated into the SQL string.
    """
    where, params = [], []
    if key:
        where.append("key = ?")
        params.append(key)
    if bpm_min is not None:
        where.append("tempo_bpm >= ?")
        params.append(bpm_min)
    if bpm_max is not None:
        where.append("tempo_bpm <= ?")
        params.append(bpm_max)

    sql = ("SELECT id, file, duration_s, sample_rate_hz, channels, "
           "tempo_bpm, loudness_lufs, key, key_confidence, "
           "chord_progression, audio_path, analyzed_at FROM tracks")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {ORDER_COLUMNS[order]} LIMIT ?"
    params.append(limit)

    with closing(connect(db_path)) as con:
        return [dict(row) for row in con.execute(sql, params)]


def get_track(track_id, db_path=DB_PATH):
    """Fetch one track by id, including the full result JSON."""
    with closing(connect(db_path)) as con:
        row = con.execute("SELECT * FROM tracks WHERE id = ?",
                          (track_id,)).fetchone()
        return dict(row) if row else None
