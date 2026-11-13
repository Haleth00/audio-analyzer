# Audio Analyzer

[![CI](https://github.com/Haleth00/audio-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/Haleth00/audio-analyzer/actions/workflows/ci.yml)

A Python project that analyzes songs: detect the tempo (BPM), detect the musical key, and separate vocals from the instrumental. The long-term goal is a small web service that does this from an uploaded audio file.

**Status:** early work in progress. I'm learning Python while building it, so the code is minimal for now.

## Planned features

- [x] Scan a folder of audio files and report duration, sample rate and channels
- [x] Tempo detection
- [x] Key detection
- [x] Vocal / instrumental separation (using a pretrained model)
- [x] Web interface and API
- [x] Docker packaging

## Setup (Windows)

Requirements: Python 3.12 and Git.

```
git clone git@github.com:Haleth00/audio-analyzer.git
cd audio-analyzer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scan.py
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `cmd`.

## Setup (Linux / macOS)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scan.py
```

## Command line

```
python scan.py [folder] [--separate] [--csv [FILE]] [--tag]
```

- `[folder]` — the folder to scan (default: `audio/`)
- `--separate` — also write FLAC stems to `separated/`: Demucs' four tracks (vocals, drums, bass, other) plus `no_vocals` for karaoke (needs a GPU to be pleasant)
- `--csv [FILE]` — also write a CSV report of every analysis (default name: `report.csv`), sortable in any spreadsheet
- `--tag` — write BPM and key into the files' own tags (mp3: TBPM/TKEY, flac/ogg: Vorbis comments), so DJ software imports them pre-analyzed. Modifies files in place; other tags are preserved.

## Web service

Start the API server from the project directory (direnv loads the
environment on cd):

```
./serve            # or ./serve --reload for auto-restart on edits
```

Then open <http://127.0.0.1:8000/> for the upload page (tempo, key, and
stems with playable/downloadable results), or
<http://127.0.0.1:8000/docs> for the interactive API documentation. From
a terminal:

```
curl -F "file=@audio/some_song.flac" http://127.0.0.1:8000/analyze
```

Separation runs as a background job, since it takes seconds to minutes:
upload and get a job id (HTTP 202), poll until done, then download the
stems:

```
curl -F "file=@audio/some_song.flac" http://127.0.0.1:8000/separate   # -> {"job_id": ...}
curl http://127.0.0.1:8000/separate/<job_id>                          # -> status + stem URLs
curl http://127.0.0.1:8000/separate/<job_id>/stems/vocals -o vocals.flac
```

Separation needs torch and benefits hugely from a GPU, so it runs on
the host server only -- not inside the slim Docker container, which
serves `/analyze` alone. Lyrics transcription (`POST /transcribe`,
same job protocol) needs Whisper and is host-only too; it takes a
`mode`: `mix` (original), `denoised` (DeepFilterNet first -- for
speech recordings), or `vocals` (isolated vocal stem, denoised --
the quality route for songs). Practice tracks (`POST /practice`,
slow-down/transpose) and reference-mix comparison (`POST /compare`,
two uploads) are pure librosa/pyloudnorm DSP and work everywhere,
container included. The page greys out whatever the deployment
can't do.

## Library

Every analysis -- scanner runs and API uploads alike -- is remembered
in an SQLite database at `~/.local/share/audio-analyzer/library.db`;
re-analyzing a file replaces its row.

```
python scan.py --no-save        # scan without recording
python lookup.py --key "C minor" --bpm-min 100 --bpm-max 140 --order tempo
curl "http://127.0.0.1:8000/api/library?key=C%20minor&order=recent"
```

Or browse it: open <http://127.0.0.1:8000/library> for the sortable,
searchable table with per-track detail and mix building.

Build a DJ set from it -- consecutive tracks stay within 6% tempo and
compatible on the Camelot wheel (same key, relative major/minor, or
adjacent), each transition explained:

```
python mix.py "Soen - Huntress.mp3" --length 8
curl "http://127.0.0.1:8000/playlist?start=Soen%20-%20Huntress.mp3"
```

The upload page has a "Build a mix from this track" button too.

## Planned features (free expansion)

- [x] Lyrics transcription (Whisper)
- [x] Chord detection (full triad / sixth / seventh vocabulary)
- [x] Song structure segmentation
- [x] Loudness / dynamics (EBU R128)
- [x] Library database + search
- [x] Harmonic-mix playlist builder
- [x] Speech denoising (DeepFilterNet) with lyrics source modes
- [x] Practice tracks (slow down / transpose, pitch kept, on the mix or a single instrument stem)
- [x] Reference-mix comparison (loudness, spectral balance, dynamics)

## Docker

Build and run the API in a container:

```
docker build -t audio-analyzer .
docker run -p 8000:8000 audio-analyzer
```

The image is deliberately slim (~900 MB): it installs only
`requirements-api.txt`, which excludes torch and Demucs -- the container
serves the upload page and `/analyze`, while separation stays on the
host (the page disables the option when `/capabilities` says so).

## Development

Run the test suite (all test audio is synthesized -- no fixture files):

```
pip install -r requirements-dev.txt
pytest -v
```

GitHub Actions runs the same suite on every push.

## Notes

- Put test audio files in an `audio/` folder (it is not tracked by Git).
- `flake.nix`, `flake.lock` and `.envrc` are for the author's NixOS development environment. They can be ignored on other systems.
