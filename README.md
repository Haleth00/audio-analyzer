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
python scan.py [folder] [--separate] [--csv [FILE]]
```

- `[folder]` — the folder to scan (default: `audio/`)
- `--separate` — also write vocals/instrumental FLAC stems to `separated/` (needs a GPU to be pleasant)
- `--csv [FILE]` — also write a CSV report of every analysis (default name: `report.csv`), sortable in any spreadsheet

## Web service

Start the API server (see Setup above for activating the venv first):

```
uvicorn server:app --reload
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
same job protocol) needs Whisper and is host-only too; the page greys
out whatever the deployment can't do.

## Planned features (free expansion)

- [x] Lyrics transcription (Whisper)
- [ ] Chord detection
- [ ] Song structure segmentation
- [ ] Loudness / dynamics (EBU R128)
- [ ] Library database + search
- [ ] Harmonic-mix playlist builder

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
