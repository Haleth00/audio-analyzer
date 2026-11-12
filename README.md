# Audio Analyzer

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

## Web service

Start the API server (see Setup above for activating the venv first):

```
uvicorn server:app --reload
```

Then open <http://127.0.0.1:8000/docs> for interactive API documentation
where you can upload a file and see the JSON response, or call it from a
terminal:

```
curl -F "file=@audio/some_song.flac" http://127.0.0.1:8000/analyze
```

## Docker

Build and run the API in a container:

```
docker build -t audio-analyzer .
docker run -p 8000:8000 audio-analyzer
```

The image is deliberately slim (~900 MB): it installs only
`requirements-api.txt`, which excludes torch and Demucs -- the container
analyzes tempo and key, while separation stays on the host CLI for now.

## Notes

- Put test audio files in an `audio/` folder (it is not tracked by Git).
- `flake.nix`, `flake.lock` and `.envrc` are for the author's NixOS development environment. They can be ignored on other systems.
