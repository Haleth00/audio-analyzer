# Audio Analyzer

A Python project that analyzes songs: detect the tempo (BPM), detect the musical key, and separate vocals from the instrumental. The long-term goal is a small web service that does this from an uploaded audio file.

**Status:** early work in progress. I'm learning Python while building it, so the code is minimal for now.

## Planned features

- [ ] Scan a folder of audio files and report duration, sample rate and channels
- [ ] Tempo detection
- [ ] Key detection
- [ ] Vocal / instrumental separation (using a pretrained model)
- [ ] Web interface and API
- [ ] Docker packaging

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

## Notes

- Put test audio files in an `audio/` folder (it is not tracked by Git).
- `flake.nix`, `flake.lock` and `.envrc` are for the author's NixOS development environment. They can be ignored on other systems.
