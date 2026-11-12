"""Web service for the audio analyzer.

Development run:  uvicorn server:app --reload
Then open:        http://127.0.0.1:8000/docs

The analyses are imported from analyze.py -- the same functions the
scan.py CLI uses.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

import analyze

app = FastAPI(title="Audio Analyzer", version="0.1.0")


@app.get("/health")
def health():
    """Cheap endpoint for checking that the service is alive."""
    return {"status": "ok"}


# Note this is a plain `def`, not `async def`: the analysis is
# CPU-bound and takes seconds, and FastAPI runs sync endpoints in a
# thread pool so one long analysis doesn't block every other request.
@app.post("/analyze")
def analyze_upload(file: UploadFile):
    """Analyze an uploaded audio file; returns tempo and key as JSON."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in analyze.AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix!r}; "
                   f"expected one of {sorted(analyze.AUDIO_EXTENSIONS)}",
        )

    # The upload arrives as a stream; park it in a temporary file so
    # the analysis functions can read it like any other file.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix or "", delete=False)
    try:
        with tmp:
            shutil.copyfileobj(file.file, tmp)
        result = analyze.analyze_file(Path(tmp.name))
        # Report the name the client sent, not our temp file's name.
        if file.filename:
            result["file"] = file.filename
        return result
    finally:
        # Runs whether analysis succeeded or raised.
        Path(tmp.name).unlink(missing_ok=True)
