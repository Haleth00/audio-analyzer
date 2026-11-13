"""Web service for the audio analyzer.

Development run:  uvicorn server:app --reload
Then open:        http://127.0.0.1:8000/docs

Analyses come from analyze.py; separation from separation.py.
Separation is slow and needs torch, so it runs as a background job:
POST /separate returns a job id immediately, and you poll
GET /separate/{job_id} until the stems are ready to download.
"""

import importlib.util
import shutil
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import analyze
import compare
import enhance
import library
import mix
import practice
import separation
import transcribe

BASE_DIR = Path(__file__).parent

# Separation needs torch and demucs, transcription needs whisper,
# denoising needs DeepFilterNet -- none of which the slim Docker
# container has. Probe for them without importing (which would be
# slow) and let the front-end know what's on the menu.
SEPARATION_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("demucs") is not None
)
TRANSCRIPTION_AVAILABLE = importlib.util.find_spec("whisper") is not None
ENHANCEMENT_AVAILABLE = importlib.util.find_spec("df") is not None

app = FastAPI(title="Audio Analyzer", version="0.3.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"),
          name="static")


@app.middleware("http")
async def always_revalidate(request, call_next):
    """Tell browsers to re-check pages and assets on every load.

    Without this, a browser that cached yesterday's app.js renders it
    against today's index.html -- buttons that reference elements
    which no longer exist, i.e. a mysteriously half-broken page.
    Revalidation is cheap (304s when nothing changed) and correct
    for a tool that changes daily.
    """
    response = await call_next(request)
    if request.url.path.startswith("/static") or \
            "text/html" in response.headers.get("content-type", ""):
        response.headers["Cache-Control"] = "no-cache"
    return response

# In-memory job registry. Fine for one process and one user; a restart
# forgets every job, and multiple uvicorn workers would each see only
# their own dict -- a database is the eventual answer, not today.
JOBS: dict[str, dict] = {}
JOBS_DIR = Path("jobs")

STEM_FILES = {
    "vocals": "vocals.flac",
    "instrumental": "no_vocals.flac",
    "drums": "drums.flac",
    "bass": "bass.flac",
    "other": "other.flac",
}


@app.get("/", include_in_schema=False)
def index():
    """The upload page. include_in_schema keeps /docs tidy."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/library", include_in_schema=False)
def library_page():
    """The collection browser: everything ever analyzed."""
    return FileResponse(BASE_DIR / "static" / "library.html")


@app.get("/capabilities")
def capabilities():
    """What this deployment can do (the slim container: no torch)."""
    return {
        "separation": SEPARATION_AVAILABLE,
        "transcription": TRANSCRIPTION_AVAILABLE,
        "enhancement": ENHANCEMENT_AVAILABLE,
    }


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
    tmp = Path("/tmp") / f"analyze-{uuid.uuid4().hex}{suffix}"
    try:
        with open(tmp, "wb") as out:
            shutil.copyfileobj(file.file, out)
        result = analyze.analyze_file(tmp)
        # Report the name the client sent, not our temp file's name.
        if file.filename:
            result["file"] = file.filename
        # Keep a copy so separation/practice can run on this track
        # from the library later, without a re-upload.
        stored = library.AUDIO_DIR / f"{uuid.uuid4().hex}{suffix}"
        stored.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(tmp, stored)
            library.record(result, audio_path=stored)
        except Exception:
            # The analysis succeeded; failing to remember it must not
            # fail the request.
            stored.unlink(missing_ok=True)
            try:
                library.record(result)
            except Exception:
                pass
        return result
    finally:
        # Runs whether analysis succeeded or raised.
        tmp.unlink(missing_ok=True)


@app.get("/api/library")
def library_search(key: str | None = None,
                   bpm_min: float | None = None,
                   bpm_max: float | None = None,
                   order: str = "file",
                   limit: int = 100):
    """Search the recorded analysis library (the JSON behind
    /library).

    Query parameters become function arguments automatically.
    """
    if order not in library.ORDER_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"order must be one of {sorted(library.ORDER_COLUMNS)}",
        )
    tracks = library.search(key=key, bpm_min=bpm_min, bpm_max=bpm_max,
                            order=order, limit=limit)
    for track in tracks:
        audio = track.pop("audio_path", None)
        # Can't separate what we can't reach (uploads before audio
        # storage, or a moved/deleted original).
        track["has_audio"] = bool(audio) and Path(audio).exists()
    return {"count": len(tracks), "tracks": tracks}


@app.get("/api/library/{track_id}")
def library_track(track_id: int):
    """One track's full analysis, including the stored result JSON."""
    track = library.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="unknown track id")
    return track


@app.get("/playlist")
def playlist(start: str, length: int = 8):
    """Build a harmonic mix starting at a library track."""
    try:
        return mix.build_playlist(start, length=length)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.post("/compare")
def compare_uploads(mix: UploadFile, reference: UploadFile):
    """Compare your mix against a reference track.

    Two uploads, one JSON answer: loudness, spectral balance per
    band, dynamics -- all relative to the reference. Pure DSP, no
    models, fast enough to answer synchronously.
    """
    mix_path = Path("/tmp") / f"compare-{uuid.uuid4().hex}"
    ref_path = Path("/tmp") / f"compare-{uuid.uuid4().hex}"
    try:
        with open(mix_path, "wb") as out:
            shutil.copyfileobj(mix.file, out)
        with open(ref_path, "wb") as out:
            shutil.copyfileobj(reference.file, out)
        result = compare.compare_files(mix_path, ref_path)
        if mix.filename:
            result["mix"]["file"] = mix.filename
        if reference.filename:
            result["reference"]["file"] = reference.filename
        return result
    finally:
        mix_path.unlink(missing_ok=True)
        ref_path.unlink(missing_ok=True)


def _create_separation_job(source_path, display_name):
    """Register a separation job reading audio from source_path.

    The audio is copied into the job dir under the name run_job
    expects (input*), whether it came from an upload or the library.
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path,
                    job_dir / ("input" + Path(source_path).suffix))
    JOBS[job_id] = {
        "job_id": job_id,
        "kind": "separation",
        "file": display_name,
        "status": "pending",
    }
    return job_id


@app.post("/library/{track_id}/separate", status_code=202)
def separate_library_track(track_id: int,
                           background_tasks: BackgroundTasks):
    """Separate a stored library track (no re-upload needed)."""
    track = library.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="unknown track id")
    audio = track.get("audio_path")
    if not audio or not Path(audio).exists():
        raise HTTPException(
            status_code=409,
            detail="this track has no reachable audio "
                   "(analyzed before audio storage, or the original "
                   "file moved) -- re-upload it once",
        )
    job_id = _create_separation_job(audio, track["file"])
    background_tasks.add_task(run_job, job_id)
    return JOBS[job_id] | {"poll": f"/separate/{job_id}"}


@app.post("/separate-all", status_code=202)
def separate_everything(background_tasks: BackgroundTasks):
    """Queue separation jobs for every library track with reachable
    audio. Returns the job ids; poll each via /separate/{id}."""
    job_ids = []
    for track in library.search(limit=1000, order="recent"):
        audio = track.get("audio_path")
        if audio and Path(audio).exists():
            job_id = _create_separation_job(audio, track["file"])
            background_tasks.add_task(run_job, job_id)
            job_ids.append(job_id)
    if not job_ids:
        raise HTTPException(
            status_code=409,
            detail="no library tracks have reachable audio",
        )
    return {"count": len(job_ids),
            "jobs": [JOBS[j] | {"poll": f"/separate/{j}"} for j in job_ids]}


@app.post("/separate", status_code=202)
def start_separation(file: UploadFile, background_tasks: BackgroundTasks):
    """Accept a file for vocal/instrumental separation.

    Returns immediately with a job id; poll GET /separate/{job_id}
    until status is "done" (or "failed"). 202 is the status code for
    "accepted, but the work has not finished yet".
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in analyze.AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix!r}; "
                   f"expected one of {sorted(analyze.AUDIO_EXTENSIONS)}",
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir / f"input{suffix}", "wb") as out:
        shutil.copyfileobj(file.file, out)

    JOBS[job_id] = {
        "job_id": job_id,
        "file": file.filename,
        "status": "pending",
    }
    # The task runs in a worker thread once this response has been sent.
    background_tasks.add_task(run_job, job_id)
    # dict | dict merges two dicts into a new one (Python 3.9+).
    return JOBS[job_id] | {"poll": f"/separate/{job_id}"}


def run_job(job_id):
    """Do the actual separation; runs in a background thread."""
    job = JOBS[job_id]
    job["status"] = "running"
    job_dir = JOBS_DIR / job_id
    try:
        input_path = next(job_dir.glob("input*"))  # the one we saved
        separation.separate_file(input_path, job_dir)
        job["status"] = "done"
        job["stems"] = {name: f"/separate/{job_id}/stems/{name}"
                        for name in STEM_FILES}
    except Exception as error:
        job["status"] = "failed"
        job["error"] = str(error)
    finally:
        park_all_models()


@app.get("/separate/{job_id}")
def job_status(job_id: str):
    """Poll a separation job: pending -> running -> done | failed."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="unknown job id")
    return JOBS[job_id]


PRACTICE_STEMS = ("mix", "vocals", "drums", "bass", "other")


@app.post("/practice", status_code=202)
def start_practice(file: UploadFile, background_tasks: BackgroundTasks,
                   speed: float = 1.0, semitones: int = 0,
                   stem: str = "mix"):
    """Make a practice track: slow down (pitch kept) and/or transpose.

    speed 0.75 = 75% speed; semitones -12..+12. stem picks what to
    practice: the full mix, or one isolated Demucs stem (drums, bass,
    other -- guitars and keys live in "other"). Same job protocol as
    the other endpoints. The mix path is pure librosa and works in
    the slim container; stems need Demucs (host only).
    """
    if not (practice.SPEED_RANGE[0] <= speed <= practice.SPEED_RANGE[1]):
        raise HTTPException(
            status_code=400,
            detail=f"speed must be between {practice.SPEED_RANGE[0]} "
                   f"and {practice.SPEED_RANGE[1]}",
        )
    if not (practice.SEMITONE_RANGE[0] <= semitones
            <= practice.SEMITONE_RANGE[1]):
        raise HTTPException(
            status_code=400,
            detail=f"semitones must be between {practice.SEMITONE_RANGE[0]} "
                   f"and {practice.SEMITONE_RANGE[1]}",
        )
    if stem not in PRACTICE_STEMS:
        raise HTTPException(
            status_code=400,
            detail=f"stem must be one of: {', '.join(PRACTICE_STEMS)}",
        )
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in analyze.AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix!r}; "
                   f"expected one of {sorted(analyze.AUDIO_EXTENSIONS)}",
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir / f"input{suffix}", "wb") as out:
        shutil.copyfileobj(file.file, out)

    JOBS[job_id] = {
        "job_id": job_id,
        "kind": "practice",
        "file": file.filename,
        "speed": speed,
        "semitones": semitones,
        "stem": stem,
        "status": "pending",
    }
    background_tasks.add_task(run_practice_job, job_id)
    return JOBS[job_id] | {"poll": f"/practice/{job_id}"}


def run_practice_job(job_id):
    """Do the practice-track processing; runs in a background thread."""
    job = JOBS[job_id]
    job["status"] = "running"
    job_dir = JOBS_DIR / job_id
    try:
        input_path = next(job_dir.glob("input*"))  # the one we saved
        path = input_path
        if job["stem"] != "mix":
            # Isolate the requested stem first, then free Demucs'
            # GPU memory before the DSP stage (same discipline as the
            # lyrics chain).
            separation.separate_file(path, job_dir)
            separation.park()
            path = job_dir / f"{job['stem']}.flac"
        samples, sr = practice.practice_file(
            path, speed=job["speed"], semitones=job["semitones"])
        sf.write(job_dir / "practice.flac", samples, sr)
        job["status"] = "done"
        job["track_url"] = f"/practice/{job_id}/track"
    except Exception as error:
        job["status"] = "failed"
        job["error"] = str(error)
    finally:
        park_all_models()


@app.get("/practice/{job_id}")
def practice_status(job_id: str):
    """Poll a practice job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="unknown job id")
    return JOBS[job_id]


@app.get("/practice/{job_id}/track")
def practice_track(job_id: str):
    """Download the finished practice track."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="unknown job id")
    job = JOBS[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=409,
                            detail=f"job is {job['status']}, not done")
    return FileResponse(JOBS_DIR / job_id / "practice.flac",
                        media_type="audio/flac", filename="practice.flac")


def park_all_models():
    """Park every heavy model after a job.

    Each module's park() forgets its cached model and returns GPU
    memory; on a 4 GB card, leaving Demucs resident while Whisper
    loads is a guaranteed CUDA-out-of-memory.
    """
    for module in (separation, enhance, transcribe):
        try:
            module.park()
        except Exception:
            pass


@app.get("/separate/{job_id}/stems/{stem}")
def download_stem(job_id: str, stem: str):
    """Download one stem of a finished job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="unknown job id")
    job = JOBS[job_id]
    if job["status"] != "done":
        # 409 = conflict: the request makes sense, but not in the
        # job's current state.
        raise HTTPException(status_code=409,
                            detail=f"job is {job['status']}, not done")
    if stem not in STEM_FILES:
        raise HTTPException(status_code=404,
                            detail=f"stem must be one of {sorted(STEM_FILES)}")
    path = JOBS_DIR / job_id / STEM_FILES[stem]
    return FileResponse(path, media_type="audio/flac",
                        filename=STEM_FILES[stem])


@app.post("/transcribe", status_code=202)
def start_transcription(file: UploadFile, background_tasks: BackgroundTasks,
                        mode: str = "mix"):
    """Accept a file for lyrics transcription (another async job).

    mode picks the input Whisper gets: "mix" the original, "denoised"
    the DeepFilterNet-cleaned audio (for speech recordings), or
    "vocals" the Demucs vocal stem, denoised -- the best quality for
    songs. Same contract as /separate otherwise.
    """
    if mode not in ("mix", "denoised", "vocals"):
        raise HTTPException(
            status_code=400,
            detail="mode must be one of: mix, denoised, vocals",
        )
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in analyze.AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix!r}; "
                   f"expected one of {sorted(analyze.AUDIO_EXTENSIONS)}",
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir / f"input{suffix}", "wb") as out:
        shutil.copyfileobj(file.file, out)

    JOBS[job_id] = {
        "job_id": job_id,
        "kind": "transcription",
        "mode": mode,
        "file": file.filename,
        "status": "pending",
    }
    background_tasks.add_task(run_transcription_job, job_id)
    return JOBS[job_id] | {"poll": f"/transcribe/{job_id}"}


def run_transcription_job(job_id):
    """Do the transcription; runs in a background thread."""
    job = JOBS[job_id]
    job["status"] = "running"
    job_dir = JOBS_DIR / job_id
    try:
        input_path = next(job_dir.glob("input*"))  # the one we saved
        path = input_path
        if job["mode"] == "vocals":
            # Isolate the vocal stem first; DeepFilterNet then cleans
            # whatever instruments leaked through.
            separation.separate_file(path, job_dir)
            separation.park()  # free Demucs before Whisper loads
            path = job_dir / "vocals.flac"
        result = transcribe.transcribe_file(
            path, denoise=job["mode"] != "mix")
        job["status"] = "done"
        # Lyrics are small, so they ride along in the JSON response
        # instead of being a file download like the stems.
        job["lyrics"] = result["text"]
        job["language"] = result["language"]
    except Exception as error:
        job["status"] = "failed"
        job["error"] = str(error)
    finally:
        park_all_models()


@app.get("/transcribe/{job_id}")
def transcription_status(job_id: str):
    """Poll a transcription job: pending -> running -> done | failed."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="unknown job id")
    return JOBS[job_id]
