// UI glue for the Audio Analyzer front-end. No framework: just fetch
// calls to the same JSON API that /docs exposes, plus a poll loop for
// the async separation job.

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileLabel = document.getElementById("file-label");
const go = document.getElementById("go");
const separateBox = document.getElementById("separate-checkbox");
const transcribeBox = document.getElementById("transcribe-checkbox");
const errorBox = document.getElementById("error");
const results = document.getElementById("results");
const facts = document.getElementById("facts");
const resultName = document.getElementById("result-name");
const separation = document.getElementById("separation");
const sepStatus = document.getElementById("sep-status");
const stems = document.getElementById("stems");
const lyricsSection = document.getElementById("lyrics");
const lyricsStatus = document.getElementById("lyrics-status");
const lyricsText = document.getElementById("lyrics-text");

// Ask the server what this deployment can do. The slim Docker
// container has no torch or whisper, so it reports false and we grey
// the options out instead of letting jobs fail.
fetch("/capabilities")
  .then((r) => r.json())
  .then((cap) => {
    for (const [box, wrap, feature, why] of [
      [separateBox, "separate-wrap", cap.separation,
        "not available in this deployment (separation needs torch)"],
      [transcribeBox, "transcribe-wrap", cap.transcription,
        "not available in this deployment (transcription needs whisper)"],
    ]) {
      if (!feature) {
        box.disabled = true;
        const el = document.getElementById(wrap);
        el.classList.add("disabled");
        el.title = why;
      }
    }
  });

fileInput.addEventListener("change", () => {
  go.disabled = !fileInput.files.length;
  fileLabel.textContent =
    fileInput.files[0]?.name || "Choose an audio file…";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  hide(errorBox);
  results.hidden = false;
  separation.hidden = true;
  stems.hidden = true;
  lyricsSection.hidden = true;
  facts.replaceChildren();
  go.disabled = true;
  go.textContent = "working…";

  try {
    renderFacts(await postFile("/analyze", file));
    // Both optional jobs can run; each handles its own polling.
    const jobs = [];
    if (separateBox.checked) jobs.push(runSeparation(file));
    if (transcribeBox.checked) jobs.push(runTranscription(file));
    await Promise.all(jobs);
  } catch (error) {
    showError(error.message);
  } finally {
    go.disabled = false;
    go.textContent = "Analyze";
  }
});

async function postFile(url, file) {
  const body = new FormData(); // the browser sets multipart headers
  body.append("file", file);
  const response = await fetch(url, { method: "POST", body });
  if (!response.ok) {
    const detail = (await response.json()).detail;
    throw new Error(detail || response.statusText);
  }
  return response.json();
}

function renderFacts(data) {
  // textContent everywhere: a weird filename can never inject HTML.
  resultName.textContent = data.file;
  const rows = [
    ["duration", data.duration_s != null ? `${data.duration_s} s` : null],
    ["sample rate", data.sample_rate_hz != null
      ? `${data.sample_rate_hz} Hz` : null],
    ["channels", data.channels],
    ["tempo", data.tempo_bpm != null ? `${data.tempo_bpm} BPM` : "unknown"],
    ["key", data.key
      ? `${data.key} (conf ${data.key_confidence})` : "unknown"],
    ["chords", data.chord_progression || "unknown"],
  ];
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value ?? "—";
    facts.append(dt, dd);
  }

  renderChordTimeline(data.chord_segments);
}

function renderChordTimeline(segments) {
  const timeline = document.getElementById("chord-timeline");
  timeline.replaceChildren();
  timeline.parentElement.hidden = !segments;
  if (!segments) return;
  for (const seg of segments) {
    const line = document.createElement("div");
    line.className = "chord-line";
    const when = document.createElement("span");
    when.className = "chord-when";
    when.textContent = `${fmtTime(seg.start)}–${fmtTime(seg.end)}`;
    const what = document.createElement("span");
    what.textContent = seg.chord;
    line.append(when, what);
    timeline.append(line);
  }
}

function fmtTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// Poll a job endpoint every 2 s until it is done or failed. Shared
// by separation and transcription -- they speak the same protocol.
async function pollJob(pollPath) {
  return new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      try {
        const response = await fetch(pollPath);
        if (!response.ok) throw new Error("job vanished");
        const state = await response.json();
        if (state.status === "done") {
          clearInterval(timer);
          resolve(state);
        } else if (state.status === "failed") {
          clearInterval(timer);
          reject(new Error(state.error));
        }
      } catch (error) {
        clearInterval(timer);
        reject(error);
      }
    }, 2000);
  });
}

async function runSeparation(file) {
  separation.hidden = false;
  sepStatus.textContent = "uploading…";

  const job = await postFile("/separate", file);
  sepStatus.textContent = "separating… this can take a while";
  const done = await pollJob(`/separate/${job.job_id}`);

  document.getElementById("audio-vocals").src = done.vocals_url;
  document.getElementById("audio-inst").src = done.instrumental_url;
  document.getElementById("dl-vocals").href = done.vocals_url;
  document.getElementById("dl-inst").href = done.instrumental_url;
  sepStatus.textContent = "done";
  stems.hidden = false;
}

async function runTranscription(file) {
  lyricsSection.hidden = false;
  lyricsText.hidden = true;
  lyricsStatus.textContent = "uploading…";

  const job = await postFile("/transcribe", file);
  lyricsStatus.textContent = "transcribing… singing is hard for Whisper";
  const done = await pollJob(`/transcribe/${job.job_id}`);

  lyricsText.textContent =
    done.lyrics || "(no words detected)";
  lyricsStatus.textContent =
    `done — detected language: ${done.language}`;
  lyricsText.hidden = false;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
function hide(element) {
  element.hidden = true;
}
