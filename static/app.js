// UI glue for the Audio Analyzer front-end. No framework: just fetch
// calls to the same JSON API that /docs exposes, plus a poll loop for
// the async separation job.

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileLabel = document.getElementById("file-label");
const go = document.getElementById("go");
const separateBox = document.getElementById("separate-checkbox");
const transcribeBox = document.getElementById("transcribe-checkbox");
const lyricsSource = document.getElementById("lyrics-source");
const practiceBox = document.getElementById("practice-checkbox");
const practiceControls = document.getElementById("practice-controls");
const practiceSection = document.getElementById("practice-section");
const practiceStatus = document.getElementById("practice-status");
const practicePlayer = document.getElementById("practice-player");
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
// container has no torch, whisper or deepfilternet, so it reports
// false and we grey the options out instead of letting jobs fail.
fetch("/capabilities")
  .then((r) => r.json())
  .then((cap) => {
    if (!cap.separation) {
      separateBox.disabled = true;
      document.getElementById("separate-wrap").classList.add("disabled");
      document.getElementById("separate-wrap").title =
        "not available in this deployment (separation needs torch)";
    }
    if (!cap.transcription) {
      transcribeBox.disabled = true;
      document.getElementById("transcribe-wrap").classList.add("disabled");
      document.getElementById("transcribe-wrap").title =
        "not available in this deployment (transcription needs whisper)";
    }
    // The fancier lyrics modes need the extra models.
    for (const option of lyricsSource.options) {
      const needs = {
        denoised: cap.enhancement,
        vocals: cap.enhancement && cap.separation,
      }[option.value];
      if (needs === false) option.disabled = true;
    }
  });

fileInput.addEventListener("change", () => {
  go.disabled = !fileInput.files.length;
  fileLabel.textContent =
    fileInput.files[0]?.name || "Choose an audio file…";
});

transcribeBox.addEventListener("change", () => {
  lyricsSource.disabled = !transcribeBox.checked;
});

practiceBox.addEventListener("change", () => {
  practiceControls.hidden = !practiceBox.checked;
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
  practiceSection.hidden = true;
  document.getElementById("mix-button").hidden = true;
  document.getElementById("mix-details").hidden = true;
  facts.replaceChildren();
  go.disabled = true;
  go.textContent = "working…";

  try {
    renderFacts(await postFile("/analyze", file));
    // Optional jobs run side by side; each handles its own polling.
    const jobs = [];
    if (separateBox.checked) jobs.push(runSeparation(file));
    if (transcribeBox.checked) {
      jobs.push(runTranscription(file, lyricsSource.value));
    }
    if (practiceBox.checked) {
      jobs.push(runPractice(file,
        parseFloat(document.getElementById("practice-speed").value),
        parseInt(document.getElementById("practice-semitones").value, 10)));
    }
    await Promise.all(jobs);
  } catch (error) {
    showError(error.message);
  } finally {
    go.disabled = false;
    go.textContent = "Analyze";
  }
});

async function runPractice(file, speed, semitones) {
  practiceSection.hidden = false;
  practicePlayer.hidden = true;
  practiceStatus.textContent = "uploading…";

  const query = `speed=${speed}&semitones=${semitones}`;
  const job = await postFile(`/practice?${query}`, file);
  practiceStatus.textContent = "processing…";
  const done = await pollJob(`/practice/${job.job_id}`);

  practicePlayer.src = done.track_url;
  const link = document.getElementById("practice-download");
  link.href = done.track_url;
  practiceStatus.textContent = "done";
  practicePlayer.hidden = false;
}

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
    ["loudness", data.loudness_lufs != null
      ? `${data.loudness_lufs} LUFS` : "unknown"],
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
  renderStructure(data.sections);
  setupMixButton(data.file);
}

function setupMixButton(fileName) {
  const button = document.getElementById("mix-button");
  const details = document.getElementById("mix-details");
  button.hidden = false;
  button.disabled = false;
  button.onclick = async () => {
    button.disabled = true;
    try {
      const response = await fetch(
        `/playlist?start=${encodeURIComponent(fileName)}`);
      if (!response.ok) throw new Error((await response.json()).detail);
      renderMix(await response.json());
    } catch (error) {
      showError(error.message);
    } finally {
      button.disabled = false;
    }
  };
  details.hidden = true;
}

function renderMix(result) {
  const list = document.getElementById("mix-list");
  list.replaceChildren();
  document.getElementById("mix-details").hidden = false;
  for (let i = 0; i < result.tracks.length; i++) {
    const t = result.tracks[i];
    const line = document.createElement("div");
    line.className = "chord-line";
    const when = document.createElement("span");
    when.className = "chord-when";
    when.textContent = `${i + 1}.`;
    const what = document.createElement("span");
    what.textContent =
      `${t.file} · ${t.tempo_bpm} BPM · ${t.camelot}`;
    line.append(when, what);
    list.append(line);
    if (i < result.transitions.length) {
      const step = result.transitions[i];
      const why = document.createElement("div");
      why.className = "chord-why";
      why.textContent =
        `↳ ${step.tempo_change_pct > 0 ? "+" : ""}` +
        `${step.tempo_change_pct}% tempo · ${step.key_move}`;
      list.append(why);
    }
  }
}

function renderStructure(sections) {
  const timeline = document.getElementById("structure-timeline");
  timeline.replaceChildren();
  timeline.parentElement.hidden = !sections;
  if (!sections) return;
  for (const s of sections) {
    const line = document.createElement("div");
    line.className = "chord-line";
    const when = document.createElement("span");
    when.className = "chord-when";
    when.textContent = `${fmtTime(s.start)}–${fmtTime(s.end)}`;
    const what = document.createElement("span");
    what.textContent =
      `${s.label} · ${s.duration_s}s · ` +
      `${Math.round(s.brightness_hz)} Hz`;
    line.append(when, what);
    timeline.append(line);
  }
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

  stems.replaceChildren();
  for (const [name, url] of Object.entries(done.stems)) {
    const row = document.createElement("div");
    row.className = "stem";
    const label = document.createElement("span");
    label.className = "stem-name";
    label.textContent = name;
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = url;
    const link = document.createElement("a");
    link.href = url;
    link.download = url.split("/").pop() === "no_vocals.flac"
      ? "no_vocals.flac" : `${name}.flac`;
    link.textContent = "download";
    row.append(label, audio, link);
    stems.append(row);
  }
  sepStatus.textContent = "done";
  stems.hidden = false;
}

async function runTranscription(file, mode) {
  lyricsSection.hidden = false;
  lyricsText.hidden = true;
  lyricsStatus.textContent = "uploading…";

  const job = await postFile(`/transcribe?mode=${mode}`, file);
  const message = mode === "vocals"
    ? "separating, denoising, transcribing… (the long road)"
    : "transcribing… singing is hard for Whisper";
  lyricsStatus.textContent = message;
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

// --- Reference comparison (its own form) ---

const compareForm = document.getElementById("compare-form");
const mixInput = document.getElementById("mix-input");
const refInput = document.getElementById("ref-input");
const compareGo = document.getElementById("compare-go");

for (const [input, label] of [
  [mixInput, "mix-label"], [refInput, "ref-label"],
]) {
  input.addEventListener("change", () => {
    document.getElementById(label).textContent =
      input.files[0]?.name || (input === mixInput ? "your mix…" : "reference track…");
    compareGo.disabled = !(mixInput.files.length && refInput.files.length);
  });
}

compareForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  hide(errorBox);
  compareGo.disabled = true;
  compareGo.textContent = "comparing…";
  try {
    const body = new FormData();
    body.append("mix", mixInput.files[0]);
    body.append("reference", refInput.files[0]);
    const response = await fetch("/compare", { method: "POST", body });
    if (!response.ok) {
      throw new Error((await response.json()).detail || response.statusText);
    }
    renderComparison(await response.json());
  } catch (error) {
    showError(error.message);
  } finally {
    compareGo.disabled = false;
    compareGo.textContent = "Compare";
  }
});

function renderComparison(result) {
  const section = document.getElementById("compare-results");
  section.hidden = false;
  document.getElementById("compare-title").textContent =
    `${result.mix.file || "mix"} vs ${result.reference.file || "reference"}`;

  const notes = document.getElementById("compare-notes");
  notes.replaceChildren();
  for (const note of result.notes) {
    const li = document.createElement("li");
    li.textContent = note;
    notes.append(li);
  }

  const bands = document.getElementById("compare-bands");
  bands.replaceChildren();
  for (const band of result.bands) {
    const row = document.createElement("div");
    row.className = "band-row";
    const name = document.createElement("span");
    name.className = "band-name";
    name.textContent = `${band.name} (${band.from}–${band.to})`;
    const track = document.createElement("div");
    track.className = "bar-track";
    const bar = document.createElement("div");
    bar.className = "bar " + (band.delta_db >= 0 ? "heavy" : "light");
    // Scale ±6 dB across the whole bar.
    bar.style.width = `${Math.min(Math.abs(band.delta_db), 6) / 6 * 100}%`;
    track.append(bar);
    const value = document.createElement("span");
    value.className = "band-value";
    value.textContent = `${band.delta_db > 0 ? "+" : ""}${band.delta_db} dB`;
    row.append(name, track, value);
    bands.append(row);
  }
}
