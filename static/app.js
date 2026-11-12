// UI glue for the Audio Analyzer front-end. No framework: just fetch
// calls to the same JSON API that /docs exposes, plus a poll loop for
// the async separation job.

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileLabel = document.getElementById("file-label");
const go = document.getElementById("go");
const separateBox = document.getElementById("separate-checkbox");
const errorBox = document.getElementById("error");
const results = document.getElementById("results");
const facts = document.getElementById("facts");
const resultName = document.getElementById("result-name");
const separation = document.getElementById("separation");
const sepStatus = document.getElementById("sep-status");
const stems = document.getElementById("stems");

// Ask the server what this deployment can do. The slim Docker
// container has no torch, so it reports separation: false and we
// grey the option out instead of letting jobs fail.
fetch("/capabilities")
  .then((r) => r.json())
  .then((cap) => {
    if (!cap.separation) {
      separateBox.disabled = true;
      document.getElementById("separate-wrap").classList.add("disabled");
      document.getElementById("separate-wrap").title =
        "not available in this deployment (separation needs torch)";
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
  facts.replaceChildren();
  go.disabled = true;
  go.textContent = "working…";

  try {
    renderFacts(await postFile("/analyze", file));
    if (separateBox.checked) await runSeparation(file);
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
  ];
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value ?? "—";
    facts.append(dt, dd);
  }
}

async function runSeparation(file) {
  separation.hidden = false;
  sepStatus.textContent = "uploading…";

  const job = await postFile("/separate", file);
  sepStatus.textContent = "separating… this can take a while";

  // Poll every 2 s until the job is done or failed.
  const done = await new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      try {
        const response = await fetch(`/separate/${job.job_id}`);
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

  document.getElementById("audio-vocals").src = done.vocals_url;
  document.getElementById("audio-inst").src = done.instrumental_url;
  document.getElementById("dl-vocals").href = done.vocals_url;
  document.getElementById("dl-inst").href = done.instrumental_url;
  sepStatus.textContent = "done";
  stems.hidden = false;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
function hide(element) {
  element.hidden = true;
}
