// The library browser: table of analyzed tracks, client-side filter
// and sorting, detail panel fed by the stored result JSON.

const tbody = document.getElementById("tracks-body");
const filterBox = document.getElementById("filter");
const emptyNote = document.getElementById("empty");
const detail = document.getElementById("detail");

let tracks = [];
let sortKey = "analyzed_at";
let sortDown = true;
let selected = null; // the currently selected row's full result

document.querySelectorAll("th.sortable").forEach((th) => {
  th.addEventListener("click", () => {
    if (sortKey === th.dataset.key) {
      sortDown = !sortDown;
    } else {
      sortKey = th.dataset.key;
      sortDown = false;
    }
    renderTable();
  });
});

filterBox.addEventListener("input", renderTable);

fetch("/api/library?limit=500&order=recent")
  .then((r) => r.json())
  .then((data) => {
    tracks = data.tracks;
    renderTable();
  });

function visibleTracks() {
  const needle = filterBox.value.trim().toLowerCase();
  const filtered = needle
    ? tracks.filter((t) =>
        Object.values(t).join(" ").toLowerCase().includes(needle))
    : tracks;
  return [...filtered].sort((a, b) => {
    const va = a[sortKey] ?? "";
    const vb = b[sortKey] ?? "";
    // Numbers compare numerically, strings lexically.
    const cmp = (typeof va === "number" && typeof vb === "number")
      ? va - vb
      : String(va).localeCompare(String(vb));
    return sortDown ? -cmp : cmp;
  });
}

function renderTable() {
  tbody.replaceChildren();
  emptyNote.hidden = tracks.length > 0;
  for (const t of visibleTracks()) {
    const tr = document.createElement("tr");
    tr.dataset.id = t.id;
    for (const value of [t.file, t.tempo_bpm, t.key,
                         t.loudness_lufs, fmtDuration(t.duration_s),
                         (t.analyzed_at || "").slice(0, 16)]) {
      const td = document.createElement("td");
      td.textContent = value ?? "—";
      tr.append(td);
    }
    const actions = document.createElement("td");
    actions.className = "row-actions";
    tr.append(actions);
    if (t.has_audio) {
      const button = document.createElement("button");
      button.textContent = "separate";
      button.addEventListener("click", (e) => {
        e.stopPropagation();
        separateOne(t.id, actions);
      });
      actions.append(button);
    } else {
      const note = document.createElement("span");
      note.className = "muted";
      note.textContent = "no audio stored";
      actions.append(note);
    }
    tr.addEventListener("click", () => showDetail(t.id, tr));
    tbody.append(tr);
  }
}

async function separateOne(trackId, cell) {
  cell.replaceChildren(waitingNote("queued…"));
  try {
    const response = await fetch(`/library/${trackId}/separate`,
                                 { method: "POST" });
    if (!response.ok) throw new Error((await response.json()).detail);
    const job = await response.json();
    cell.replaceChildren(waitingNote("separating…"));
    const done = await pollJob(job.poll);
    fillStemLinks(cell, done.stems);
  } catch (error) {
    cell.replaceChildren(waitingNote(error.message));
  }
}

function waitingNote(text) {
  const span = document.createElement("span");
  span.className = "muted";
  span.textContent = text;
  return span;
}

function fillStemLinks(cell, stems) {
  cell.replaceChildren();
  for (const [name, url] of Object.entries(stems)) {
    const link = document.createElement("a");
    link.href = url;
    link.textContent = name;
    cell.append(link);
  }
}

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

document.getElementById("separate-all").onclick = async () => {
  const status = document.getElementById("batch-status");
  status.textContent = "queueing…";
  try {
    const response = await fetch("/separate-all", { method: "POST" });
    if (!response.ok) throw new Error((await response.json()).detail);
    const batch = await response.json();
    let finished = 0;
    status.textContent = `0 / ${batch.count} done`;
    for (const job of batch.jobs) {
      pollJob(job.poll).then((done) => {
        const row = tracks.find((t) => t.file === job.file);
        if (row) {
          const cell = tbody.querySelector(
            `tr[data-id="${row.id}"] .row-actions`);
          if (cell) fillStemLinks(cell, done.stems);
        }
        finished += 1;
        status.textContent = `${finished} / ${batch.count} done`;
      }).catch(() => {
        finished += 1;
        status.textContent = `${finished} / ${batch.count} (some failed)`;
      });
    }
  } catch (error) {
    status.textContent = error.message;
  }
};

async function showDetail(id, row) {
  document.querySelectorAll("#tracks-body tr").forEach((r) =>
    r.classList.remove("selected"));
  row.classList.add("selected");

  const response = await fetch(`/api/library/${id}`);
  selected = await response.json();
  const result = parseResult(selected.result_json);

  detail.hidden = false;
  document.getElementById("detail-name").textContent = selected.file;

  const facts = document.getElementById("detail-facts");
  facts.replaceChildren();
  for (const [label, value] of [
    ["duration", `${selected.duration_s} s`],
    ["tempo", `${selected.tempo_bpm} BPM`],
    ["key", `${selected.key} (conf ${selected.key_confidence})`],
    ["loudness", `${selected.loudness_lufs} LUFS`],
  ]) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value ?? "—";
    facts.append(dt, dd);
  }

  document.getElementById("detail-chords").textContent =
    result.chord_progression || "";

  const structure = document.getElementById("detail-structure");
  structure.replaceChildren();
  for (const s of result.sections || []) {
    const line = document.createElement("div");
    line.className = "chord-line";
    const when = document.createElement("span");
    when.className = "chord-when";
    when.textContent = `${fmtTime(s.start)}–${fmtTime(s.end)}`;
    const what = document.createElement("span");
    what.textContent = `${s.label} · ${s.duration_s}s`;
    line.append(when, what);
    structure.append(line);
  }

  document.getElementById("mix-details").hidden = true;
}

function parseResult(json) {
  try {
    return JSON.parse(json);
  } catch {
    return {};
  }
}

document.getElementById("mix-button").onclick = async () => {
  if (!selected) return;
  const button = document.getElementById("mix-button");
  button.disabled = true;
  try {
    const response = await fetch(
      `/playlist?start=${encodeURIComponent(selected.file)}`);
    if (!response.ok) throw new Error((await response.json()).detail);
    renderMix(await response.json());
  } finally {
    button.disabled = false;
  }
};

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
    what.textContent = `${t.file} · ${t.tempo_bpm} BPM · ${t.camelot}`;
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

function fmtTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function fmtDuration(seconds) {
  if (seconds == null) return "—";
  return `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60))
    .padStart(2, "0")}`;
}
