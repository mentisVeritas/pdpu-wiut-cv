const COLORS = {
  accident: "#ff5d5d",
  near_miss: "#ff9f43",
  red_light: "#ff4d6d",
  wrong_way: "#c77dff",
  illegal_u_turn: "#b8c0ff",
  stopped_vehicle: "#4cc9f0",
  jaywalking: "#80ed99",
  failure_to_yield: "#f72585",
  illegal_turn: "#9bf6ff",
  solid_line_crossing: "#ffd166",
  stop_line: "#ef476f",
  congestion: "#118ab2",
  road_obstacle: "#8d99ae",
  fire_smoke: "#f77f00",
};

const player = document.getElementById("player");
const progress = document.getElementById("progress");
const timeline = document.getElementById("timeline");
const legend = document.getElementById("legend");
const raw = document.getElementById("raw");
let events = [];
let durationHint = 60;

function setStatus(text) {
  progress.textContent = text;
}

function pickEvents(payload) {
  if (Array.isArray(payload)) return payload;
  if (payload && payload.videos) {
    const first = Object.values(payload.videos)[0];
    return (first && first.events) || [];
  }
  if (payload && Array.isArray(payload.events)) return payload.events;
  return [];
}

function renderLegend(used) {
  legend.innerHTML = "";
  used.forEach((label) => {
    const chip = document.createElement("span");
    chip.textContent = label;
    chip.style.background = COLORS[label] || "#ccc";
    legend.appendChild(chip);
  });
}

function renderTimeline(list, duration) {
  timeline.innerHTML = "";
  const dur = Math.max(duration || durationHint, 1);
  const used = [...new Set(list.map((e) => e[2]))];
  renderLegend(used);
  list.forEach(([start, end, label]) => {
    const bar = document.createElement("i");
    bar.title = `${label} ${start.toFixed(1)}–${end.toFixed(1)}s`;
    bar.style.left = `${(start / dur) * 100}%`;
    bar.style.width = `${Math.max(0.8, ((end - start) / dur) * 100)}%`;
    bar.style.background = COLORS[label] || "#888";
    bar.addEventListener("click", () => {
      player.currentTime = start;
      player.play().catch(() => {});
    });
    timeline.appendChild(bar);
  });
  raw.hidden = false;
  raw.textContent = JSON.stringify(list, null, 2);
}

document.getElementById("video").addEventListener("change", (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  if (player.src) URL.revokeObjectURL(player.src);
  player.src = URL.createObjectURL(file);
  setStatus(`${file.name} · ${(file.size / 1e6).toFixed(1)} MB — attach events JSON or load the example.`);
});

player.addEventListener("loadedmetadata", () => {
  durationHint = player.duration || durationHint;
  if (events.length) renderTimeline(events, durationHint);
});

document.getElementById("json").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    events = pickEvents(payload);
    setStatus(`Loaded ${events.length} event(s) from ${file.name}.`);
    renderTimeline(events, player.duration || durationHint);
  } catch (err) {
    setStatus(`Could not parse JSON: ${err.message}`);
  }
});

document.getElementById("load-example").addEventListener("click", async () => {
  const payload = await fetch("example_predictions.json").then((r) => r.json());
  events = pickEvents(payload);
  durationHint = 60;
  setStatus(`Example timeline (${events.length} events). Upload a video and click a bar to seek.`);
  renderTimeline(events, durationHint);
});

document.getElementById("run").addEventListener("click", () => {
  if (!events.length) {
    setStatus("No events yet. Load the example or a predictions JSON.");
    return;
  }
  if (!player.src) setStatus("Timeline ready. Add a video to seek into it.");
  else setStatus(`${events.length} events on the timeline — click a bar to jump.`);
  renderTimeline(events, player.duration || durationHint);
});
