// Live demo: the Hugging Face Space. Empty until the Space is up.
const DEMO_URL = "";

const TEAM = [
  {
    name: "Begzad Kenesbaev",
    role: "Lead · pipeline, scene, submission",
    did: "Drew the scene layout (roadway, crossings, stop line, traffic light, solid lines), labelled events, " +
         "directed the rules and Part B, owns the repository and the submission.",
    links: { GitHub: "https://github.com/mentisVeritas" },
  },
  {
    name: "Fariza Raxmanova",
    role: "Data · annotation and EDA",
    did: "Annotation of the sample clips with the official start/end conventions, EDA notes.",
    links: { GitHub: "https://github.com/farizarakhmanova" },
  },
  {
    name: "mallokodev",
    role: "Product · website and demo",
    did: "Website and the live demo.",
    links: { GitHub: "https://github.com/httpswap" },
  },
];

const EVENT_COLORS = {
  red_light: "#f03c3c", stop_line: "#fa78b4", solid_line_crossing: "#e678fa", jaywalking: "#3cbefa",
  failure_to_yield: "#6464f0", stopped_vehicle: "#3cc8a0", congestion: "#fa6e78", wrong_way: "#82c83c",
};
const SIGNAL_COLORS = { R: "#e5484d", G: "#30a46c", Y: "#f5a524", "?": "#8b8d98" };
const SIGNAL_NAMES = { R: "red", G: "green", Y: "amber", "?": "unknown" };
const COUNT_COLORS = { car: "#f5a623", bus: "#5b8def", truck: "#b36bf2", motorcycle: "#3cbefa", bicycle: "#c8c83c", person: "#3dd68c" };

const $ = (id) => document.getElementById(id);
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

let INDEX = null;
const cache = {};
let riskChart = null;
let countChart = null;

async function loadVideo(stem) {
  if (!cache[stem]) cache[stem] = await (await fetch(`assets/${stem}.json`)).json();
  return cache[stem];
}

function heroStats() {
  const vids = INDEX.videos;
  const events = vids.reduce((n, v) => n + Object.values(v.events_by_class).reduce((a, b) => a + b, 0), 0);
  const minutes = vids.reduce((n, v) => n + v.duration, 0) / 60;
  const factors = vids.filter((v) => v.runtime).map((v) => v.runtime.total_sec / v.duration);
  const classes = new Set(vids.flatMap((v) => Object.keys(v.events_by_class)));
  const cards = [
    [vids.length, "sample clips, 3840×2160"],
    [minutes.toFixed(1), "minutes of video analysed"],
    [events, "events in predictions_samples.json"],
    [classes.size, "classes fired on the samples"],
  ];
  if (factors.length) cards.push([`${Math.max(...factors).toFixed(2)}×`, "worst runtime / duration (budget 3×)"]);
  $("heroStats").innerHTML = cards.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
}

function tabs(mount, onPick) {
  mount.innerHTML = "";
  INDEX.videos.forEach((v, i) => {
    const stem = v.video.replace(/\.mp4$/, "");
    const b = document.createElement("button");
    b.type = "button";
    b.role = "tab";
    b.textContent = stem;
    b.setAttribute("aria-selected", i === 0 ? "true" : "false");
    b.addEventListener("click", () => {
      mount.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      onPick(stem);
    });
    mount.appendChild(b);
  });
}

function timeline(mount, data, video) {
  const d = data.duration;
  const labels = Object.keys(data.events_by_class);
  const pct = (t) => `${(100 * t / d).toFixed(3)}%`;
  let html = "";
  for (const [lid, runs] of Object.entries(data.signal)) {
    html += `<div class="lane"><div class="lane-name">signal ${lid}</div><div class="lane-track">` +
      runs.map(([a, b, s]) => `<div class="seg" style="left:${pct(a)};width:${pct(Math.max(b - a, 0.3))};background:${SIGNAL_COLORS[s]}" title="${SIGNAL_NAMES[s]} ${a.toFixed(1)}–${b.toFixed(1)} s" data-t="${a}"></div>`).join("") +
      `</div></div>`;
  }
  for (const lab of labels) {
    html += `<div class="lane"><div class="lane-name">${lab}</div><div class="lane-track">` +
      data.events.filter((e) => e[2] === lab).map(([a, b]) =>
        `<div class="seg" style="left:${pct(a)};width:${pct(Math.max(b - a, 0.3))};background:${EVENT_COLORS[lab] || "#888"}" title="${lab} ${a.toFixed(1)}–${b.toFixed(1)} s" data-t="${a}"></div>`).join("") +
      `</div></div>`;
  }
  const step = d > 240 ? 60 : d > 90 ? 30 : 10;
  let ticks = "";
  for (let t = 0; t <= d; t += step) ticks += `<span style="left:${pct(t)}">${fmt(t)}</span>`;
  html += `<div class="axis"><div></div><div>${ticks}</div></div>`;
  mount.innerHTML = html;
  mount.querySelectorAll(".lane-track").forEach((track) => {
    const head = document.createElement("div");
    head.className = "playhead";
    track.appendChild(head);
    track.addEventListener("click", (e) => {
      const seg = e.target.closest(".seg");
      const r = track.getBoundingClientRect();
      video.currentTime = seg ? Number(seg.dataset.t) : ((e.clientX - r.left) / r.width) * d;
      video.play().catch(() => {});
    });
  });
  video.ontimeupdate = () => mount.querySelectorAll(".playhead").forEach((h) => { h.style.left = pct(video.currentTime); });
}

function lineChart(existing, canvas, datasets, maxY, threshold) {
  if (existing) existing.destroy();
  const grid = css("--border"), tick = css("--text-dim");
  const plugins = { legend: { display: datasets.length > 1, labels: { color: tick, boxWidth: 10 } } };
  const opts = {
    animation: false, maintainAspectRatio: false, parsing: false, normalized: true,
    scales: {
      x: { type: "linear", ticks: { color: tick, callback: (v) => fmt(v) }, grid: { color: grid } },
      y: { min: 0, max: maxY, ticks: { color: tick }, grid: { color: grid } },
    },
    plugins, elements: { point: { radius: 0 } },
  };
  if (threshold != null) {
    datasets = datasets.concat([{ label: "threshold", data: [{ x: 0, y: threshold }, { x: 1e6, y: threshold }],
      borderColor: tick, borderDash: [5, 4], borderWidth: 1 }]);
    opts.scales.x.max = Math.max(...datasets[0].data.map((p) => p.x));
    plugins.legend.display = false;
  }
  return new Chart(canvas, { type: "line", data: { datasets }, options: opts });
}

async function showResult(stem) {
  const data = await loadVideo(stem);
  const video = $("resVideo");
  video.src = `assets/${stem}.mp4#t=0.1`;
  const light = Object.values(data.signal_mean_sec)[0] || {};
  $("resMeta").innerHTML = [
    ["Clip", data.video], ["Size", `${data.width}×${data.height} @ ${data.fps} fps`], ["Duration", `${data.duration.toFixed(1)} s`],
    ["Scene alignment", data.align.aligned ? `shift ${data.align.shift_px.join(", ")} px · scale ${data.align.scale}` : "not aligned"],
    ["Red / green phase", light.R ? `${light.R} s / ${light.G} s` : "partial cycles only"],
    ["Tracks", Object.entries(data.tracks_by_class).map(([k, v]) => `${v} ${k}`).join(", ")],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  const counts = Object.entries(data.events_by_class);
  $("resCounts").innerHTML = counts.length
    ? counts.map(([k, v]) => `<span><i style="background:${EVENT_COLORS[k] || "#888"}"></i>${k} · ${v}</span>`).join("")
    : "<span>No events.</span>";
  timeline($("resTimeline"), data, video);
  riskChart = lineChart(riskChart, $("riskChart"),
    [{ label: "risk", data: data.risk.map(([x, y]) => ({ x, y })), borderColor: css("--red"), borderWidth: 1.2 }], 1, 0.5);
}

async function showEda(stem) {
  const data = await loadVideo(stem);
  const t = data.counts.car.map((_, i) => i);
  countChart = lineChart(countChart, $("countChart"), Object.entries(data.counts).map(([k, arr]) => ({
    label: k, data: arr.map((y, i) => ({ x: t[i], y })), borderColor: COUNT_COLORS[k] || "#888", borderWidth: 1.2,
  })), undefined);
  const d = data.duration;
  const [lid, runs] = Object.entries(data.signal)[0] || ["", []];
  $("signalBar").innerHTML = `<div class="timeline"><div class="lane"><div class="lane-name">signal ${lid}</div><div class="lane-track">` +
    runs.map(([a, b, s]) => `<div class="seg" style="left:${100 * a / d}%;width:${100 * Math.max(b - a, 0.3) / d}%;background:${SIGNAL_COLORS[s]}" title="${SIGNAL_NAMES[s]} ${a.toFixed(1)}–${b.toFixed(1)} s"></div>`).join("") +
    `</div></div></div>`;
  const m = data.signal_mean_sec[lid] || {};
  $("signalNote").textContent = m.R ? `Full phases in this clip: red ${m.R} s, green ${m.G} s on average. Amber and blinking green are the short gaps.`
    : "This clip holds no complete red and green phase.";
  $("heatV").src = `assets/${stem}_heat_vehicles.jpg`;
  $("heatP").src = `assets/${stem}_heat_people.jpg`;
  $("flowImg").src = `assets/${stem}_flow.jpg`;
}

function edaTable() {
  const rows = INDEX.videos.map((v) => {
    const s = Object.values(v.signal_mean_sec)[0] || {};
    const light = v.brightness > 110 ? "day, sun" : v.brightness > 75 ? "day, shade" : "evening";
    const tr = v.tracks_by_class;
    return `<tr><td>${v.video}</td><td class="num">${v.width}×${v.height}</td><td class="num">${v.fps}</td>` +
      `<td class="num">${v.duration.toFixed(1)} s</td><td>${light}</td>` +
      `<td class="num">${v.align.aligned ? v.align.shift_px.join(", ") : "–"}</td>` +
      `<td class="num">${(tr.car || 0) + (tr.bus || 0) + (tr.truck || 0) + (tr.motorcycle || 0)}</td>` +
      `<td class="num">${v.pedestrians_on_foot}</td><td class="num">${v.peak_vehicles}</td>` +
      `<td class="num">${s.R ? `${s.R} / ${s.G}` : "–"}</td></tr>`;
  }).join("");
  $("edaTable").innerHTML = `<thead><tr><th>Clip</th><th>Resolution</th><th>fps</th><th>Duration</th><th>Lighting</th>` +
    `<th>Shift vs reference (px, 720p)</th><th>Vehicle tracks</th><th>Pedestrians on foot</th><th>Peak vehicles / s</th>` +
    `<th>Red / green (s)</th></tr></thead><tbody>${rows}</tbody>`;
}

function runtimeTable() {
  const rows = INDEX.videos.filter((v) => v.runtime).map((v) => {
    const r = v.runtime;
    return `<tr><td>${v.video}</td><td class="num">${v.duration.toFixed(1)} s</td><td class="num">${r.part_a_sec} s</td>` +
      `<td class="num">${r.part_b_sec} s</td><td class="num">${r.total_sec} s</td><td class="num">${r.budget_sec} s</td>` +
      `<td class="num">${(r.total_sec / v.duration).toFixed(2)}×</td></tr>`;
  }).join("");
  $("runtimeTable").innerHTML = rows
    ? `<thead><tr><th>Clip</th><th>Duration</th><th>Part A</th><th>Part B</th><th>Total</th><th>Budget</th><th>Total / duration</th></tr></thead><tbody>${rows}</tbody>`
    : "<tbody><tr><td>Measured on an Apple M-series laptop; see the harness log in predictions_samples.json.</td></tr></tbody>";
}

function gallery() {
  const ex = Object.entries(INDEX.examples);
  $("gallery").innerHTML = ex.map(([lab, e]) =>
    `<figure><img loading="lazy" src="assets/example_${lab}.jpg" alt="${lab} in ${e.video}">` +
    `<figcaption><b>${lab}</b> · ${e.video} · ${e.start.toFixed(1)}–${e.end.toFixed(1)} s</figcaption></figure>`).join("");
}

function team() {
  $("people").innerHTML = TEAM.map((p) =>
    `<div class="card"><h3>${p.name}</h3><p class="role">${p.role}</p><p>${p.did}</p>` +
    `<div class="links">${Object.entries(p.links).map(([k, u]) => `<a href="${u}">${k}</a>`).join("")}</div></div>`).join("");
}

function demo() {
  if (DEMO_URL) {
    $("demoMount").innerHTML = `<p><a class="btn" href="${DEMO_URL}" target="_blank" rel="noopener">Open the demo in a new tab</a></p>` +
      `<iframe class="demo-frame" src="${DEMO_URL}" title="PDPU live demo" allow="clipboard-write"></iframe>`;
    $("demoLink").innerHTML = `<a href="${DEMO_URL}">Live demo</a>: Hugging Face Space, CPU`;
  } else {
    $("demoMount").innerHTML = `<div class="card"><p>The hosted demo is being deployed. Meanwhile it runs locally in one command:</p>` +
      `<p><code>pip install -r requirements.txt gradio && python demo/app.py</code></p></div>`;
  }
}

(async function init() {
  team();
  demo();
  try {
    INDEX = await (await fetch("assets/index.json")).json();
  } catch (e) {
    $("heroStats").innerHTML = `<p class="note">Results are loading from assets/ and are not available right now.</p>`;
    return;
  }
  heroStats();
  edaTable();
  runtimeTable();
  gallery();
  const first = INDEX.videos[0].video.replace(/\.mp4$/, "");
  tabs($("resultTabs"), showResult);
  tabs($("edaTabs"), showEda);
  showResult(first);
  showEda(first);
})();
