const state = {
  ws: null,
  connected: false,
  running: false,
  targetClass: "left_hand",
  classes: ["left_hand", "right_hand", "feet", "tongue"],
  trainedClasses: ["left_hand", "right_hand", "feet", "tongue"],
  channels: ["F3", "F4", "C3", "C4", "Cz", "P3", "P4", "Oz"],
  eegSeries: {},
  maxWaveSamples: 500,
  probabilityHistory: {
    left_hand: [],
    right_hand: [],
    feet: [],
    tongue: [],
  },
};

const waveCanvas = document.getElementById("eeg-canvas");
const waveCtx = waveCanvas.getContext("2d");
const trendCanvas = document.getElementById("trend-canvas");
const trendCtx = trendCanvas.getContext("2d");

const classButtonsContainer = document.getElementById("class-buttons");
const connectionDot = document.getElementById("connection-dot");
const connectionText = document.getElementById("connection-text");
const runningMetric = document.getElementById("metric-running");
const qualityMetric = document.getElementById("metric-quality");
const targetMetric = document.getElementById("metric-target");
const rateMetric = document.getElementById("metric-rate");
const trainedClassesMetric = document.getElementById("metric-trained-classes");

const predictedClassEl = document.getElementById("predicted-class");
const predictedConfidenceEl = document.getElementById("predicted-confidence");
const latencyEl = document.getElementById("latency");
const uncertainTag = document.getElementById("uncertain-tag");
const meterFill = document.getElementById("meter-fill");
const probabilityBars = document.getElementById("probability-bars");

const tabA = document.getElementById("tab-a");
const tabB = document.getElementById("tab-b");
const screenA = document.getElementById("screen-a");
const screenB = document.getElementById("screen-b");

init();

function init() {
  seedSeriesBuffers();
  resetProbabilityHistory();
  setupTabs();
  setupControls();
  renderClassButtons();
  renderProbabilityBars();
  connectSocket();
  requestAnimationFrame(renderLoop);
}

function seedSeriesBuffers() {
  state.eegSeries = {};
  state.channels.forEach((channel) => {
    state.eegSeries[channel] = [];
  });
}

function resetProbabilityHistory() {
  state.probabilityHistory = {};
  state.classes.forEach((cls) => {
    state.probabilityHistory[cls] = [];
  });
}

function setupTabs() {
  tabA.addEventListener("click", () => setScreen("a"));
  tabB.addEventListener("click", () => setScreen("b"));
}

function setScreen(which) {
  const isA = which === "a";
  tabA.classList.toggle("active", isA);
  tabB.classList.toggle("active", !isA);
  screenA.classList.toggle("active", isA);
  screenB.classList.toggle("active", !isA);
}

function setupControls() {
  document.getElementById("start-btn").addEventListener("click", async () => {
    const seed = Number(document.getElementById("seed-input").value);
    await postJson("/api/session/start", { seed: Number.isFinite(seed) ? seed : null });
  });

  document.getElementById("stop-btn").addEventListener("click", async () => {
    await postJson("/api/session/stop", {});
  });

  document.getElementById("restart-seeded").addEventListener("click", async () => {
    const seed = Number(document.getElementById("seed-input").value);
    await postJson("/api/session/start", { seed: Number.isFinite(seed) ? seed : null });
  });
}

function renderClassButtons() {
  classButtonsContainer.innerHTML = "";
  state.classes.forEach((cls) => {
    const button = document.createElement("button");
    button.className = "btn btn-ghost class-pill";
    if (!state.trainedClasses.includes(cls)) {
      button.classList.add("untrained");
    }
    button.textContent = cls.replace("_", " ");
    button.dataset.className = cls;
    button.addEventListener("click", async () => {
      await fetch(`/api/session/class/${cls}`, { method: "POST" });
    });
    classButtonsContainer.appendChild(button);
  });
  updateClassButtonState();
}

function updateClassButtonState() {
  const buttons = classButtonsContainer.querySelectorAll("button");
  buttons.forEach((button) => {
    button.classList.toggle("active", button.dataset.className === state.targetClass);
  });
  targetMetric.textContent = state.targetClass;
  trainedClassesMetric.textContent = state.trainedClasses.join(", ");
}

function updateConnectionUi() {
  connectionDot.classList.toggle("online", state.connected);
  connectionDot.classList.toggle("offline", !state.connected);
  connectionText.textContent = state.connected ? "Connected" : "Disconnected";
}

function connectSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${window.location.host}/ws/live`);
  state.ws = ws;

  ws.addEventListener("open", () => {
    state.connected = true;
    updateConnectionUi();
    ws.send("ping");
  });

  ws.addEventListener("close", () => {
    state.connected = false;
    updateConnectionUi();
    setTimeout(connectSocket, 1200);
  });

  ws.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    processEvent(payload);
  });
}

function processEvent(payload) {
  if (payload.type === "hello") {
    state.running = payload.running;
    state.targetClass = payload.targetClass;
    state.classes = payload.classes;
    state.trainedClasses = payload.trainedClasses || payload.classes;
    state.channels = payload.channels;
    seedSeriesBuffers();
    resetProbabilityHistory();
    renderClassButtons();
    renderProbabilityBars();
    runningMetric.textContent = state.running ? "running" : "stopped";
    rateMetric.textContent = `${payload.sampleRate} Hz`;
  }

  if (payload.type === "status") {
    state.running = payload.running;
    runningMetric.textContent = state.running ? "running" : "stopped";
  }

  if (payload.type === "target_class") {
    state.targetClass = payload.targetClass;
    updateClassButtonState();
  }

  if (payload.type === "eeg_frame") {
    qualityMetric.textContent = `${Math.round(payload.quality * 100)}%`;
    appendChunk(payload);
  }

  if (payload.type === "prediction") {
    updatePrediction(payload);
  }
}

function appendChunk(payload) {
  payload.channels.forEach((channel, idx) => {
    const chunk = payload.samples[idx] || [];
    const series = state.eegSeries[channel] || [];
    series.push(...chunk);
    if (series.length > state.maxWaveSamples) {
      series.splice(0, series.length - state.maxWaveSamples);
    }
    state.eegSeries[channel] = series;
  });
}

function updatePrediction(payload) {
  const conf = payload.confidence || 0;
  predictedClassEl.textContent = (payload.predictedClass || "unknown").toUpperCase();
  predictedConfidenceEl.textContent = `${Math.round(conf * 100)}%`;
  latencyEl.textContent = `${payload.latencyMs.toFixed(1)} ms`;
  meterFill.style.width = `${Math.min(100, Math.max(0, conf * 100))}%`;
  uncertainTag.style.display = conf < 0.55 ? "block" : "none";

  state.classes.forEach((cls) => {
    const val = payload.probabilities[cls] || 0;
    const trend = state.probabilityHistory[cls];
    trend.push(val);
    if (trend.length > 120) {
      trend.shift();
    }
  });

  updateProbabilityBars(payload.probabilities);
}

function renderProbabilityBars() {
  probabilityBars.innerHTML = "";
  state.classes.forEach((cls) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span>${cls.replace("_", " ")}</span>
      <div class="prob-track"><div class="prob-fill" data-cls="${cls}" style="width:0%"></div></div>
      <span data-label="${cls}">0%</span>
    `;
    probabilityBars.appendChild(row);
  });
}

function updateProbabilityBars(probabilities) {
  state.classes.forEach((cls) => {
    const val = probabilities[cls] || 0;
    const fill = probabilityBars.querySelector(`.prob-fill[data-cls='${cls}']`);
    const label = probabilityBars.querySelector(`span[data-label='${cls}']`);
    if (fill) {
      fill.style.width = `${Math.round(val * 100)}%`;
    }
    if (label) {
      label.textContent = `${Math.round(val * 100)}%`;
    }
  });
}

function renderLoop() {
  drawWaveforms();
  drawTrend();
  requestAnimationFrame(renderLoop);
}

function drawWaveforms() {
  const w = waveCanvas.width;
  const h = waveCanvas.height;
  waveCtx.clearRect(0, 0, w, h);

  waveCtx.fillStyle = "#0b1a28";
  waveCtx.fillRect(0, 0, w, h);

  const count = state.channels.length;
  const laneHeight = h / count;
  const maxSamples = state.maxWaveSamples;

  state.channels.forEach((channel, idx) => {
    const yCenter = laneHeight * idx + laneHeight / 2;
    const series = state.eegSeries[channel] || [];

    waveCtx.strokeStyle = "rgba(80, 127, 162, 0.45)";
    waveCtx.beginPath();
    waveCtx.moveTo(0, yCenter);
    waveCtx.lineTo(w, yCenter);
    waveCtx.stroke();

    waveCtx.fillStyle = "#95b7cf";
    waveCtx.font = "12px IBM Plex Mono";
    waveCtx.fillText(channel, 8, yCenter - 8);

    if (series.length < 2) {
      return;
    }

    waveCtx.strokeStyle = "#2dd4bf";
    waveCtx.lineWidth = 1.2;
    waveCtx.beginPath();
    series.forEach((point, sampleIdx) => {
      const x = (sampleIdx / maxSamples) * w;
      const y = yCenter - point * (laneHeight * 0.09);
      if (sampleIdx === 0) {
        waveCtx.moveTo(x, y);
      } else {
        waveCtx.lineTo(x, y);
      }
    });
    waveCtx.stroke();
  });
}

function drawTrend() {
  const w = trendCanvas.width;
  const h = trendCanvas.height;
  trendCtx.clearRect(0, 0, w, h);

  trendCtx.fillStyle = "#0b1a28";
  trendCtx.fillRect(0, 0, w, h);

  trendCtx.strokeStyle = "rgba(75, 121, 154, 0.35)";
  for (let i = 0; i <= 5; i += 1) {
    const y = (h / 5) * i;
    trendCtx.beginPath();
    trendCtx.moveTo(0, y);
    trendCtx.lineTo(w, y);
    trendCtx.stroke();
  }

  const palette = {
    left_hand: "#14b8a6",
    right_hand: "#60a5fa",
    feet: "#f59e0b",
    tongue: "#f472b6",
  };

  state.classes.forEach((cls) => {
    const values = state.probabilityHistory[cls];
    if (!values || values.length < 2) {
      return;
    }

    trendCtx.strokeStyle = palette[cls] || "#e2e8f0";
    trendCtx.lineWidth = 2;
    trendCtx.beginPath();
    values.forEach((value, idx) => {
      const x = (idx / 119) * w;
      const y = h - value * h;
      if (idx === 0) {
        trendCtx.moveTo(x, y);
      } else {
        trendCtx.lineTo(x, y);
      }
    });
    trendCtx.stroke();
  });
}

async function postJson(url, body) {
  await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}
