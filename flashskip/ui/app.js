const $ = (selector) => document.querySelector(selector);

const form = $("#generationForm");
const output = $("#output");
const button = $("#generateButton");
const buttonLabel = $(".button-label");
const metrics = $("#metrics");
const copyButton = $("#copyButton");
const modelSelect = $("#modelSelect");
const modelMeta = $("#modelMeta");
const modelIntro = $("#modelIntro");
let controller = null;
let generatedText = "";
let switchingModel = false;

function bindRange(inputSelector, outputSelector, format = (value) => value) {
  const input = $(inputSelector);
  const value = $(outputSelector);
  const update = () => { value.textContent = format(input.value); };
  input.addEventListener("input", update);
  update();
}

bindRange("#maxTokens", "#lengthValue");
bindRange("#temperature", "#tempValue", (value) => Number(value).toFixed(1));
bindRange("#topK", "#topKValue");

function applyStatus(data) {
  if (data.available_models) {
    const signature = JSON.stringify(data.available_models);
    if (modelSelect.dataset.models !== signature) {
      modelSelect.replaceChildren(...data.available_models.map(({ key, label }) => new Option(label, key)));
      modelSelect.dataset.models = signature;
    }
  }
  const state = $(".model-state");
  state.classList.remove("error");
  state.classList.add("ready");
  $("#statusText").textContent = `${data.model} · ${data.device}`;
  $("#checkpointName").textContent = `${data.checkpoint} · ${(data.tokens_seen / 1e6).toFixed(2)}M TOKENS`;
  modelSelect.value = data.model_key;
  modelMeta.textContent = `${(data.parameters / 1e6).toFixed(2)}M params · ${(data.tokens_seen / 1e6).toFixed(2)}M tokens`;
  modelIntro.textContent = `${data.model} has ${(data.parameters / 1e6).toFixed(2)}M parameters and is loaded locally on ${data.device}. Every word below is generated on this machine.`;
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Model unavailable");
    applyStatus(await response.json());
    modelSelect.disabled = false;
  } catch (error) {
    const state = $(".model-state");
    state.classList.remove("ready");
    state.classList.add("error");
    $("#statusText").textContent = error.message;
    modelSelect.disabled = true;
  }
}

function setRunning(running) {
  button.disabled = switchingModel;
  modelSelect.disabled = running || switchingModel;
  buttonLabel.textContent = running ? "Stop" : "Generate";
  button.type = running ? "button" : "submit";
  output.classList.toggle("streaming", running);
}

function renderText(text) {
  generatedText = text;
  output.classList.remove("empty", "error");
  output.replaceChildren(document.createTextNode(text));
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  output.appendChild(cursor);
}

async function switchModel(modelKey) {
  if (switchingModel || controller) return;
  switchingModel = true;
  button.disabled = true;
  modelSelect.disabled = true;
  const label = modelSelect.options[modelSelect.selectedIndex]?.textContent || modelKey;
  const state = $(".model-state");
  state.classList.remove("ready", "error");
  $("#statusText").textContent = `Loading ${label}…`;
  modelMeta.textContent = "Switching model…";

  try {
    const response = await fetch("/api/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelKey }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Model switch failed (${response.status})`);
    applyStatus(data);
    metrics.textContent = `${data.model} READY`;
  } catch (error) {
    state.classList.add("error");
    $("#statusText").textContent = error.message;
    await loadStatus();
  } finally {
    switchingModel = false;
    button.disabled = false;
    modelSelect.disabled = false;
  }
}

async function generate() {
  controller = new AbortController();
  generatedText = "";
  renderText("");
  setRunning(true);
  metrics.textContent = "STARTING";
  copyButton.disabled = true;

  const payload = {
    prompt: $("#prompt").value,
    max_new_tokens: Number($("#maxTokens").value),
    temperature: Number($("#temperature").value),
    top_k: Number($("#topK").value),
  };

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) throw new Error(`Generation failed (${response.status})`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.type === "start") metrics.textContent = `${event.prompt_tokens} PROMPT TOKENS`;
        if (event.type === "token") {
          renderText(event.text);
          metrics.textContent = `${event.index} TOKENS · ${event.tokens_per_second.toFixed(1)} TOK/S`;
        }
        if (event.type === "done") {
          renderText(event.text);
          metrics.textContent = `${event.generated_tokens} TOKENS · ${event.tokens_per_second.toFixed(1)} TOK/S`;
        }
        if (event.type === "error") throw new Error(event.message);
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      metrics.textContent = "STOPPED";
    } else {
      output.classList.add("error");
      renderText(error.message);
      output.classList.add("error");
      metrics.textContent = "ERROR";
    }
  } finally {
    controller = null;
    setRunning(false);
    copyButton.disabled = !generatedText;
  }
}

modelSelect.addEventListener("change", () => {
  switchModel(modelSelect.value);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  generate();
});

button.addEventListener("click", () => {
  if (controller) controller.abort();
});

$("#prompt").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    if (controller) controller.abort(); else generate();
  }
});

copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(generatedText);
  copyButton.textContent = "Copied";
  setTimeout(() => { copyButton.textContent = "Copy output"; }, 1200);
});

loadStatus();
