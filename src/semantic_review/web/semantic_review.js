"use strict";

const LABEL_NAMES = { palm: "PALM", non_palm: "NON-PALM", ambiguous: "AMBIGUOUS", "": "— (unlabeled)" };
const KEY_LABELS = { p: "palm", n: "non_palm", a: "ambiguous" };

const state = {
  total: 0,
  item: null,
  busy: false,
  armedChange: null,
  showRawCrop: false,
  undoStack: [],
  preloaded: new Map(),
};

const $ = (id) => document.getElementById(id);

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "status " + (kind || "");
}

async function api(path, options) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  const payload = await response.json().catch(() => ({ error: response.statusText }));
  if (!response.ok) {
    const error = new Error(payload.error || response.statusText);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function imageUrl(kind, position) {
  return `/image/${kind}/${position}.jpg`;
}

function preload(position) {
  if (position === null || position === undefined || state.preloaded.has(position)) return;
  const images = ["context", "crop", "crop_raw"].map((kind) => {
    const img = new Image();
    img.src = imageUrl(kind, position);
    return img;
  });
  state.preloaded.set(position, images);
  if (state.preloaded.size > 12) state.preloaded.delete(state.preloaded.keys().next().value);
}

function renderCrop() {
  const item = state.item;
  if (!item) return;
  $("img-crop").src = imageUrl(state.showRawCrop ? "crop_raw" : "crop", item.position);
  $("crop-caption").textContent = state.showRawCrop
    ? "Enlarged crop · raw pixels (ticks mark target extent) · [H] show box"
    : "Enlarged crop · target box · [H] raw pixels";
}

function renderItem() {
  const item = state.item;
  state.armedChange = null;
  $("completed").textContent = `Completed: ${item.completed} / ${item.total}`;
  $("position").textContent = `Item ${item.position + 1} of ${item.total} · ${item.display_id}`;
  $("img-context").src = imageUrl("context", item.position);
  renderCrop();
  const labelEl = $("human-label");
  labelEl.textContent = LABEL_NAMES[item.human_label];
  labelEl.className = item.human_label;
  document.querySelectorAll(".labels button").forEach((button) => {
    button.classList.toggle("current", button.dataset.label === item.human_label);
  });
  preload(item.next_unlabeled);
  if (item.position + 1 < item.total) preload(item.position + 1);
}

async function show(position) {
  if (position === null || position === undefined) return;
  const item = await api(`/api/item?position=${position}`);
  state.item = item;
  renderItem();
}

async function withBusy(fn) {
  if (state.busy) return;
  state.busy = true;
  try {
    await fn();
  } catch (error) {
    setStatus(`Error: ${error.message}`, "err");
  } finally {
    state.busy = false;
  }
}

async function postLabel(item, label, confirmChange) {
  return api("/api/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      position: item.position,
      display_id: item.display_id,
      label: label,
      expected_current: item.human_label,
      confirm_change: confirmChange,
    }),
  });
}

function applyLabel(label) {
  return withBusy(async () => {
    const item = state.item;
    if (!item) return;
    const current = item.human_label;
    if (label === current) {
      state.armedChange = null;
      setStatus(`Already ${LABEL_NAMES[label]} — nothing changed`, "warn");
      return;
    }
    if (current) {
      if (state.armedChange !== label) {
        state.armedChange = label;
        setStatus(`Change ${LABEL_NAMES[current]} → ${LABEL_NAMES[label]}? Press the same key again to confirm.`, "warn");
        return;
      }
      const result = await postLabel(item, label, true);
      state.undoStack.push({ position: item.position, display_id: item.display_id, previous: current, next: label });
      state.item = result;
      renderItem();
      setStatus(`✓ Changed ${item.display_id}: ${LABEL_NAMES[current]} → ${LABEL_NAMES[label]} (saved)`, "ok");
      return;
    }
    const result = await postLabel(item, label, false);
    state.undoStack.push({ position: item.position, display_id: item.display_id, previous: "", next: label });
    setStatus(`✓ Saved ${item.display_id}: ${LABEL_NAMES[label]}`, "ok");
    if (result.next_unlabeled === null) {
      state.item = result;
      renderItem();
      setStatus(`✓ Saved ${item.display_id}: ${LABEL_NAMES[label]} — all ${result.total} items are labeled.`, "ok");
      return;
    }
    await show(result.next_unlabeled);
  });
}

function undo() {
  return withBusy(async () => {
    const last = state.undoStack[state.undoStack.length - 1];
    if (!last) {
      setStatus("Nothing to undo in this session", "warn");
      return;
    }
    const target = await api(`/api/item?position=${last.position}`);
    if (target.human_label !== last.next) {
      state.undoStack.pop();
      setStatus(`Cannot undo: ${last.display_id} changed since (now ${LABEL_NAMES[target.human_label]})`, "err");
      return;
    }
    const message = `Revert ${last.display_id} from ${LABEL_NAMES[last.next]} to ${LABEL_NAMES[last.previous]}?`;
    if (!window.confirm(message)) return;
    const result = await postLabel(target, last.previous, true);
    state.undoStack.pop();
    state.item = result;
    renderItem();
    setStatus(`✓ Reverted ${last.display_id} to ${LABEL_NAMES[last.previous]} (saved)`, "ok");
  });
}

function navigate(getPosition) {
  return withBusy(async () => {
    const item = state.item;
    if (!item) return;
    const position = getPosition(item);
    if (position === null || position === undefined) {
      setStatus("No unlabeled items left", "ok");
      return;
    }
    setStatus("", "");
    await show(position);
  });
}

const goPrevious = () => navigate((item) => Math.max(0, item.position - 1));
const goNextUnlabeled = () => navigate((item) => item.next_unlabeled);
const goSkip = () => navigate((item) => (item.position + 1) % item.total);

function toggleCrop() {
  state.showRawCrop = !state.showRawCrop;
  renderCrop();
}

document.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (KEY_LABELS[key]) {
    event.preventDefault();
    applyLabel(KEY_LABELS[key]);
  } else if (key === "ArrowLeft") {
    event.preventDefault();
    goPrevious();
  } else if (key === "ArrowRight") {
    event.preventDefault();
    goNextUnlabeled();
  } else if (key === "s") {
    goSkip();
  } else if (key === "u") {
    undo();
  } else if (key === "h") {
    toggleCrop();
  }
});

document.querySelectorAll(".labels button").forEach((button) => {
  button.addEventListener("click", () => applyLabel(button.dataset.label));
});
$("btn-prev").addEventListener("click", goPrevious);
$("btn-next").addEventListener("click", goNextUnlabeled);
$("btn-skip").addEventListener("click", goSkip);
$("btn-undo").addEventListener("click", undo);
document.querySelectorAll("button").forEach((button) => {
  button.addEventListener("mouseup", () => button.blur());
});

(async function init() {
  try {
    const s = await api("/api/state");
    state.total = s.total;
    $("mode-title").textContent = `· ${s.mode_title} · reviewer ${s.reviewer}`;
    await show(s.first_unlabeled === null ? 0 : s.first_unlabeled);
    if (s.first_unlabeled === null) setStatus(`All ${s.total} items are labeled.`, "ok");
  } catch (error) {
    setStatus(`Error: ${error.message}`, "err");
  }
})();
