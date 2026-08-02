/*
 * Sehat — UI orchestration.
 * Wires drag-and-drop, file picker and camera capture to the two inference
 * modes (on-device ONNX runtime, or a clinic-run remote server), renders
 * results with the mandatory disclaimer, and maintains the session history.
 */

import {
  predictOnDevice,
  ModelMissingError,
  RuntimeUnavailableError,
} from './inference.js';
import { predictRemote, ApiError, OfflineError } from './api.js';
import { loadConfig, saveConfig, loadHistory, addHistoryEntry, clearHistory } from './store.js';

const DISCLAIMER = 'Decision-support only. Not a medical diagnosis. Confirm with a qualified radiologist.';
const LABEL_ELEVATED = 'Elevated TB likelihood — refer for confirmation';
const LABEL_OK = 'No strong TB signal';

const PRIVACY_ON_DEVICE = 'Images are analysed on this device and are never uploaded or stored.';
const PRIVACY_REMOTE = 'Images are sent to your clinic server for analysis. This app never stores images.';

const MODE_NAMES = { 'on-device': 'On-device', remote: 'Remote' };

const PROGRESS_MESSAGES = {
  runtime: 'Fetching inference runtime (one-time download, then works offline)…',
  model: 'Loading screening model…',
  analyse: 'Analysing image…',
};

const $ = (id) => document.getElementById(id);

const els = {
  modeOnDevice: $('mode-on-device'),
  modeRemote: $('mode-remote'),
  endpointField: $('endpoint-field'),
  endpointInput: $('endpoint-input'),
  thresholdInput: $('threshold-input'),
  thresholdValue: $('threshold-value'),
  privacyNote: $('privacy-note'),
  dropzone: $('dropzone'),
  pickButton: $('pick-button'),
  cameraButton: $('camera-button'),
  fileInput: $('file-input'),
  cameraInput: $('camera-input'),
  preview: $('preview'),
  previewImage: $('preview-image'),
  previewName: $('preview-name'),
  clearImageButton: $('clear-image-button'),
  statusPanel: $('status-panel'),
  statusText: $('status-text'),
  errorBox: $('error-box'),
  errorTitle: $('error-title'),
  errorMessage: $('error-message'),
  errorLink: $('error-link'),
  resultPanel: $('result-panel'),
  resultHeading: $('result-heading'),
  resultCard: $('result-card'),
  resultProbability: $('result-probability'),
  resultLabel: $('result-label'),
  resultThreshold: $('result-threshold'),
  resultMode: $('result-mode'),
  resultTime: $('result-time'),
  resultLatency: $('result-latency'),
  resultDisclaimer: $('result-disclaimer'),
  historyList: $('history-list'),
  historyEmpty: $('history-empty'),
  clearHistoryButton: $('clear-history-button'),
};

let config = loadConfig();
let currentImage = null; // { file, source, previewUrl }
let busy = false;

const dayFormat = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' });
const timeFormat = new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });

function formatTimestamp(date) {
  return `${dayFormat.format(date)} · ${timeFormat.format(date)}`;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatLatency(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function labelFor(probability, threshold) {
  return probability >= threshold ? LABEL_ELEVATED : LABEL_OK;
}

function setStatus(message) {
  els.statusText.textContent = message;
  els.statusPanel.hidden = false;
}

function hideStatus() {
  els.statusPanel.hidden = true;
}

function setBusy(nextBusy) {
  busy = nextBusy;
  els.pickButton.disabled = nextBusy;
  els.cameraButton.disabled = nextBusy;
  els.clearImageButton.disabled = nextBusy;
}

function hideError() {
  els.errorBox.hidden = true;
  els.errorLink.hidden = true;
}

function showError(err) {
  hideStatus();
  let title = 'Screening could not be completed';
  let message = 'An unexpected error occurred. Please try again.';
  let showModelLink = false;

  if (err instanceof ModelMissingError) {
    title = 'Model file not found';
    message = err.message;
    showModelLink = true;
  } else if (err instanceof RuntimeUnavailableError) {
    title = 'Inference runtime unavailable';
    message = err.message;
  } else if (err instanceof OfflineError) {
    title = 'No connection';
    message = err.message;
  } else if (err instanceof ApiError) {
    title = 'Server error';
    message = err.message;
  } else if (err && err.message) {
    message = err.message;
  }

  els.errorTitle.textContent = title;
  els.errorMessage.textContent = message;
  els.errorLink.hidden = !showModelLink;
  els.errorBox.hidden = false;
  els.errorBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function applyConfigToUI() {
  els.modeOnDevice.checked = config.mode === 'on-device';
  els.modeRemote.checked = config.mode === 'remote';
  els.endpointField.hidden = config.mode !== 'remote';
  els.endpointInput.value = config.endpoint;
  els.thresholdInput.value = String(config.threshold);
  els.thresholdValue.textContent = formatPercent(config.threshold);
  els.privacyNote.textContent = config.mode === 'remote' ? PRIVACY_REMOTE : PRIVACY_ON_DEVICE;
}

function renderHistory() {
  const entries = loadHistory();
  els.historyList.textContent = '';
  els.historyEmpty.hidden = entries.length > 0;

  for (const entry of entries) {
    const item = document.createElement('li');
    item.className = 'history-item';

    const time = document.createElement('span');
    time.className = 'history-time';
    time.textContent = formatTimestamp(new Date(entry.ts));

    const prob = document.createElement('span');
    prob.className = 'history-prob';
    prob.textContent = formatPercent(entry.probability);

    const label = document.createElement('span');
    const elevated = entry.label === LABEL_ELEVATED || /elevated/i.test(entry.label);
    label.className = `history-label ${elevated ? 'is-elevated' : 'is-ok'}`;
    label.textContent = entry.label;

    const mode = document.createElement('span');
    mode.className = 'history-mode';
    mode.textContent = MODE_NAMES[entry.mode] || entry.mode;

    item.append(time, prob, label, mode);
    els.historyList.append(item);
  }
}

function renderResult({ probability, label, threshold, mode, latencyMs, disclaimer }) {
  const elevated = probability >= threshold;
  els.resultCard.classList.toggle('is-elevated', elevated);
  els.resultProbability.textContent = (probability * 100).toFixed(1);
  els.resultLabel.textContent = label;
  els.resultThreshold.textContent = formatPercent(threshold);
  els.resultMode.textContent = MODE_NAMES[mode] || mode;
  els.resultTime.textContent = formatTimestamp(new Date());
  els.resultLatency.textContent = formatLatency(latencyMs);
  els.resultDisclaimer.textContent = disclaimer || DISCLAIMER;
  els.resultPanel.hidden = false;
  els.resultHeading.focus({ preventScroll: true });
  els.resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function decodeImage(file) {
  if ('createImageBitmap' in window) {
    try {
      return await createImageBitmap(file);
    } catch {
      // Fall through to <img> decoding below.
    }
  }
  return null;
}

async function handleFile(file) {
  if (!file) return;
  if (!file.type || !file.type.startsWith('image/')) {
    showError(new Error('That file is not an image. Please choose a PNG or JPEG chest X-ray.'));
    return;
  }

  clearCurrentImage();
  hideError();
  els.resultPanel.hidden = true;

  const previewUrl = URL.createObjectURL(file);
  const source = await decodeImage(file);
  currentImage = { file, source: source || null, previewUrl };

  els.previewImage.src = previewUrl;
  els.previewName.textContent = file.name || 'Captured image';
  els.preview.hidden = false;

  await runScreening();
}

function clearCurrentImage() {
  if (currentImage) {
    URL.revokeObjectURL(currentImage.previewUrl);
    if (currentImage.source && typeof currentImage.source.close === 'function') {
      currentImage.source.close();
    }
    currentImage = null;
  }
}

async function runScreening() {
  if (busy || !currentImage) return;
  setBusy(true);
  hideError();
  els.resultPanel.hidden = true;

  const started = performance.now();
  try {
    if (config.mode === 'on-device') {
      if (!currentImage.source) {
        throw new Error('This image could not be decoded. Try a PNG or JPEG export.');
      }
      const onProgress = (phase) => setStatus(PROGRESS_MESSAGES[phase] || PROGRESS_MESSAGES.analyse);
      setStatus(PROGRESS_MESSAGES.runtime);
      const probability = await predictOnDevice(currentImage.source, onProgress);
      setStatus(PROGRESS_MESSAGES.analyse);
      const latencyMs = performance.now() - started;
      const label = labelFor(probability, config.threshold);
      const result = {
        probability,
        label,
        threshold: config.threshold,
        mode: 'on-device',
        latencyMs,
        disclaimer: DISCLAIMER,
      };
      renderResult(result);
      addHistoryEntry({ ts: new Date().toISOString(), probability, label, mode: 'on-device' });
    } else {
      setStatus('Uploading to the clinic server…');
      const data = await predictRemote(currentImage.file, { endpoint: config.endpoint });
      renderResult({
        probability: data.probability,
        label: data.label,
        threshold: data.threshold,
        mode: 'remote',
        latencyMs: data.latency_ms,
        disclaimer: data.disclaimer,
      });
      addHistoryEntry({
        ts: new Date().toISOString(),
        probability: data.probability,
        label: data.label,
        mode: 'remote',
      });
    }
    renderHistory();
    hideStatus();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

function wireEvents() {
  els.modeOnDevice.addEventListener('change', () => {
    config = saveConfig({ mode: 'on-device' });
    applyConfigToUI();
  });
  els.modeRemote.addEventListener('change', () => {
    config = saveConfig({ mode: 'remote' });
    applyConfigToUI();
  });

  els.endpointInput.addEventListener('change', () => {
    const endpoint = els.endpointInput.value.trim();
    config = saveConfig(endpoint ? { endpoint } : {});
    applyConfigToUI();
  });

  els.thresholdInput.addEventListener('input', () => {
    els.thresholdValue.textContent = formatPercent(Number(els.thresholdInput.value));
  });
  els.thresholdInput.addEventListener('change', () => {
    config = saveConfig({ threshold: Number(els.thresholdInput.value) });
    applyConfigToUI();
  });

  els.pickButton.addEventListener('click', () => els.fileInput.click());
  els.cameraButton.addEventListener('click', () => els.cameraInput.click());

  const onInputChange = (input) => {
    handleFile(input.files && input.files[0]);
    input.value = '';
  };
  els.fileInput.addEventListener('change', () => onInputChange(els.fileInput));
  els.cameraInput.addEventListener('change', () => onInputChange(els.cameraInput));

  els.dropzone.addEventListener('dragover', (event) => {
    event.preventDefault();
    els.dropzone.classList.add('is-dragover');
  });
  els.dropzone.addEventListener('dragleave', () => {
    els.dropzone.classList.remove('is-dragover');
  });
  els.dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    els.dropzone.classList.remove('is-dragover');
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    handleFile(file);
  });

  // Stop the browser from navigating away when a file is dropped outside the dropzone.
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => event.preventDefault());

  els.clearImageButton.addEventListener('click', () => {
    clearCurrentImage();
    els.preview.hidden = true;
    els.previewImage.removeAttribute('src');
    els.resultPanel.hidden = true;
    hideError();
  });

  els.clearHistoryButton.addEventListener('click', () => {
    if (loadHistory().length === 0) return;
    if (window.confirm('Clear all screening results from this session?')) {
      clearHistory();
      renderHistory();
    }
  });
}

function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch((err) => {
      console.warn('Service worker registration failed; offline support is unavailable.', err);
    });
  });
}

function init() {
  applyConfigToUI();
  renderHistory();
  wireEvents();
  registerServiceWorker();
}

init();
