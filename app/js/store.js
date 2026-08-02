/*
 * Tiny localStorage persistence for Sehat.
 *
 * PRIVACY: only user preferences (mode, endpoint, threshold) and screening
 * metadata (timestamp, probability, label, mode) are ever written here.
 * X-ray images are NEVER persisted — they exist in memory only for the
 * duration of a single screening.
 *
 * All access is defensive: if storage is unavailable (private browsing,
 * storage disabled) the app silently falls back to in-memory defaults.
 */

import { DEFAULT_ENDPOINT } from './api.js';

const CONFIG_KEY = 'sehat.config.v1';
const HISTORY_KEY = 'sehat.history.v1';
const HISTORY_LIMIT = 50;

const VALID_MODES = ['on-device', 'remote'];
const MIN_THRESHOLD = 0.05;
const MAX_THRESHOLD = 0.95;

const DEFAULT_CONFIG = {
  mode: 'on-device',
  endpoint: DEFAULT_ENDPOINT,
  threshold: 0.5,
};

function read(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function write(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

function sanitizeConfig(candidate) {
  const config = { ...DEFAULT_CONFIG };
  if (candidate && typeof candidate === 'object') {
    if (VALID_MODES.includes(candidate.mode)) config.mode = candidate.mode;
    if (typeof candidate.endpoint === 'string' && candidate.endpoint.trim().length > 0) {
      config.endpoint = candidate.endpoint.trim();
    }
    if (typeof candidate.threshold === 'number' && Number.isFinite(candidate.threshold)) {
      config.threshold = Math.min(MAX_THRESHOLD, Math.max(MIN_THRESHOLD, candidate.threshold));
    }
  }
  return config;
}

export function loadConfig() {
  return sanitizeConfig(read(CONFIG_KEY));
}

export function saveConfig(patch) {
  const next = sanitizeConfig({ ...loadConfig(), ...patch });
  write(CONFIG_KEY, next);
  return next;
}

function isValidEntry(entry) {
  return (
    entry &&
    typeof entry.ts === 'string' &&
    typeof entry.probability === 'number' &&
    entry.probability >= 0 &&
    entry.probability <= 1 &&
    typeof entry.label === 'string' &&
    VALID_MODES.includes(entry.mode)
  );
}

export function loadHistory() {
  const raw = read(HISTORY_KEY);
  if (!Array.isArray(raw)) return [];
  return raw.filter(isValidEntry).slice(0, HISTORY_LIMIT);
}

export function addHistoryEntry(entry) {
  if (!isValidEntry(entry)) return loadHistory();
  const next = [entry, ...loadHistory()].slice(0, HISTORY_LIMIT);
  write(HISTORY_KEY, next);
  return next;
}

export function clearHistory() {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch {
    // Storage unavailable: nothing to clear.
  }
}
