/*
 * Optional remote mode for Sehat: POST the X-ray to a clinic-run prediction
 * server as multipart/form-data (field name "file") and validate the exact
 * expected response shape:
 *
 *   { probability, label, threshold, latency_ms, disclaimer }
 *
 * Handles timeouts and offline/unreachable-server conditions with humane,
 * actionable errors. The remote call is never cached by the service worker.
 */

export const DEFAULT_ENDPOINT = 'http://localhost:8000/predict';
const DEFAULT_TIMEOUT_MS = 15000;

export class ApiError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ApiError';
  }
}

export class OfflineError extends Error {
  constructor(message) {
    super(message);
    this.name = 'OfflineError';
  }
}

function validateResponse(data) {
  if (data === null || typeof data !== 'object') {
    throw new ApiError('The server returned an unexpected response (expected a JSON object).');
  }
  const required = ['probability', 'label', 'threshold', 'latency_ms', 'disclaimer'];
  const missing = required.filter((key) => !(key in data));
  if (missing.length > 0) {
    throw new ApiError(`The server response is missing field(s): ${missing.join(', ')}.`);
  }
  if (typeof data.probability !== 'number' || data.probability < 0 || data.probability > 1) {
    throw new ApiError('The server returned an invalid probability (expected a number between 0 and 1).');
  }
  if (typeof data.label !== 'string' || data.label.length === 0) {
    throw new ApiError('The server returned an invalid label.');
  }
  if (typeof data.threshold !== 'number' || data.threshold < 0 || data.threshold > 1) {
    throw new ApiError('The server returned an invalid threshold.');
  }
  if (typeof data.latency_ms !== 'number' || data.latency_ms < 0) {
    throw new ApiError('The server returned an invalid latency_ms.');
  }
  if (typeof data.disclaimer !== 'string') {
    throw new ApiError('The server returned an invalid disclaimer.');
  }
  return data;
}

export async function predictRemote(file, { endpoint = DEFAULT_ENDPOINT, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  if (!endpoint || typeof endpoint !== 'string') {
    throw new ApiError('No prediction endpoint is configured. Add one in Settings.');
  }
  if (navigator.onLine === false) {
    throw new OfflineError(
      'You appear to be offline. Remote mode needs a connection to the clinic server — switch to on-device mode to keep screening.'
    );
  }

  const form = new FormData();
  form.append('file', file, file.name || 'xray');

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(endpoint, { method: 'POST', body: form, signal: controller.signal });
  } catch (err) {
    if (err && err.name === 'AbortError') {
      throw new ApiError(
        `No response from the server within ${Math.round(timeoutMs / 1000)} seconds. Check that it is running and try again.`
      );
    }
    throw new OfflineError(
      'Could not reach the prediction server. Check the endpoint URL and your connection, or switch to on-device mode.'
    );
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    throw new ApiError(`The prediction server responded with HTTP ${response.status}.`);
  }

  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError('The prediction server returned a response that was not valid JSON.');
  }
  return validateResponse(data);
}
