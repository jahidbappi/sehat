/*
 * Sehat service worker — offline-first.
 *
 * Strategy:
 *   - App shell (HTML/CSS/JS/manifest/icon): cache-first from a versioned
 *     shell cache, populated at install time.
 *   - CDN runtime assets (jsDelivr: onnxruntime-web JS + WASM) and same-origin
 *     files fetched later (including model/model.int8.onnx): cache-first into
 *     a runtime cache, so repeat visits work fully offline.
 *   - Navigations: network-first with a short timeout, so new deployments
 *     reach users on the next visit; falls back to the cached shell when
 *     offline/slow, then to an offline fallback page.
 *   - Prediction calls (any /predict path) are NEVER intercepted or cached.
 *   - sw.js itself is never served from cache; it always comes from the
 *     network (Vercel also sends it with Cache-Control: no-cache).
 *
 * Bump CACHE_VERSION whenever the shell asset list changes; old caches are
 * purged on activation.
 */

const CACHE_VERSION = 'v2';
const SHELL_CACHE = `sehat-shell-${CACHE_VERSION}`;
const RUNTIME_CACHE = `sehat-runtime-${CACHE_VERSION}`;
const KNOWN_CACHES = [SHELL_CACHE, RUNTIME_CACHE];

// Navigations give up on the network after this long and use the cache.
const NAVIGATION_TIMEOUT_MS = 3500;

const SHELL_ASSETS = [
  './',
  './index.html',
  './css/style.css',
  './js/app.js',
  './js/api.js',
  './js/inference.js',
  './js/store.js',
  './manifest.webmanifest',
  './icons/icon.svg',
  './model/README.md',
];

const OFFLINE_FALLBACK_HTML = `<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Sehat — offline</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 32rem; margin: 4rem auto; padding: 0 1rem; line-height: 1.6;">
  <h1>You are offline</h1>
  <p>Sehat has not been cached on this device yet. Connect to the internet once and reload so the app can be stored for offline use.</p>
  <p><strong>Decision-support only. Not a medical diagnosis. Confirm with a qualified radiologist.</strong></p>
</body>
</html>`;

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((key) => !KNOWN_CACHES.includes(key)).map((key) => caches.delete(key)))
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);

  // Inference API calls are never cached.
  if (url.pathname.includes('/predict')) return;

  // The service worker script must always come from the network so updates
  // are picked up immediately.
  if (url.pathname === '/sw.js') return;

  if (request.mode === 'navigate') {
    event.respondWith(handleNavigation(event));
    return;
  }

  const sameOrigin = url.origin === self.location.origin;
  const pinnedCdn = url.origin === 'https://cdn.jsdelivr.net';
  if (sameOrigin || pinnedCdn) {
    event.respondWith(cacheFirst(request));
  }
});

async function handleNavigation(event) {
  const cache = await caches.open(SHELL_CACHE);

  let networkResponse = null;
  try {
    networkResponse = await fetchWithTimeout(event.request, NAVIGATION_TIMEOUT_MS);
  } catch {
    networkResponse = null;
  }

  if (networkResponse && networkResponse.ok) {
    // Refresh the cached shell; waitUntil keeps the worker alive long enough
    // for the update to land without delaying the response.
    event.waitUntil(cache.put('./index.html', networkResponse.clone()));
    return networkResponse;
  }

  const cached = await cache.match('./index.html');
  if (cached) return cached;

  return (
    networkResponse ||
    new Response(OFFLINE_FALLBACK_HTML, {
      headers: { 'Content-Type': 'text/html; charset=utf-8' },
    })
  );
}

function fetchWithTimeout(request, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('navigation fetch timed out')), timeoutMs);
    fetch(request).then(
      (response) => {
        clearTimeout(timer);
        resolve(response);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok || response.type === 'opaque') {
    const cache = await caches.open(RUNTIME_CACHE);
    await cache.put(request, response.clone());
  }
  return response;
}
