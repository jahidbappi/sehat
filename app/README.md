# Sehat — TB Screening (offline-first clinic PWA)

The face of **Project Sehat**: a static, buildless progressive web app that
screens chest X-rays for TB in low-resource clinics. A nurse with a laptop and
no internet can drag in an X-ray and get a screening probability in seconds.

> **Decision-support only. Not a medical diagnosis. Confirm with a qualified
> radiologist.** This disclaimer is shown persistently in the app and repeated
> on every result.

## Run it

No build step, no dependencies to install — plain HTML/CSS/JS (ES modules).
Serve the folder with any static file server:

```bash
cd app
python3 -m http.server 8000
# then open http://localhost:8000
```

A service worker provides the offline support, so the app must be served over
`http://localhost` (any port) or `https://` — opening `index.html` via
`file://` will not work.

## Deploy to Vercel

The app is static and buildless, so it deploys as-is. Two options:

**(a) Vercel dashboard** — import the repo, set **Root Directory** to `app/`.
No build command and no output directory override are needed; `vercel.json`
ships the header rules (service worker always revalidates, the model file gets
a long-lived immutable cache).

**(b) Vercel CLI:**

```bash
cd app && vercel deploy --prod
```

**Honest caveats for a production deployment:**

- `model/model.int8.onnx` is gitignored (too large for git), so it is **not**
  part of the deploy. On-device inference on the deployed site needs the model
  hosted separately — e.g. Vercel Blob or another external URL that
  `js/inference.js` is pointed at — or the app used in **remote server mode**
  against a hosted `sehat.serving` instance.
- The FastAPI serving service (`sehat.serving`) is better deployed on
  Railway, Render, or Hugging Face Spaces than on Vercel serverless — the
  model artifact size and cold starts are a poor fit for serverless functions.

## Two screening modes

| Mode | How it works | Requirements |
| --- | --- | --- |
| **On-device** (default) | Runs `model/model.int8.onnx` in the browser with onnxruntime-web (WASM). Images never leave the laptop. | One-time internet connection on first load to fetch the WASM runtime (cached afterwards); the exported model placed at `app/model/model.int8.onnx` (see `model/README.md`). |
| **Remote server** | POSTs the image as multipart/form-data to a configurable endpoint (default `http://localhost:8000/predict`). | The clinic prediction server running and reachable. |

The remote endpoint must respond with exactly:

```json
{
  "probability": 0.87,
  "label": "Elevated TB likelihood — refer for confirmation",
  "threshold": 0.5,
  "latency_ms": 42,
  "disclaimer": "Decision-support only. Not a medical diagnosis. Confirm with a qualified radiologist."
}
```

Timeouts (15 s), unreachable servers, malformed responses, and offline states
all produce humane, actionable error messages.

## Privacy guarantees

- **Images are never stored and never uploaded in on-device mode.** They are
  decoded in memory, screened, and discarded. Dragging in another image or
  pressing *Remove image* releases it immediately.
- **Session history stores metadata only** — timestamp, probability, label,
  mode — in `localStorage` on the device. No pixels, no identifiers.
- **No trackers, no analytics, no third-party CSS/JS frameworks.** The only
  network dependency is the pinned, SRI-verified onnxruntime-web runtime from
  jsDelivr (`v1.27.0`), cached locally by the service worker after first load.

## Offline behaviour

- The app shell, the ONNX runtime (JS + WASM), and — once fetched — the model
  file are all cached by the service worker (`sw.js`), so repeat visits work
  fully offline.
- `/predict` calls are never intercepted or cached.
- If the app has never been loaded while online, an offline fallback page
  explains the one-time-connection requirement.

## Preprocessing parity with training

`js/inference.js` mirrors the Python pipeline step for step:

| Training (torchvision) | In-browser (typed arrays only) |
| --- | --- |
| decode as RGB | canvas `drawImage` |
| `Resize((224, 224))` bilinear, squash-to-square | 224×224 canvas draw with smoothing enabled |
| `ToTensor()` → `[0, 1]` | `value / 255` |
| `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` | per-channel `(x - mean) / std` |
| `float32` NCHW | `Float32Array` → `ort.Tensor('float32', …, [1,3,224,224])` |
| single raw logit | `sigmoid(logit)` in JS |

## Accessibility & UX

- Semantic HTML with ARIA live regions; results are announced and focused.
- Full keyboard support: skip link, native radio mode toggle, real buttons,
  visible focus rings, Enter/Space activation.
- WCAG AA contrast in both light and dark (`prefers-color-scheme`) themes.
- Large touch targets (≥ 44–48 px), responsive down to 360 px wide.
- Reduced-motion support; empty/loading/error states for every stage.

## File map

```
app/
├── index.html              # single-page app shell
├── css/style.css           # clinical design system (light + dark)
├── js/app.js               # UI orchestration
├── js/inference.js         # on-device ONNX inference + preprocessing
├── js/api.js               # optional remote /predict client
├── js/store.js             # localStorage config + session history (no images)
├── sw.js                   # offline-first service worker
├── manifest.webmanifest    # installable PWA manifest
├── icons/icon.svg          # original lungs mark
└── model/README.md         # where to drop model.int8.onnx
```
