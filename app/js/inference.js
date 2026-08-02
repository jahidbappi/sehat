/*
 * On-device TB-screening inference for Sehat.
 *
 * Runtime: onnxruntime-web v1.27.0 (pinned), fetched at runtime from jsDelivr:
 *   https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.js
 * The <script> is injected with crossorigin="anonymous" and the SRI integrity
 * hash below (SHA-384, computed against the exact pinned file), so a tampered
 * CDN payload is rejected by the browser. The multi-MB WASM binaries are
 * loaded by the runtime from the same pinned dist/ directory
 * (ort.env.wasm.wasmPaths) and cached by sw.js for repeat offline use.
 *
 * Preprocessing mirrors the Python training/serving pipeline exactly:
 *   PIL/torchvision: RGB -> Resize((224, 224)) bilinear, squash-to-square
 *                    -> ToTensor() (uint8 / 255 -> [0, 1])
 *                    -> Normalize(ImageNet mean/std) -> float32 NCHW
 * Here:              HTMLCanvas/ImageBitmap -> 224x224 drawImage with
 *                    smoothing (bilinear resampling, squash-to-square)
 *                    -> typed-array conversion (uint8 / 255)
 *                    -> per-channel (x - mean) / std -> Float32Array [1,3,224,224]
 * The model emits a single raw logit; sigmoid is applied here in JS.
 */

const ORT_VERSION = '1.27.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist`;
const ORT_SCRIPT_URL = `${ORT_BASE}/ort.min.js`;
const ORT_SCRIPT_INTEGRITY = 'sha384-uEaiZh7//Wy463wyKt2IRGCp/U7xzZL3k9Qd5Nm760KK9EkwN2nPN2DSEELB6WgC';

export const MODEL_URL = 'model/model.int8.onnx';
export const INPUT_SIZE = 224;

// ImageNet channel statistics, identical to the Python pipeline.
const IMAGENET_MEAN = new Float32Array([0.485, 0.456, 0.406]);
const IMAGENET_STD = new Float32Array([0.229, 0.224, 0.225]);

let ortLoadPromise = null;
let sessionPromise = null;

export class ModelMissingError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ModelMissingError';
  }
}

export class RuntimeUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = 'RuntimeUnavailableError';
  }
}

function loadRuntime() {
  if (globalThis.ort) return Promise.resolve(globalThis.ort);
  if (!ortLoadPromise) {
    ortLoadPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = ORT_SCRIPT_URL;
      script.integrity = ORT_SCRIPT_INTEGRITY;
      script.crossOrigin = 'anonymous';
      script.async = true;
      script.onload = () => {
        const ort = globalThis.ort;
        if (!ort) {
          ortLoadPromise = null;
          reject(new RuntimeUnavailableError('The inference runtime loaded but did not initialise.'));
          return;
        }
        ort.env.wasm.wasmPaths = `${ORT_BASE}/`;
        resolve(ort);
      };
      script.onerror = () => {
        ortLoadPromise = null;
        reject(new RuntimeUnavailableError(
          'The on-device inference runtime could not be downloaded. Connect to the internet once so it can be cached for offline use, then try again.'
        ));
      };
      document.head.appendChild(script);
    });
  }
  return ortLoadPromise;
}

export async function ensureSession(onProgress = () => {}) {
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    onProgress('runtime');
    const ort = await loadRuntime();
    onProgress('model');
    let probe = null;
    try {
      probe = await fetch(MODEL_URL, { method: 'HEAD' });
    } catch {
      // Offline first run: fall through and let session creation attempt it;
      // the service worker may still serve the model from cache.
    }
    if (probe && !probe.ok) {
      sessionPromise = null;
      throw new ModelMissingError(
        `The on-device model (${MODEL_URL}) is not installed on this device. Export it from the training pipeline and place it in the app/model/ folder.`
      );
    }
    try {
      return await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    } catch (cause) {
      sessionPromise = null;
      throw new Error(
        `The on-device model (${MODEL_URL}) could not be loaded (${cause && cause.message ? cause.message : cause}). ` +
        'The file may be corrupted or may never have been fetched while online.'
      );
    }
  })();
  return sessionPromise;
}

function get2dContext(width, height) {
  if (typeof OffscreenCanvas !== 'undefined') {
    const ctx = new OffscreenCanvas(width, height).getContext('2d');
    if (ctx) return ctx;
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas.getContext('2d', { willReadFrequently: true });
}

export function imageToTensor(source, ort) {
  const ctx = get2dContext(INPUT_SIZE, INPUT_SIZE);
  // Squash-to-square draw matches torchvision Resize((224, 224)) as used in
  // training (no aspect-ratio crop); canvas smoothing performs bilinear
  // resampling equivalent to PIL BILINEAR.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const planeSize = INPUT_SIZE * INPUT_SIZE;
  const buffer = new Float32Array(3 * planeSize);
  for (let i = 0; i < planeSize; i++) {
    const px = i * 4;
    buffer[i] = (data[px] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    buffer[planeSize + i] = (data[px + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    buffer[2 * planeSize + i] = (data[px + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
  return new ort.Tensor('float32', buffer, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export function logitToProbability(logits) {
  if (logits.length === 1) return sigmoid(logits[0]);
  if (logits.length === 2) {
    // Defensive fallback for a 2-class head: softmax of the positive class.
    const max = Math.max(logits[0], logits[1]);
    const e0 = Math.exp(logits[0] - max);
    const e1 = Math.exp(logits[1] - max);
    return e1 / (e0 + e1);
  }
  throw new Error(`Unexpected model output length ${logits.length}; expected a single raw logit.`);
}

export async function predictOnDevice(imageSource, onProgress = () => {}) {
  const session = await ensureSession(onProgress);
  const ort = await loadRuntime();
  const tensor = imageToTensor(imageSource, ort);
  const feeds = { [session.inputNames[0]]: tensor };
  const outputs = await session.run(feeds);
  const logits = outputs[session.outputNames[0]].data;
  return logitToProbability(logits);
}
