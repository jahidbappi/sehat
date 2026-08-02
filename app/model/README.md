# `app/model/` — on-device model drop zone

The Sehat web app expects the exported ONNX model at exactly this path:

```
app/model/model.int8.onnx
```

If the file is missing, the app shows a "Model file not found" error and
points here. Remote mode (`POST /predict` to a clinic server) works without it.

## Producing the model

From the repository root, run the export pipeline (see the training code under
`src/` for the exact flags):

```bash
python -m sehat.export --checkpoint <path/to/checkpoint.ckpt> --quantize int8 --out app/model/model.int8.onnx
```

## Expected model contract

The app's preprocessing mirrors the training/serving pipeline, so the export
must match it:

- **Input**: `float32` tensor, shape `[1, 3, 224, 224]` (NCHW), RGB,
  bilinear resize to 224×224 (squash-to-square, no center crop), scaled to
  `[0, 1]`, then ImageNet-normalized with
  mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Output**: a single raw logit. The app applies `sigmoid(logit)` in JS to
  obtain the screening probability.

## Version control

Model files are **not committed** to the repository. They are large binary
artifacts and may be governed by data-sharing or clinical-governance
agreements; distribute them out-of-band (clinic USB drives, internal object
storage) instead.
