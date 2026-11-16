# Thermal fault detection – Vercel app

Upload an RGB or thermal control-panel image → bounding boxes from your trained
**YOLOv8s** or **Faster R-CNN** models. Not a static site: inference runs in Python
serverless functions (`/api/*`) using ONNX Runtime (PyTorch is far too big for Vercel's 250 MB limit).

```
public/index.html            UI (upload, model/dataset toggle, confidence slider, overlay)
api/_common.py               preprocessing, ONNX inference, NMS, HTTP handler
api/detect_yolo.py           YOLOv8s  (RGB + Thermal)
api/detect_frcnn_rgb.py      Faster R-CNN (RGB)      } separate functions so each one
api/detect_frcnn_thermal.py  Faster R-CNN (Thermal)  } stays under the size limit
scripts/convert_models.py    .pt -> .onnx (run locally, once)
models/                      <- converted models go here
vercel.json  requirements.txt  .python-version
```

## 1. Convert the .pt files (once, on your machine)

```bash
pip install -r scripts/requirements-convert.txt
python scripts/convert_models.py --root /path/to/thermal-fault-detection-in-control-panels
# add --yolo-imgsz 512 etc. if YOLO was not trained at 640
```

It reads the four `best.pt` files from the folders you listed and writes
`models/yolo_rgb|yolo_thermal|frcnn_rgb|frcnn_thermal` `.onnx` + `.json` (class names).
Faster R-CNN is converted to fp16 to fit Vercel's size limit (it falls back to fp32 if fp16 fails).
Check the printed class names; edit `models/*.json` if needed.

## 2. Deploy

```bash
npm i -g vercel
vercel          # preview
vercel --prod   # production
```
Or push this folder to a GitHub repo and import it in Vercel (Framework preset: **Other**).
Note: GitHub rejects files >100 MB; if a model is bigger, use option below.

## Models too large to upload?
Upload the `models/*` files to any public/private-signed storage (e.g. Vercel Blob) and set the
env var `MODEL_BASE_URL=https://.../models` in the Vercel project. Functions download them to `/tmp`
on cold start (slower first request).

## Notes
- Browser downsizes images to ≤1280 px JPEG (Vercel request body limit ≈ 4.5 MB).
- First request per model is slow (cold start + model load); later ones are faster.
- `GET /api/detect_yolo` (etc.) reports whether the models loaded.
- `maxDuration` is 60 s (Hobby plan max); Faster R-CNN on CPU takes a few seconds.
