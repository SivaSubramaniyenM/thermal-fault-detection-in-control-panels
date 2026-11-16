"""Shared ONNX inference code for the Vercel serverless functions.

Files whose names start with "_" inside /api are not exposed as routes.
"""
import io
import json
import os
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models")
MAX_BODY = 4_400_000  # Vercel request bodies are capped at ~4.5 MB
IMAGENET_MEAN = (124, 116, 104)

_cache = {}


def _resolve(filename):
    """Local models/ file, or download from MODEL_BASE_URL into /tmp."""
    local = os.path.join(MODEL_DIR, filename)
    if os.path.exists(local):
        return local
    base = os.environ.get("MODEL_BASE_URL")
    if not base:
        raise FileNotFoundError(
            f"models/{filename} is missing. Run scripts/convert_models.py and "
            "redeploy, or set MODEL_BASE_URL."
        )
    cache_dir = os.path.join("/tmp", "models")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, filename)
    if not os.path.exists(dest):
        part = dest + ".part"
        urllib.request.urlretrieve(base.rstrip("/") + "/" + filename, part)
        os.replace(part, dest)
    return dest


def load_model(key):
    if key in _cache:
        return _cache[key]
    import onnxruntime as ort

    with open(_resolve(key + ".json"), encoding="utf-8") as f:
        meta = json.load(f)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(
        _resolve(key + ".onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    shape = session.get_inputs()[0].shape
    size = shape[-1] if isinstance(shape[-1], int) else meta.get("imgsz", 640)
    _cache[key] = (session, meta, int(size))
    return _cache[key]


def read_image(data):
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def letterbox(img, size, fill, center):
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    px = (size - nw) // 2 if center else 0
    py = (size - nh) // 2 if center else 0
    canvas.paste(resized, (px, py))
    return canvas, scale, px, py


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def _name(names, idx):
    return names[idx] if 0 <= idx < len(names) else str(idx)


def run_yolo(key, img, conf):
    session, meta, size = load_model(key)
    w, h = img.size
    canvas, scale, px, py = letterbox(img, size, (114, 114, 114), True)
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    out = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    if out.shape[0] < out.shape[1]:  # (4+nc, N) -> (N, 4+nc)
        out = out.T
    cls_scores = out[:, 4:]
    class_ids = cls_scores.argmax(1)
    scores = cls_scores.max(1)
    mask = scores >= conf
    out, class_ids, scores = out[mask], class_ids[mask], scores[mask]
    if not len(scores):
        return []
    cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    # class-aware NMS via coordinate offset
    keep = nms(boxes + class_ids[:, None] * 10000.0, scores, 0.45)[:100]
    return [
        dict(
            class_id=int(class_ids[i]),
            label=_name(meta["names"], int(class_ids[i])),
            score=float(scores[i]),
            box=[float(v) for v in boxes[i]],
        )
        for i in keep
    ]


def run_frcnn(key, img, conf):
    session, meta, size = load_model(key)
    w, h = img.size
    canvas, scale, _, _ = letterbox(img, size, IMAGENET_MEAN, False)
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    boxes, labels, scores = session.run(None, {session.get_inputs()[0].name: x})
    results = []
    for b, l, s in zip(boxes, labels, scores):
        if s < conf:
            continue
        b = b / scale
        b[[0, 2]] = b[[0, 2]].clip(0, w)
        b[[1, 3]] = b[[1, 3]].clip(0, h)
        idx = int(l) - 1  # label 0 is background
        results.append(
            dict(
                class_id=idx,
                label=_name(meta["names"], idx),
                score=float(s),
                box=[float(v) for v in b],
            )
        )
    results.sort(key=lambda r: -r["score"])
    return results[:100]


def detect(family, dataset, data, conf):
    key = f"{family}_{dataset}"
    img = read_image(data)
    t0 = time.perf_counter()
    fn = run_yolo if family == "yolo" else run_frcnn
    dets = fn(key, img, conf)
    return {
        "model": key,
        "width": img.size[0],
        "height": img.size[1],
        "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        "detections": dets,
    }


class DetectHandler(BaseHTTPRequestHandler):
    FAMILY = "yolo"
    DATASETS = ("rgb", "thermal")

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        status = {}
        for ds in self.DATASETS:
            try:
                load_model(f"{self.FAMILY}_{ds}")
                status[ds] = "ready"
            except Exception as exc:  # noqa: BLE001
                status[ds] = f"error: {exc}"
        self._send(200, {"family": self.FAMILY, "models": status})

    def do_POST(self):
        try:
            query = parse_qs(urlparse(self.path).query)
            dataset = (query.get("dataset", [self.DATASETS[0]])[0]).lower()
            if dataset not in self.DATASETS:
                return self._send(400, {"error": f"dataset must be one of {self.DATASETS}"})
            conf = float(query.get("conf", ["0.05"])[0])
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                return self._send(400, {"error": "Empty body: send the image bytes."})
            if length > MAX_BODY:
                return self._send(413, {"error": "Image too large (max ~4.4 MB)."})
            data = self.rfile.read(length)
            self._send(200, detect(self.FAMILY, dataset, data, conf))
        except FileNotFoundError as exc:
            self._send(503, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
