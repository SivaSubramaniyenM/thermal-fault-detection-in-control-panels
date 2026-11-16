"""Convert the four .pt checkpoints to ONNX for Vercel (no PyTorch needed at runtime).

Run once, locally, from anywhere:

    pip install -r scripts/requirements-convert.txt
    python scripts/convert_models.py --root /path/to/thermal-fault-detection-in-control-panels

Outputs go to ./models/ (yolo_rgb, yolo_thermal, frcnn_rgb, frcnn_thermal: .onnx + .json).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

SOURCES = {
    "yolo_rgb": ("yolo", "RGB_Dataset/models/yolov8s_cpu_20261116/weights"),
    "yolo_thermal": ("yolo", "Thermal_Dataset/models/yolov8s_cpu_25ep_20261116/weights"),
    "frcnn_rgb": ("frcnn", "RGB_Dataset/models/fasterrcnn_cpu_1ep_20261116"),
    "frcnn_thermal": ("frcnn", "Thermal_Dataset/models/fasterrcnn_cpu_1ep_20261116"),
}


def find_pt(folder: Path) -> Path:
    for name in ("best.pt", "last.pt"):
        if (folder / name).exists():
            return folder / name
    found = sorted(folder.glob("*.pt"))
    if not found:
        raise FileNotFoundError(f"No .pt file in {folder}")
    return found[0]


def convert_yolo(pt: Path, out_dir: Path, key: str, imgsz: int):
    from ultralytics import YOLO

    model = YOLO(str(pt))
    exported = Path(model.export(format="onnx", imgsz=imgsz, opset=12, simplify=False, dynamic=False))
    target = out_dir / f"{key}.onnx"
    shutil.move(str(exported), target)
    names = [model.names[i] for i in sorted(model.names)]
    (out_dir / f"{key}.json").write_text(json.dumps({"kind": "yolo", "names": names, "imgsz": imgsz}, indent=2))
    return target, names


def convert_frcnn(pt: Path, out_dir: Path, key: str, fp16: bool):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    ckpt = torch.load(pt, map_location="cpu", weights_only=True)
    if ckpt.get("architecture") != "fasterrcnn":
        raise ValueError(f"{pt} is not a fasterrcnn checkpoint: {ckpt.get('architecture')}")
    categories = ckpt["categories"]
    size = int(ckpt["image_size"])
    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, min_size=size, max_size=size,
        box_score_thresh=0.05, box_detections_per_img=100,
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, len(categories) + 1
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image):
            det = self.m([image])[0]
            return det["boxes"], det["labels"], det["scores"]

    dummy = torch.rand(3, size, size)
    target = out_dir / f"{key}.onnx"
    kwargs = dict(
        opset_version=17, input_names=["image"], output_names=["boxes", "labels", "scores"],
        dynamic_axes={"boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}},
    )
    with torch.no_grad():
        try:
            torch.onnx.export(Wrapper(model), (dummy,), str(target), dynamo=False, **kwargs)
        except TypeError:  # older torch without the dynamo argument
            torch.onnx.export(Wrapper(model), (dummy,), str(target), **kwargs)

    (out_dir / f"{key}.json").write_text(
        json.dumps({"kind": "frcnn", "names": [c["name"] for c in categories], "imgsz": size}, indent=2)
    )
    if fp16:
        try_fp16(target, size)
    return target, [c["name"] for c in categories]


def try_fp16(path: Path, size: int):
    """Halve the file size (Vercel's 250 MB function limit). Falls back to fp32 on any problem."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16

    before = path.stat().st_size / 1e6
    try:
        block = list(getattr(float16, "DEFAULT_OP_BLOCK_LIST", [])) or [
            "NonMaxSuppression", "TopK", "RoiAlign", "Resize", "Range", "CumSum", "Min", "Max", "Upsample"]
        converted = float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True, op_block_list=block)
        tmp = path.with_suffix(".fp16.onnx")
        onnx.save(converted, str(tmp))
        sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        sess.run(None, {"image": np.random.rand(3, size, size).astype("float32")})
        tmp.replace(path)
        print(f"   fp16: {before:.0f} MB -> {path.stat().st_size / 1e6:.0f} MB")
    except Exception as exc:  # noqa: BLE001
        print(f"   fp16 conversion failed ({exc}); keeping fp32 ({before:.0f} MB)")
        if path.with_suffix(".fp16.onnx").exists():
            path.with_suffix(".fp16.onnx").unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Folder containing RGB_Dataset/ and Thermal_Dataset/")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    ap.add_argument("--yolo-imgsz", type=int, default=640, help="Image size YOLO was trained with")
    ap.add_argument("--no-fp16", action="store_true", help="Keep Faster R-CNN in fp32")
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), help="Convert a subset")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for key, (kind, rel) in SOURCES.items():
        if args.only and key not in args.only:
            continue
        pt = find_pt(args.root / rel)
        print(f"== {key}: {pt}")
        if kind == "yolo":
            target, names = convert_yolo(pt, args.out, key, args.yolo_imgsz)
        else:
            target, names = convert_frcnn(pt, args.out, key, not args.no_fp16)
        print(f"   -> {target.name} ({target.stat().st_size / 1e6:.0f} MB), classes: {names}")
    print("\nCheck that YOLO and Faster R-CNN class names match for each dataset; edit models/*.json if not.")


if __name__ == "__main__":
    sys.exit(main())
