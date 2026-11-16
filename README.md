# RGB and Thermal Fault Detection

An object-detection pipeline for comparing YOLOv8s, Faster R-CNN and SSD300 on separate RGB and thermal image datasets. It validates COCO annotations, prepares augmented training data, trains detectors and evaluates them on held-out test splits.

## Repository Contents

- `run_detection_pipeline.py`: staged validation, preparation, training and evaluation.
- `validate_coco.py`, `augment_coco.py`, `coco_to_yolo.py`: data validation, augmentation and conversion.
- `train_torchvision.py`, `evaluate_models.py`: torchvision training and common COCO evaluation.
- `merge_coco.py`, `split_coco_modalities.py`, `yolo_to_coco.py`: dataset utilities.
- `test_detection_pipeline.py`: synthetic regression and training smoke tests.
- `results/`: previously recorded two-model CPU comparisons, plots and per-class diagnostics.
- `DETECTION_WORKFLOW.md`: detailed model, augmentation, metric and workflow notes.

## Data And Checkpoints

The original local package includes `RGB_Dataset/` and `Thermal_Dataset/`, each with the COCO split annotations and their referenced images. It also includes the available YOLOv8s and Faster R-CNN best checkpoints and category mappings under those directories. These directories and all `.pt` files are ignored by Git: they **will not be included in a GitHub push or clone**. No SSD checkpoint is included; train one before using the three-model runner evaluation.

To run after cloning, obtain the data and weights separately from an authorized source and restore the following layout:

```text
RGB_Dataset/                         # same layout for Thermal_Dataset/
  annotations/instances_{train,val,test}.json
  images/{train,val,test}/           # files referenced by each annotation JSON
  mapping/category_mapping.json      # required with existing YOLO checkpoint
  models/                            # optional existing best checkpoints
```

The local RGB splits contain 404 train, 86 validation and 88 test images; the thermal splits contain 347, 74 and 76 respectively. The provided Roboflow export notes do not establish redistribution rights for the datasets. Verify rights and any privacy obligations before publishing images, annotations, weights or results. No project code license has been specified; choose an appropriate license before inviting reuse. Ultralytics and other dependencies have their own licensing terms.

## Setup

These commands use PowerShell on Windows and should be run from the project directory. A Python 3.14 environment was used for the smoke tests; install compatible PyTorch and torchvision builds for your platform. For CUDA, follow the official PyTorch installation instructions for your GPU before installing the remaining requirements. Training on CPU can be slow, and pretrained weights may be downloaded on first use.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements_detection.txt
```

On macOS or Linux, activate the virtual environment using the platform's usual command and use its `python` executable in place of `.\.venv\Scripts\python.exe`.

## Run The Pipeline

The default commands process both modalities independently. Add `--datasets RGB_Dataset` or `--datasets Thermal_Dataset` to select one. The validation command exits with a nonzero status if it finds invalid annotations; the supplied RGB data has bounding boxes that need clipping.

```powershell
# Inspect the original annotations and images.
.\.venv\Scripts\python.exe run_detection_pipeline.py --stage validate

# Write repaired annotation copies, augment training images and export YOLO data.
.\.venv\Scripts\python.exe run_detection_pipeline.py --stage prepare --clip-boxes

# Train YOLOv8s, Faster R-CNN and SSD300, then evaluate their checkpoints.
.\.venv\Scripts\python.exe run_detection_pipeline.py --stage train --device cpu
.\.venv\Scripts\python.exe run_detection_pipeline.py --stage evaluate --device cpu
```

Preparation writes generated images into the dataset's training image folder and output files under `detection_runs/`. It refuses to overwrite prepared artifacts; use a different `--output` directory for a fresh run and account for existing augmentation images. Source annotation JSON files are not changed. See `DETECTION_WORKFLOW.md` for individual script commands and important augmentation and evaluation details.

### Evaluate The Existing Two-Model Thermal Run

The locally supplied thermal checkpoints can be evaluated without SSD. Choose a new output directory on each run:

```powershell
.\.venv\Scripts\python.exe evaluate_models.py --annotations Thermal_Dataset/annotations/instances_test.json --images Thermal_Dataset/images/test --yolo Thermal_Dataset/models/yolov8s_cpu_25ep_20261116/weights/best.pt --yolo-mapping Thermal_Dataset/mapping/category_mapping.json --fasterrcnn Thermal_Dataset/models/fasterrcnn_cpu_1ep_20261116/best.pt --output evaluation_runs/thermal_example --device cpu
```

For RGB, validate and, if necessary, clip the test annotations into a new JSON before evaluating. `evaluate_models.py --help` lists the available options, including the optional `--ssd` checkpoint.

## Recorded Results

The committed `results/` directory contains CPU results for YOLOv8s and Faster R-CNN, not the three-model workflow. These runs used different training budgets, so they are not a controlled comparison of model architectures. Metrics are fractions, and timings depend on hardware and software versions. Review split independence, data provenance and the methodological caveats in `DETECTION_WORKFLOW.md` before interpreting the results.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest test_detection_pipeline -v
```

The tests use synthetic images and brief smoke training, not pretrained downloads or accuracy benchmarks.
