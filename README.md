# Autonomous Driving Vision System

**Modular, GPU-accelerated computer-vision pipeline for autonomous driving** — combines semantic segmentation, YOLOv8 object detection, lane detection, traffic light/sign recognition and monocular depth estimation. Includes an interactive Gradio UI supporting both **image** and **video** processing (outputs processed MP4s).

---

## Features

* Semantic segmentation (DeepLabv3 + ResNet50) — colorized overlays
* Real-time object detection (YOLOv8n) — class, confidence, bbox and priority (VRUs, traffic signs, vehicles)
* Lane detection — Canny + Hough + polynomial fitting, lane center & deviation estimate
* Traffic light & sign recognition — color/contour based light detection + recommended action
* Monocular depth estimation (MiDaS_small) — fast, colorized depth maps
* Full **image + video** pipeline with Gradio UI and per-module toggles
* GPU-aware: automatically uses CUDA if available

---

## Quick start

### 1. Clone

```bash
git clone <your-repo-url>
cd autonomous-driving-vision-system
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

Save the example below to `requirements.txt` and install:

```txt
torch
torchvision
transformers
gradio
opencv-python
albumentations
segmentation-models-pytorch
timm
scikit-learn
pillow
numpy
matplotlib
ultralytics
kaggle
```

Then:

```bash
pip install -r requirements.txt
```

> **FFmpeg** is required for robust video writing.
>
> * Ubuntu / Colab: `sudo apt update && sudo apt install -y ffmpeg`
> * Windows / macOS: install ffmpeg and add it to `PATH`.

### 4. Run

```bash
python app.py
```

The app launches a Gradio interface. If running in Colab, allow installation cells to run; in notebooks, run the `create_gradio_interface()` cell.

---

## Usage

* **Image Analysis**: Upload an image → select modules (Segmentation, Detection, Lane, Traffic, Depth) → **Analyze Image**. You’ll get per-module outputs, combined overlay and a short detection summary.
* **Video Processing**: Upload an MP4 → select modules → **Process Video**. The backend processes frames and outputs a processed MP4.

**Tip:** For fast iteration, disable heavier modules (segmentation / depth) or process every 2nd/3rd frame.

---

## How it works (high level)

1. Read frame with OpenCV.
2. Run independent modules (segmentation, detection, lane, traffic lights, depth).
3. Blend module visualizations into a single combined image per frame.
4. For video: write processed frames to a temporary MP4 and return it.

Optimizations in code:

* ResNet50 (lighter backbone) for segmentation.
* YOLOv8 **Nano** model for fast detection.
* MiDaS_small for lightweight depth.
* Resize large images (>1024 px width) before segmentation to save compute.

---

## Suggested repo layout

```
.
├── app.py                 # main Gradio application
├── README.md
├── requirements.txt
├── models/                # optional: store yolov8n.pt or other weights
├── assets/                # sample images / videos
├── notebooks/             # Colab / Jupyter demos
└── LICENSE
```

---

## Troubleshooting

* `ImportError: ultralytics` → `pip install ultralytics`. Ultralytics may auto-download `yolov8n.pt` on first run.
* Slow / OOM on CPU → use CUDA GPU. Code prints `Running on cuda` or `cpu`.
* Video not producing output → ensure `ffmpeg` is installed and available in `PATH`.

---

## Performance tips

* Use a GPU (T4 / RTX series) for near real-time performance.
* Skip frames for long videos to reduce processing time.
* Replace `yolov8n.pt` with different YOLO variants depending on accuracy/speed tradeoffs.

---

## Contributing

PRs welcome. Ideas:

* Add multi-frame tracking (e.g., DeepSORT) for persistent IDs.
* Replace rule-based traffic-light detection with a trained classifier.
* Add ROS / vehicle control integration.
* Add unit tests and CI.

---

## License

Suggested: **MIT** .

---
