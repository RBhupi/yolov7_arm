# YOLOv7 for Vehicle Detection in ARM ENA camera images

Yolo model code is in yolo directory. The model weights are 75 MB size and are not uploaded here.  detection directory has all the scripts i have written. Yolo model taken from Seongha's github repo for waggle node.

A reproducible setup for running **YOLOv7** object detection.
works on hires images (1920×1080) using **MPS/CPU/CUDA** acceleration.

---

## 1. Environment Setup

### Option A — Reproduce from `environment.yml`
```bash
conda env create -f environment.yml
conda activate yolov7
```

This is exact copy of my environment which may have several packages not used in this code.
```
pip install -r requirements.txt
```
Check that you have installed pytorch correctly

```
python -c "import torch, cv2, pandas; print(torch.__version__, cv2.__version__, pandas.__version__)"
```

Running
```
conda activate yolov7
python code/python/detection/yolo-detect-mac_hires.py

```

