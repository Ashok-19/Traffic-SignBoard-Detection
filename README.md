# Traffic Signboard Detection (YOLO26, Laptop-Friendly)

## Todo

- [x] First Iteration model with basic traffic signboard detection

- [ ] Add labeled data and update the model to detect more specific sign types (e.g., stop sign, zebra crossing, etc.)

- [ ] Build an audio feedback system that can alert the user when a signboard is detected, and provide information about the sign

- [ ] Build a rule based system to detect high priority sign types to filter out noise and only alert the user for important signs

- [ ] Optimize the model and inference pipeline for real-time performance.

- [ ] Test the system in real-world scenarios and gather feedback for further improvements.

- [ ] Port entire pipeline jetson nano 2gb for on-device inference.

- [ ] Re-optimize again for qualcomm 6490 chip
 
This project curates a **single-class** dataset (`traffic_sign_board`) and trains/tests a **YOLO26 nano** detector locally.

## Current curated dataset

Path:

`/home/nnmax/Desktop/forge-alpha/signboard-yolo/data/curated/signboard_yolo26_lite`

Split stats:

| Split | Images | Boxes |
|---|---:|---:|
| Train | 2480 | 3000 |
| Val | 336 | 393 |
| Test | 312 | 371 |
| **Total** | **3128** | **3764** |

Source mix in curated set:

| Source | Images | Boxes |
|---|---:|---:|
| github_abhayvashokan | 1796 | 2342 |
| roboflow_traffic_sign_yolo26 | 500 | 586 |
| roboflow_indian_traffic_sign_yolo26 | 832 | 836 |

---

## In-depth analysis of the two YOLO26 Roboflow zips

### 1) `traffic sign.yolo26.zip`
- Raw images/labels: **2067 / 2067**
- Classes: **2** (`gap in median`, `no parking`)
- Labeled boxes: **1037** (many empty label files)
- Box area (normalized): Q50 ≈ **0.019** (small/medium signs)
- Median resolution: **253×237**

### 2) `Indian Traffic Sign.yolo26.zip`
- Raw images/labels: **6750 / 6750**
- Classes: **85**
- Labeled boxes: **6769**
- Box area (normalized): Q50 ≈ **0.605** (mostly sign-focused crops)
- Median resolution: **178×220**

### Curation choices for local training
- Unified to one class (`traffic_sign_board`)
- Dropped tiny boxes: `--min-box-area 0.0005`
- Dropped very large crop-like boxes: `--max-box-area 0.90`
- Exact perceptual dedup applied (`--dedup-threshold 0`)
- Source caps from `configs/sources.yaml` to keep dataset laptop-sized:
  - `github_abhay: 1800`
  - `roboflow_traffic_sign_yolo26: 500`
  - `roboflow_indian_traffic_sign_yolo26: 900`

---

## Files delivered

- `configs/sources.yaml` (source registry + per-source caps)
- `scripts/curate_signboard_dataset.py` (curation pipeline)
- `scripts/train_local.py` (YOLO26 local training)
- `scripts/test_local.py` (test/eval + inference)
- `scripts/fetch_sources.py` (optional downloader)
- `requirements.txt`

---

## 1) Setup

```bash
cd /home/nnmax/Desktop/forge-alpha/signboard-yolo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Optional download step

If you want to fetch additional configured sources:

```bash
python scripts/fetch_sources.py --config configs/sources.yaml
```

For Roboflow API downloads:

```bash
export ROBOFLOW_API_KEY="your_key"
```

---

## 3) Rebuild the curated YOLO26 dataset

```bash
python scripts/curate_signboard_dataset.py \
  --sources-config configs/sources.yaml \
  --output-dir data/curated/signboard_yolo26_lite \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --min-box-area 0.0005 \
  --max-box-area 0.90 \
  --dedup-threshold 0
```

Reports generated:
- `data/curated/signboard_yolo26_lite/reports/curation_report.json`
- `data/curated/signboard_yolo26_lite/reports/stats.json`
- `data/curated/signboard_yolo26_lite/reports/manifest.csv`
- `data/curated/signboard_yolo26_lite/reports/source_analysis.json`

---

## 4) Train locally (YOLO26)

```bash
python scripts/train_local.py \
  --data data/curated/signboard_yolo26_lite/dataset.yaml \
  --model yolo26n.pt \
  --epochs 80 \
  --imgsz 768 \
  --batch 8 \
  --device 0
```

CPU fallback:

```bash
python scripts/train_local.py --device cpu --batch 4
```

---

## 5) Test locally

```bash
python scripts/test_local.py \
  --weights runs/signboard/yolo26n_signboard_lite/weights/best.pt \
  --data data/curated/signboard_yolo26_lite/dataset.yaml \
  --imgsz 768 \
  --batch 8 \
  --device 0
```

Optional inference:

```bash
python scripts/test_local.py \
  --weights runs/signboard/yolo26n_signboard_lite/weights/best.pt \
  --data data/curated/signboard_yolo26_lite/dataset.yaml \
  --predict-source /path/to/images_or_video
```
