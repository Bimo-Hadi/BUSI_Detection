# BUSI_Detection

Breast ultrasound triage AI for RS Beeonic (ASBE UI Mini Study Case Competition 2026).
Decision support only, not a diagnosis. A radiologist reads every image.

## Objective
Sort each breast ultrasound into **benign, malignant or normal**, so suspected malignant cases move to the top of the radiologist's worklist.

## Context
- Ultrasounds wait about 5 working days for a report, and 4 of those days are queue backlog (first come, first served).
- Confirmed cancers wait about 12 days from ultrasound to biopsy. The hospital's target is 7 days.
- The "urgent" checkbox is ticked on about 70% of requests, so it is ignored.
- The previous vendor's AI flagged about 4 in 10 benign lumps as malignant, and the pilot was stopped.

## Solution
- **Model:** EfficientNet-B0 (ImageNet weights) trained in two stages (head first, then fine-tuning), with a lesion-segmentation decoder (masks used for training only), an ensemble of 3 models, flip test-time augmentation, and a malignant threshold fixed on the validation set.
- **API:** FastAPI service that returns the predicted class, probabilities and a priority tier, and sorts a batch into a worklist.
- **Web interface:** drag and drop, sorted worklist, lesion overlay, CSV export.

**Results on the held-out BUSI test split** (117 images, 31 malignant):

| Metric | Day 2 baseline | Final model | SOP preferred |
|---|---|---|---|
| Accuracy | 0.598 | 0.846 | ≥ 0.70 |
| Malignant recall | 0.548 | 0.806 | ≥ 0.70 |
| Macro F1 | 0.546 | 0.838 | ≥ 0.65 |
| Malignant precision | 0.607 | 0.758 | ≥ 0.70 |

The test set is small, so the confidence intervals are wide (for example, malignant recall 95% CI 0.66 to 0.93). Performance must be confirmed on the hospital's own data in a shadow-mode pilot.

**Run it:** open the Colab notebook [add link], choose a GPU runtime, then Run all.

## Appendix
### 1. Model files (Keras)
`model_1.keras`, `model_2.keras`, `model_3.keras` (about 113 MB each). They are too large for GitHub, so download them here: [https://drive.google.com/drive/folders/1ygLUTuZm9frCDjNEc7QNIVg-iv9JdCus?usp=sharing].

### 2. API
```bash
cp -r beeonic_ui/static beeonic_ui/add_ui.py beeonic_ui/Dockerfile beeonic_ui/docker-compose.yml beeonic_api/
cd beeonic_api
python add_ui.py
pip install -r requirements.txt
uvicorn app:app --port 8000
```
Web interface: 'http://127.0.0.1:8000`. API docs: `/docs`. Link here : [https://drive.google.com/drive/folders/1yTAqfBnS85OkBPL-TTHRxQULb-sgMuqm?usp=sharing]

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness check |
| POST | `/v1/predict` | one image |
| POST | `/v1/predict/batch` | up to 20 images, sorted worklist |

### 3. Data and references
- Al-Dhabyani W, et al. Dataset of breast ultrasound images (BUSI). *Data in Brief*, 2020;28:104863. doi:10.1016/j.dib.2019.104863
- Oktay O, et al. Attention U-Net. arXiv:1804.03999, 2018.
- Tan M, Le QV. EfficientNet. ICML 2019.
- Selvaraju RR, et al. Grad-CAM. ICCV 2017.

### 4. Limitations
Single-source public dataset with an image-level split, no patient IDs, no external clinical validation, and no Kemenkes permit yet. For research and shadow-mode evaluation only.
