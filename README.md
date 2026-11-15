# BUSI_Detection
Breast ultrasound triage AI for RS Beeonic (ASBE UI Mini Study Case Competition 2026). Decision support only, not a diagnosis. A radiologist reads every image.

## Objective:
Sort each breast ultrasound into **benign, malignant or normal**, so suspected malignant cases move to the top of the radiologist's worklist.

## Context:
- Ultrasounds wait about 5 working days for a report, and 4 of those days are queue backlog (first come, first served).
- Confirmed cancers wait about 12 days from ultrasound to biopsy. The hospital's target is 7 days.
- The "urgent" checkbox is ticked on about 70% of requests, so it is ignored.
- The previous vendor's AI flagged about 4 in 10 benign lumps as malignant, and the pilot was stopped.

## Solution:
- **Model:** EfficientNet-B0 (ImageNet weights) trained in two stages (head first, then fine-tuning), with a lesion-segmentation decoder (masks used for training only), an ensemble of 3 models, flip test-time augmentation, and a malignant threshold fixed on the validation set.
- **API:** FastAPI service that returns the predicted class, probabilities and a priority tier, and sorts a batch into a worklist.
- **Web interface:** drag and drop, sorted worklist, lesion overlay, CSV export.

### Appendix
1. File models keras (1, 2, 3)
2. API
