# Beeonic Breast Ultrasound Triage API

Decision-support service for RS Beeonic. It returns the predicted class (benign / malignant / normal), class
probabilities, a suggested worklist priority and the predicted lesion area. **A radiologist reads every image.**
It runs fully on-premise (a small server inside the hospital); images are processed in memory and not stored.

## Run locally
```bash
pip install -r requirements.txt
export BEEONIC_API_KEY="choose-a-long-random-key"     # omit only for local development
uvicorn app:app --host 0.0.0.0 --port 8000
```
Interactive docs: `http://localhost:8000/docs`

## Docker
```bash
docker build -t beeonic-api .
docker run -e BEEONIC_API_KEY="choose-a-long-random-key" -p 8000:8000 beeonic-api
```

## Endpoints
| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | liveness check |
| GET | `/v1/info` | key | model version, threshold, triage table, reported test metrics |
| POST | `/v1/predict` | key | one image (multipart field `file`); add `?include_overlay=true` for a base64 lesion overlay |
| POST | `/v1/predict/batch` | key | up to 20 images, returns a sorted worklist (malignant first) |

```bash
curl -X POST "http://localhost:8000/v1/predict" -H "X-API-Key: $BEEONIC_API_KEY" -F "file=@scan.png"
```
Example response:
```json
{"predicted_class": "malignant", "probabilities": {"benign": 0.08, "malignant": 0.9, "normal": 0.02},
 "malignant_probability": 0.9, "decision_threshold_malignant": 0.5,
 "triage": {"priority": "URGENT", "suggested_read_within": "1 day"},
 "lesion_area_pct": 6.4, "needs_extra_attention": false, "model_version": "2.0", "disclaimer": "..."}
```

## Notes
* Triage times are proposed pilot targets, not hospital policy.
* Use the same TensorFlow/Keras major version as the training notebook (see `models/config.json`).
* Not a certified medical device (no Kemenkes distribution permit yet). Use for shadow-mode evaluation only.
* For production: put it behind HTTPS (reverse proxy), rotate the API key, and log only request metadata, never images.
