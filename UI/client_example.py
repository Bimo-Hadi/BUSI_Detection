# Example client:  python client_example.py my_scan.png
import os, sys, requests

URL = os.environ.get("BEEONIC_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ["BEEONIC_API_KEY"]} if os.environ.get("BEEONIC_API_KEY") else {}

with open(sys.argv[1], "rb") as f:
    r = requests.post(f"{URL}/v1/predict", headers=HEADERS, files={"file": f}, params={"include_overlay": "false"}, timeout=60)
r.raise_for_status()
print(r.json())
