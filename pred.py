"""
Run Roboflow inference on all test images and save predictions as JSON.
Install first: pip install roboflow

Usage:
    python get_predictions.py
"""

import json
import os
from roboflow import Roboflow
                

# --- Fill these in from your Roboflow Deploy tab ---
API_KEY = "YsmThSUUMwB8RbUKMvU8"
PROJECT_NAME = "my-first-project-fwkpc2"   # e.g. "marine-plastic-detection"
VERSION_NUMBER = 2                    # e.g. 1
# ---------------------------------------------------

TEST_IMAGES_DIR = "images/Test"   # adjust to your path
OUTPUT_FILE = "predictions.json"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("julias-workspace-3ywy3").project("my-first-project-fwkpc")
model = project.version(VERSION_NUMBER).model

all_predictions = []

images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"Running inference on {len(images)} test images...")

for i, fname in enumerate(images):
    img_path = os.path.join(TEST_IMAGES_DIR, fname)
    try:
        result = model.predict(img_path, confidence=40).json()
        all_predictions.append({
            "image": fname,
            "predictions": result["predictions"]
        })
        print(f"[{i+1}/{len(images)}] {fname} — {len(result['predictions'])} detections")
    except Exception as e:
        print(f"[{i+1}/{len(images)}] {fname} — ERROR: {e}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_predictions, f, indent=2)

print(f"\nDone. Saved to {OUTPUT_FILE}")