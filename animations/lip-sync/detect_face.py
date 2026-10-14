"""Find face landmarks (lips, eyelids, chin) on each character cutout with MediaPipe Face Mesh.

Usage: python detect_face.py <dir with char1.png/char2.png and face_landmarker.task>
Writes landmarks.json (478 points per character, in cutout pixel coords).
Model: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""
import sys, json
import numpy as np, mediapipe as mp
from PIL import Image
from mediapipe.tasks.python import vision, BaseOptions

S = sys.argv[1]
det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=f"{S}/face_landmarker.task"), num_faces=1))
HEAD_BOXES = {1: (40, 0, 340, 330), 2: (80, 0, 398, 360)}  # head region in each cutout
SCALE = 3  # faces are small; upscale before detection

res = {}
for i, box in HEAD_BOXES.items():
    im = Image.open(f"{S}/char{i}.png").convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    crop = bg.crop(box).convert("RGB")
    crop = crop.resize((crop.width * SCALE, crop.height * SCALE), Image.LANCZOS)
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(crop)))
    if not r.face_landmarks:
        sys.exit(f"no face found in char{i}.png")
    res[i] = [(box[0] + p.x * crop.width / SCALE, box[1] + p.y * crop.height / SCALE) for p in r.face_landmarks[0]]
json.dump(res, open(f"{S}/landmarks.json", "w"))
print("wrote landmarks.json")
