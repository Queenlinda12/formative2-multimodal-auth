"""
Task 2 - Image Data Collection and Processing
=============================================
Author: Aime Igirimpuhwe (Image lead), GitHub @igaimerca

Checks every member's face photos, augments them and saves one feature row per
image, original and augmented:

    images/augmented/<member>_<expression>_<augmentation>.jpg
    features/image_features.csv   member, expression, augmentation, source_image, emb_000 ... emb_127

Run from the repository root:
    python scripts/image_pipeline.py

Design decisions (explained in the notebook and report):
  * Features are 128-number face embeddings from OpenCV's pretrained SFace model,
    computed on the face that the YuNet detector finds and aligns. With only three
    photos per person, a model pretrained on millions of faces tells people apart
    far better than colour histograms, and is not thrown off by lighting.
  * Every photo gets 5 augmentations (rotation, flip, grayscale, brightness, blur).
    Each mimics a way a real login photo can differ: a tilted head, a mirrored
    selfie camera, poor colour, different light, a shaky hand.
  * `source_image` records which original photo a row came from, so the face model
    can keep a photo and its augmented copies on the same side of a train/test split.
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "images" / "raw"
AUGMENTED = ROOT / "images" / "augmented"
FEATURES_FILE = ROOT / "features" / "image_features.csv"
PRETRAINED = ROOT / "models" / "pretrained"
DETECTOR_FILE = PRETRAINED / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = PRETRAINED / "face_recognition_sface_2021dec.onnx"

EXPRESSIONS = ["neutral", "smiling", "surprised"]
NAME_PATTERN = re.compile(r"^([a-z]+)_(neutral|smiling|surprised)\.(?i:jpe?g|png)$")
HEIC_SUFFIXES = {".heic", ".heif"}  # iPhone "High Efficiency" photos, converted to JPG first
UNAUTHORIZED_PREFIX = "unauthorized"
EMBEDDING_COLUMNS = [f"emb_{i:03d}" for i in range(128)]
MIN_FACE_SCORE = 0.8
MAX_SIDE = 1280  # phone photos are downscaled first; faces stay large enough to detect

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # hide backend warnings, keep real errors


# ------------------------------------------------------------- 1. reading
def read_image(path):
    """Load a photo as a BGR array, downscaled so its longest side is at most MAX_SIDE."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read {path}; is it a JPG or PNG image?")
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


class FaceEncoder:
    """Finds the main face in a photo and turns it into a 128-number embedding."""

    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(str(DETECTOR_FILE), "", (320, 320), MIN_FACE_SCORE)
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_FILE), "")

    def detect(self, img):
        """The most confident face as YuNet's 15-number row (box, landmarks, score), or None."""
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        return faces[np.argmax(faces[:, -1])]

    def embed(self, img):
        """Unit-length embedding of the main face, or None when no face is found."""
        face = self.detect(img)
        if face is None:
            return None
        aligned = self.recognizer.alignCrop(img, face)
        vector = self.recognizer.feature(aligned).flatten().astype(np.float64)
        return vector / np.linalg.norm(vector)


# --------------------------------------------------- 2. collection check
def convert_heic(raw_dir=RAW):
    """Saves a JPG copy of every iPhone .heic photo that doesn't have one yet; returns the new names."""
    heic = [p for p in sorted(raw_dir.iterdir()) if p.suffix.lower() in HEIC_SUFFIXES]
    if not heic:
        return []
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener

    register_heif_opener()
    converted = []
    for path in heic:
        target = path.with_suffix(".jpg")
        if target.exists():
            continue
        with Image.open(path) as photo:
            # exif_transpose applies the phone's rotation tag so the face is upright.
            ImageOps.exif_transpose(photo).convert("RGB").save(target, "JPEG", quality=95)
        converted.append(target.name)
    return converted


def collection_report(encoder=None, raw_dir=RAW):
    """One row per team photo: is the name valid, and is exactly one clear face found?"""
    encoder = encoder or FaceEncoder()
    rows = []
    for path in sorted(raw_dir.iterdir()):
        if path.name.startswith((".", UNAUTHORIZED_PREFIX)) or path.suffix.lower() in HEIC_SUFFIXES:
            continue
        match = NAME_PATTERN.match(path.name)
        try:
            face = encoder.detect(read_image(path))
        except ValueError:  # a file OpenCV can't open, so no face can be checked
            face = None
        rows.append({
            "file": path.name,
            "member": match[1] if match else None,
            "expression": match[2] if match else None,
            "valid_name": bool(match),
            "face_found": face is not None,
            "face_score": round(float(face[-1]), 3) if face is not None else None,
        })
    return pd.DataFrame(rows)


def missing_expressions(report):
    """{member: [expressions still missing]} for every member with a gap."""
    named = report[report["valid_name"]]
    have = named.groupby("member")["expression"].apply(set)
    return {m: [e for e in EXPRESSIONS if e not in got] for m, got in have.items() if set(EXPRESSIONS) - got}


# ---------------------------------------------------------- 3. augment
def rotate(img, angle):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def grayscale(img):
    # Back to 3 channels: the face models expect a colour-shaped image.
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


AUGMENTATIONS = {
    "rotated": lambda img: rotate(img, 15),
    "flipped": lambda img: cv2.flip(img, 1),
    "grayscale": grayscale,
    "brightened": lambda img: cv2.convertScaleAbs(img, alpha=1.0, beta=50),
    "blurred": lambda img: cv2.GaussianBlur(img, (9, 9), 0),
}


def augment(img):
    """The original plus every augmentation, as {name: image}."""
    return {"original": img, **{name: fn(img) for name, fn in AUGMENTATIONS.items()}}


# ------------------------------------------------- 4. extract and save
def build_features(encoder=None, raw_dir=RAW, augmented_dir=AUGMENTED):
    """Augments every team photo, saves the copies and returns (features, rows where no face was found)."""
    encoder = encoder or FaceEncoder()
    augmented_dir.mkdir(parents=True, exist_ok=True)
    rows, no_face = [], []
    for path in sorted(raw_dir.iterdir()):
        match = NAME_PATTERN.match(path.name)
        if not match:
            continue
        member, expression = match[1], match[2]
        for name, img in augment(read_image(path)).items():
            if name != "original":
                cv2.imwrite(str(augmented_dir / f"{member}_{expression}_{name}.jpg"), img)
            embedding = encoder.embed(img)
            if embedding is None:
                no_face.append(f"{path.name} ({name})")
                continue
            rows.append({"member": member, "expression": expression, "augmentation": name,
                         "source_image": path.name, **dict(zip(EMBEDDING_COLUMNS, embedding))})
    return pd.DataFrame(rows), no_face


def main():
    converted = convert_heic()
    if converted:
        print(f"Converted {len(converted)} iPhone photos to JPG: {', '.join(converted)}\n")
    encoder = FaceEncoder()
    report = collection_report(encoder)
    print(report.to_string(index=False), "\n")
    for problem, files in [("Bad file names", report.loc[~report["valid_name"], "file"]),
                           ("No face found", report.loc[~report["face_found"], "file"])]:
        if len(files):
            print(f"{problem}: {', '.join(files)}")
    for member, missing in missing_expressions(report).items():
        print(f"{member} is missing: {', '.join(missing)}")

    features, no_face = build_features(encoder)
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_FILE, index=False)
    print(f"\nSaved {len(features)} rows ({features['source_image'].nunique()} photos x "
          f"{len(AUGMENTATIONS) + 1} versions) to {FEATURES_FILE.relative_to(ROOT)}")
    if no_face:
        print(f"Skipped {len(no_face)} images with no detectable face: {', '.join(no_face)}")


if __name__ == "__main__":
    main()
