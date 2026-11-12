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

import hashlib
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
NAME_PATTERN = re.compile(r"([a-z]+)_(neutral|smiling|surprised)\.(?i:jpe?g|png)")  # used with fullmatch
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
HEIC_SUFFIXES = {".heic", ".heif"}  # iPhone "High Efficiency" photos, converted to JPG first
UNAUTHORIZED_PREFIX = "unauthorized"
EMBEDDING_COLUMNS = [f"emb_{i:03d}" for i in range(128)]
FEATURE_COLUMNS = ["member", "expression", "augmentation", "source_image", *EMBEDDING_COLUMNS]
REPORT_COLUMNS = ["file", "member", "expression", "valid_name", "duplicate", "faces", "face_score", "ok"]
MIN_FACE_SCORE = 0.8
MAX_SIDE = 1280  # phone photos are downscaled first; faces stay large enough to detect
# YuNet can misplace a face that fills most of the frame (a close-up selfie got its box half a face too
# low, with a lower score), so faces are also looked for at half size; the most confident box is used.
DETECT_SCALES = (1.0, 0.5)

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # hide backend warnings, keep real errors


# ------------------------------------------------------------- 1. reading
def read_image(path):
    """Load a photo (JPG, PNG or iPhone HEIC) as a BGR array, longest side at most MAX_SIDE."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No image at {path}")
    if path.suffix.lower() in HEIC_SUFFIXES:
        img = cv2.cvtColor(np.asarray(open_heic(path)), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read {path.name} as an image")
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def open_heic(path):
    """An iPhone .heic photo as an upright RGB Pillow image, without metadata."""
    from PIL import Image
    from pillow_heif import register_heif_opener

    register_heif_opener()
    with Image.open(path) as photo:
        return without_metadata(photo)


def without_metadata(photo):
    """Upright RGB copy of a Pillow image with EXIF and XMP (where phones keep GPS) dropped; colour profile kept."""
    from PIL import ImageOps

    icc = photo.info.get("icc_profile") if photo.mode == "RGB" else None  # a CMYK/grey profile won't fit RGB
    clean = ImageOps.exif_transpose(photo).convert("RGB")
    # Pillow writes info["xmp"] back out when saving, so the copy's info is replaced, not just the EXIF.
    clean.info = {"icc_profile": icc} if icc else {}
    return clean


def has_metadata(photo):
    return bool(photo.getexif()) or any(key in photo.info for key in ("xmp", "XML:com.adobe.xmp"))


def same_face(a, b):
    """True when two boxes mostly overlap (by the smaller box's area): one face seen by two passes,
    even when one pass placed its box half a face off."""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a[:4], b[:4]
    overlap_w = min(ax + aw, bx + bw) - max(ax, bx)
    overlap_h = min(ay + ah, by + bh) - max(ay, by)
    if overlap_w <= 0 or overlap_h <= 0:
        return False
    return bool(overlap_w * overlap_h >= 0.5 * min(aw * ah, bw * bh))


class FaceEncoder:
    """Finds the face in a photo and turns it into a 128-number embedding."""

    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(str(DETECTOR_FILE), "", (320, 320), MIN_FACE_SCORE)
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_FILE), "")

    def detect_all(self, img):
        """Every distinct face found at any of DETECT_SCALES, most confident first.

        One row of 15 numbers per face (box, 5 landmarks, score) in img's pixels. Detections from
        all passes are merged, so a second face that only one pass sees still blocks a login.
        """
        found = []
        for scale in DETECT_SCALES:
            h, w = (int(round(n * scale)) for n in img.shape[:2])
            if min(h, w) < 1:
                continue
            small = img if scale == 1 else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(small)
            if faces is not None:
                faces[:, :14] /= scale
                found.extend(faces)
        distinct = []
        for face in sorted(found, key=lambda f: -f[-1]):
            if not any(same_face(face, kept) for kept in distinct):
                distinct.append(face)
        return np.array(distinct).reshape(-1, 15)

    def detect(self, img):
        """The most confident face, or None (for display)."""
        faces = self.detect_all(img)
        return faces[np.argmax(faces[:, -1])] if len(faces) else None

    def embed(self, img):
        """(unit-length embedding, number of faces found).

        The embedding is None unless there is exactly one face, so a second face in the frame
        (a held-up photo of a member, a bystander) can neither log in nor be enrolled.
        """
        faces = self.detect_all(img)
        if len(faces) != 1:
            return None, len(faces)
        aligned = self.recognizer.alignCrop(img, faces[0])
        vector = self.recognizer.feature(aligned).flatten().astype(np.float64)
        return vector / np.linalg.norm(vector), 1


# --------------------------------------------------- 2. collection check
def convert_heic(raw_dir=RAW):
    """Saves a JPG copy of each iPhone .heic photo (again if the .heic is newer); returns the new names."""
    converted = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() not in HEIC_SUFFIXES:
            continue
        target = path.with_suffix(".jpg")
        if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
            continue
        photo = open_heic(path)
        photo.save(target, "JPEG", quality=95, icc_profile=photo.info.get("icc_profile"))
        converted.append(target.name)
    return converted


def remove_location_data(raw_dir=RAW):
    """Re-saves photos that carry EXIF or XMP metadata without it; returns the names changed.

    Phones store the GPS location of every photo in that metadata, and this repository is
    public. The rotation tag is applied first, so the saved photo stays upright.
    """
    from PIL import Image

    cleaned = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as photo:
                if not has_metadata(photo):
                    continue
                clean = without_metadata(photo)
        except OSError:  # not an image: collection_report lists it as a bad photo
            continue
        clean.save(path, "PNG" if path.suffix.lower() == ".png" else "JPEG", quality=95,
                   icc_profile=clean.info.get("icc_profile"))
        cleaned.append(path.name)
    return cleaned


def parse_name(name):
    """(member, expression) for a name like aime_neutral.jpg (lowercase, as the audio files are), else None."""
    match = NAME_PATTERN.fullmatch(name)
    return (match[1], match[2]) if match else None


def unauthorized_photos(raw_dir=RAW):
    """The stranger photos for the rejection demo (.heic originals are converted to .jpg first)."""
    return [p for p in sorted(raw_dir.iterdir())
            if p.name.startswith(UNAUTHORIZED_PREFIX) and p.suffix.lower() in IMAGE_SUFFIXES]


def team_photos(raw_dir=RAW):
    """{(member, expression): path} for every correctly named team photo.

    The first file wins when a name repeats, and a file with the same bytes as an earlier one is
    skipped, so one selfie saved as two expressions can't sit on both sides of a train/test split.
    """
    photos, seen = {}, set()
    for path in sorted(raw_dir.iterdir()):
        key = parse_name(path.name)
        if not key or path.name.startswith(UNAUTHORIZED_PREFIX) or key in photos:
            continue
        try:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
        except OSError:  # unreadable: kept, so collection_report shows it with no face
            digest = path
        if digest not in seen:
            seen.add(digest)
            photos[key] = path
    return photos


def collection_report(encoder=None, raw_dir=RAW):
    """One row per team photo: valid name, not a duplicate, and exactly one face found."""
    encoder = encoder or FaceEncoder()
    photos = team_photos(raw_dir)
    rows = []
    for path in sorted(raw_dir.iterdir()):
        if path.name.startswith((".", UNAUTHORIZED_PREFIX)) or path.suffix.lower() in HEIC_SUFFIXES:
            continue
        key = parse_name(path.name)
        try:
            faces = encoder.detect_all(read_image(path))
        except (OSError, ValueError):  # not an image OpenCV can open
            faces = np.empty((0, 15))
        duplicate = bool(key) and photos.get(key) != path
        rows.append({
            "file": path.name,
            "member": key[0] if key else None,
            "expression": key[1] if key else None,
            "valid_name": bool(key),
            "duplicate": duplicate,
            "faces": len(faces),
            "face_score": round(float(faces[:, -1].max()), 3) if len(faces) else None,
            "ok": bool(key) and not duplicate and len(faces) == 1,
        })
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def missing_expressions(report):
    """{member: [expressions still missing]} for every member with a gap (a duplicate photo doesn't count)."""
    named = report[report["valid_name"].astype(bool)]
    usable = named[~named["duplicate"].astype(bool)]
    have = {m: set(usable.loc[usable["member"] == m, "expression"]) for m in named["member"].unique()}
    return {m: [e for e in EXPRESSIONS if e not in got] for m, got in have.items() if set(EXPRESSIONS) - got}


# ---------------------------------------------------------- 3. augment
def rotate(img, angle):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # Replicate, not reflect, the edges: a mirrored border could create a second partial face.
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


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
    """Augments every team photo, saves the copies and returns (features, skipped images with the reason)."""
    encoder = encoder or FaceEncoder()
    augmented_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for (member, expression), path in sorted(team_photos(raw_dir).items()):
        try:
            versions = augment(read_image(path))
        except (OSError, ValueError) as exc:
            skipped.append(f"{path.name}: {exc}")
            continue
        for name, img in versions.items():
            if name != "original":
                cv2.imwrite(str(augmented_dir / f"{member}_{expression}_{name}.jpg"), img)
            embedding, n_faces = encoder.embed(img)
            if embedding is None:
                skipped.append(f"{path.name} ({name}): {n_faces} faces found")
                continue
            rows.append({"member": member, "expression": expression, "augmentation": name,
                         "source_image": path.name, **dict(zip(EMBEDDING_COLUMNS, embedding))})
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS), skipped


def main():
    converted = convert_heic()
    if converted:
        print(f"Converted {len(converted)} iPhone photos to JPG: {', '.join(converted)}")
    cleaned = remove_location_data()
    if cleaned:
        print(f"Removed location metadata from: {', '.join(cleaned)}")
    encoder = FaceEncoder()
    report = collection_report(encoder)
    if report.empty:
        raise SystemExit(f"No team photos in {RAW.relative_to(ROOT)}; expected names like aime_neutral.jpg")
    print(report.to_string(index=False), "\n")
    for problem, files in [("Bad file names", report.loc[~report["valid_name"], "file"]),
                           ("Duplicate photos (only the first is used)", report.loc[report["duplicate"], "file"]),
                           ("Not exactly one face", report.loc[report["faces"] != 1, "file"])]:
        if len(files):
            print(f"{problem}: {', '.join(files)}")
    for member, missing in missing_expressions(report).items():
        print(f"{member} is missing: {', '.join(missing)}")

    features, skipped = build_features(encoder)
    if features.empty:
        raise SystemExit("No face could be extracted from any photo, so nothing was saved.")
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_FILE, index=False)
    print(f"\nSaved {len(features)} rows ({features['source_image'].nunique()} photos x up to "
          f"{len(AUGMENTATIONS) + 1} versions) to {FEATURES_FILE.relative_to(ROOT)}")
    if skipped:
        print(f"Skipped {len(skipped)} images: {'; '.join(skipped)}")


if __name__ == "__main__":
    main()
