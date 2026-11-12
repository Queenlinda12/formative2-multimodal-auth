"""
Edge cases for the image pipeline and face model.
Author: Aime Igirimpuhwe (Image lead), GitHub @igaimerca

Run from the repository root, after scripts/face_model.py has saved the model:
    python -m pytest tests
"""

import io
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageCms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import face_model as fm  # noqa: E402
import image_pipeline as ip  # noqa: E402


@pytest.fixture(scope="module")
def encoder():
    return ip.FaceEncoder()


def photo(member, expression):
    return ip.RAW / f"{member}_{expression}.jpg"


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------ file names
@pytest.mark.parametrize("name, expected", [
    ("aime_neutral.jpg", ("aime", "neutral")),
    ("aime_smiling.JPEG", ("aime", "smiling")),
    ("sheja_surprised.png", ("sheja", "surprised")),
    ("Aime_neutral.jpg", None),         # lowercase, like the audio files
    ("aıme_neutral.jpg", None),    # dotless i: prints like "aime" but would be another member
    ("aime_neutral.jpg\n", None),
    ("aime_happy.jpg", None),
    ("aime_neutral.heic", None),        # converted to .jpg first
])
def test_parse_name(name, expected):
    assert ip.parse_name(name) == expected


def test_same_photo_under_two_names_counts_once(tmp_path, encoder):
    shutil.copy(photo("aime", "neutral"), tmp_path / "aime_neutral.jpg")
    shutil.copy(photo("aime", "neutral"), tmp_path / "aime_smiling.jpg")
    assert ip.team_photos(tmp_path) == {("aime", "neutral"): tmp_path / "aime_neutral.jpg"}

    report = ip.collection_report(encoder, tmp_path)
    smiling = report.set_index("file").loc["aime_smiling.jpg"]
    assert smiling["duplicate"] and not smiling["ok"]
    assert ip.missing_expressions(report) == {"aime": ["smiling", "surprised"]}


def test_member_whose_photos_are_all_duplicates_is_missing_everything(tmp_path, encoder):
    for expression in ip.EXPRESSIONS:
        shutil.copy(photo("aime", expression), tmp_path / f"aime_{expression}.jpg")
        shutil.copy(photo("aime", expression), tmp_path / f"sheja_{expression}.jpg")
    assert ip.missing_expressions(ip.collection_report(encoder, tmp_path)) == {"sheja": ip.EXPRESSIONS}


def test_broken_files_are_reported_not_a_crash(tmp_path, encoder):
    (tmp_path / "sheja_neutral.jpg").write_text("not a photo")
    (tmp_path / "aime_smiling.jpg").mkdir()
    assert ip.remove_location_data(tmp_path) == []
    report = ip.collection_report(encoder, tmp_path)
    assert not report["ok"].any() and (report["faces"] == 0).all()


def test_unauthorized_photos_are_kept_apart(tmp_path):
    shutil.copy(ip.RAW / "unauthorized_face.jpg", tmp_path / "unauthorized_face.jpg")
    shutil.copy(photo("sheja", "neutral"), tmp_path / "unauthorized_neutral.jpg")
    shutil.copy(photo("aime", "neutral"), tmp_path / "aime_neutral.jpg")
    (tmp_path / "unauthorized_face.heic").write_bytes(b"")

    assert ip.unauthorized_photos(tmp_path) == [tmp_path / "unauthorized_face.jpg",
                                                 tmp_path / "unauthorized_neutral.jpg"]
    assert list(ip.team_photos(tmp_path)) == [("aime", "neutral")]


# -------------------------------------------------------------- metadata
def test_location_is_removed_and_colour_profile_kept(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # phone held sideways: the saved photo must come out upright
    exif[0x8825] = {1: "N", 2: (1.0, 2.0, 3.0), 3: "E", 4: (4.0, 5.0, 6.0)}  # GPS
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"/>'
    Image.new("RGB", (40, 20), "red").save(tmp_path / "geo.jpg", "JPEG", exif=exif.tobytes(), icc_profile=icc, xmp=xmp)
    Image.new("RGB", (40, 20), "red").save(tmp_path / "clean.jpg", "JPEG")
    clean_bytes = (tmp_path / "clean.jpg").read_bytes()

    assert ip.remove_location_data(tmp_path) == ["geo.jpg"]
    with Image.open(tmp_path / "geo.jpg") as saved:
        assert not saved.getexif() and "xmp" not in saved.info
        assert saved.info.get("icc_profile") == icc
        assert saved.size == (20, 40)
    assert (tmp_path / "clean.jpg").read_bytes() == clean_bytes  # untouched, not re-encoded
    assert ip.remove_location_data(tmp_path) == []


def test_no_photo_in_the_repo_has_location_data():
    for folder in (ip.RAW, ip.AUGMENTED):
        for path in folder.iterdir():
            if path.suffix.lower() in ip.IMAGE_SUFFIXES:
                with Image.open(path) as im:
                    assert not ip.has_metadata(im), f"{path.name} still has EXIF/XMP: run scripts/image_pipeline.py"


# ------------------------------------------------------------- detection
def test_close_up_face_is_lined_up(encoder):
    """At full size alone, YuNet put this close-up's box half a face too low (similarity 0.29)."""
    neutral, _ = encoder.embed(ip.read_image(photo("agnes", "neutral")))
    surprised, _ = encoder.embed(ip.read_image(photo("agnes", "surprised")))
    assert surprised @ neutral > 0.6


def test_face_filling_a_small_frame(encoder):
    """A login photo of 960 px or less is never downscaled, so it relies on the half-size pass."""
    img = cv2.imread(str(photo("agnes", "surprised")))
    h, w = img.shape[:2]
    close_up = cv2.resize(img[int(h * .12):int(h * .78), int(w * .08):int(w * .92)], (720, 960))
    neutral, _ = encoder.embed(ip.read_image(photo("agnes", "neutral")))
    embedding, n_faces = encoder.embed(close_up)
    assert n_faces == 1 and embedding @ neutral > 0.6


def test_two_faces_are_refused(encoder):
    a = ip.read_image(photo("aime", "neutral"))
    s = cv2.resize(ip.read_image(photo("sheja", "neutral")), (a.shape[1], a.shape[0]))
    embedding, n_faces = encoder.embed(np.hstack([a, s]))
    assert embedding is None and n_faces == 2


def test_second_face_seen_by_only_one_pass_is_refused(encoder):
    """A big close-up (found at half size) plus a small member photo in the corner (found at full size)."""
    img = ip.augment(ip.read_image(photo("agnes", "surprised")))["rotated"]
    member = ip.read_image(photo("aime", "neutral"))
    x, y, w, h = encoder.detect(member)[:4].astype(int)
    small_face = cv2.resize(member[max(y - h // 4, 0):y + h + h // 4, max(x - w // 4, 0):x + w + w // 4], (40, 50))
    img[-60:-10, 10:50] = small_face
    embedding, n_faces = encoder.embed(img)
    assert embedding is None and n_faces >= 2


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 50, 3), (50, 1, 3)])
def test_tiny_image_is_no_face_not_a_crash(encoder, shape):
    assert encoder.embed(np.zeros(shape, np.uint8)) == (None, 0)


# ------------------------------------------------------------- decisions
@pytest.fixture
def toy():
    """Two members with clearly separate embeddings, plus a model trained on them."""
    rng = np.random.default_rng(0)
    centers = {"aime": unit(rng.normal(size=128)), "sheja": unit(rng.normal(size=128))}
    rows = [{"member": m, "augmentation": "original" if i < 3 else "flipped", "source_image": f"{m}_{i % 3}.jpg",
             **dict(zip(ip.EMBEDDING_COLUMNS, unit(c + 0.05 * rng.normal(size=128))))}
            for m, c in centers.items() for i in range(6)]
    features = pd.DataFrame(rows)
    return fm.fit(features), centers, features, rng


def test_member_is_accepted(toy):
    model, centers, _, _ = toy
    assert fm.decide(model, centers["sheja"])["member"] == "sheja"


def test_stranger_is_unknown_even_though_the_classifier_guesses(toy):
    model, _, _, rng = toy
    d = fm.decide(model, unit(rng.normal(size=128)))
    assert d["member"] == fm.UNKNOWN and d["guess"] in {"aime", "sheja"} and d["similarity"] < fm.SIMILARITY_THRESHOLD


@pytest.mark.parametrize("n_faces, reason", [(0, "no face"), (2, "2 faces")])
def test_no_face_or_several_faces_is_unknown(toy, n_faces, reason):
    d = fm.decide(toy[0], None, n_faces)
    assert d["member"] == fm.UNKNOWN and reason in d["reason"]


def test_one_known_member_still_rejects_strangers(toy):
    _, centers, features, rng = toy
    model = fm.fit(features[features["member"] == "aime"])
    assert model["classifier"] is None
    assert fm.decide(model, centers["aime"])["member"] == "aime"
    assert fm.decide(model, unit(rng.normal(size=128)))["member"] == fm.UNKNOWN


def test_member_with_one_photo_cannot_be_evaluated(toy):
    features = toy[2]
    one_photo = features[(features["member"] == "aime") & (features["source_image"] == "aime_0.jpg")]
    with pytest.raises(ValueError, match="aime has 1"):
        fm.evaluate(pd.concat([features[features["member"] == "sheja"], one_photo]))


# ---------------------------------------------------------- predict_face
def test_predict_face_on_the_real_photos():
    assert fm.predict_face(photo("aime", "neutral")) == "aime"
    assert fm.predict_face(ip.RAW / "unauthorized_face.jpg") == fm.UNKNOWN


def test_predict_face_denies_bad_files(tmp_path):
    (tmp_path / "notes.jpg").write_text("not a photo")
    assert fm.predict_face(tmp_path / "missing.jpg") == fm.UNKNOWN
    assert fm.predict_face(tmp_path / "notes.jpg") == fm.UNKNOWN
