"""
Task 4 - Facial Recognition Model
=================================
Author: Aime Igirimpuhwe (Image lead), GitHub @igaimerca

Trains the face model on features/image_features.csv, evaluates it, and saves:

    models/face_model.joblib

Run from the repository root, after scripts/image_pipeline.py:
    python scripts/face_model.py

Use it from the app (scripts/main.py):
    from face_model import predict_face
    predict_face("images/raw/sheja_neutral.jpg")   # -> "sheja", or "unknown"

Design decisions (explained in the notebook and report):
  * Logistic Regression on the 128 embedding numbers. With a handful of photos per
    person the embeddings of different people are far apart, so a simple linear
    model separates them, and its probabilities give a usable log loss.
  * Evaluation holds out whole photos: every fold keeps one original photo per member
    AND its 5 augmented copies out of training (StratifiedGroupKFold grouped by
    `source_image`). Testing on a rotated copy of a training photo would overstate accuracy.
  * "unknown": a classifier always names one of the members, even for a stranger.
    So a face is only accepted when it is also close to that member's enrolled photos:
    cosine similarity of at least 0.363, the threshold SFace's authors publish for
    "same person". A leave-one-member-out test checks that strangers are rejected.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from image_pipeline import (EMBEDDING_COLUMNS, FEATURES_FILE, RAW, ROOT, UNAUTHORIZED_PREFIX,
                            FaceEncoder, augment, read_image)

MODEL_FILE = ROOT / "models" / "face_model.joblib"
UNKNOWN = "unknown"
SIMILARITY_THRESHOLD = 0.363  # SFace's published cosine threshold for "same person"
RANDOM_STATE = 42


# ----------------------------------------------------------- 1. model
def make_classifier():
    return LogisticRegression(C=10, max_iter=2000, random_state=RANDOM_STATE)


def fit(features):
    """Trains the classifier and keeps every embedding as the 'gallery' for the similarity check."""
    X, y = features[EMBEDDING_COLUMNS].to_numpy(), features["member"].to_numpy()
    return {
        "classifier": make_classifier().fit(X, y),
        "gallery": X,
        "gallery_members": y,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }


def decide(model, embedding):
    """Classifier's best guess, then the similarity check that turns strangers into 'unknown'."""
    if embedding is None:
        return {"member": UNKNOWN, "guess": None, "probability": 0.0, "similarity": 0.0,
                "reason": "no face found in the image"}
    probabilities = model["classifier"].predict_proba(embedding.reshape(1, -1))[0]
    best = int(np.argmax(probabilities))
    guess = model["classifier"].classes_[best]
    similarity = float(np.max(model["gallery"][model["gallery_members"] == guess] @ embedding))
    accepted = similarity >= model["similarity_threshold"]
    return {
        "member": guess if accepted else UNKNOWN,
        "guess": guess,
        "probability": float(probabilities[best]),
        "similarity": similarity,
        "reason": "matches an enrolled member" if accepted
                  else f"closest member {guess} is only {similarity:.2f} similar (needs {model['similarity_threshold']})",
    }


# ------------------------------------------------------ 2. evaluation
def cross_validate(features, n_splits=3):
    """Held-out predictions for every row; each fold holds out whole photos with their augmentations."""
    X, y = features[EMBEDDING_COLUMNS].to_numpy(), features["member"].to_numpy()
    groups = features["source_image"].to_numpy()
    classes = np.unique(y)
    out = features[["member", "expression", "augmentation", "source_image"]].copy()
    out["predicted"], out["accepted_as"], out["similarity"] = None, None, 0.0
    probabilities = np.zeros((len(features), len(classes)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train, test in splitter.split(X, y, groups):
        model = fit(features.iloc[train])
        probabilities[test] = model["classifier"].predict_proba(X[test])
        out.iloc[test, out.columns.get_loc("predicted")] = model["classifier"].predict(X[test])
        decisions = [decide(model, X[i]) for i in test]
        out.iloc[test, out.columns.get_loc("accepted_as")] = [d["member"] for d in decisions]
        out.iloc[test, out.columns.get_loc("similarity")] = [d["similarity"] for d in decisions]
    return out, probabilities, classes


def metrics(out, probabilities, classes):
    return {
        "accuracy": accuracy_score(out["member"], out["predicted"]),
        "f1_macro": f1_score(out["member"], out["predicted"], average="macro"),
        "log_loss": log_loss(out["member"], probabilities, labels=classes),
        "accepted_correctly": float((out["accepted_as"] == out["member"]).mean()),
        "lowest_genuine_similarity": float(out["similarity"].min()),
    }


def impostor_test(features):
    """Leave one member out of training entirely and check their photos come back 'unknown'."""
    rows = []
    for member in sorted(features["member"].unique()):
        known = features[features["member"] != member]
        if known["member"].nunique() < 2:
            continue
        model = fit(known)
        stranger = features[features["member"] == member][EMBEDDING_COLUMNS].to_numpy()
        decisions = [decide(model, e) for e in stranger]
        rows.append({
            "similarities": [d["similarity"] for d in decisions],
            "left_out": member,
            "images": len(decisions),
            "rejected": sum(d["member"] == UNKNOWN for d in decisions),
            "highest_similarity": round(max(d["similarity"] for d in decisions), 3),
        })
    return pd.DataFrame(rows)


def unauthorized_test(model, encoder=None, raw_dir=RAW):
    """Every unauthorized_* photo and its augmentations, through the saved model."""
    encoder = encoder or FaceEncoder()
    rows = []
    for path in sorted(raw_dir.glob(f"{UNAUTHORIZED_PREFIX}*")):
        for name, img in augment(read_image(path)).items():
            d = decide(model, encoder.embed(img))
            rows.append({"image": path.name, "augmentation": name, "result": d["member"],
                         "closest_member": d["guess"], "similarity": round(d["similarity"], 3)})
    return pd.DataFrame(rows)


# ------------------------------------------------------- 3. for the app
@lru_cache(maxsize=1)
def _load():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"{MODEL_FILE} not found. Run: python scripts/face_model.py")
    return joblib.load(MODEL_FILE), FaceEncoder()


def face_details(image_path):
    """Everything the decision was based on: member, guess, probability, similarity, reason."""
    model, encoder = _load()
    return decide(model, encoder.embed(read_image(Path(image_path))))


def predict_face(image_path):
    """The member's name if the face is recognised, otherwise "unknown"."""
    return face_details(image_path)["member"]


def main():
    features = pd.read_csv(FEATURES_FILE)
    members = sorted(features["member"].unique())
    if len(members) < 2:
        raise SystemExit(f"Need photos of at least 2 members to train; found {members}.")
    print(f"{len(features)} rows, {features['source_image'].nunique()} photos, members: {', '.join(members)}\n")

    photos_per_member = features.groupby("member")["source_image"].nunique().min()
    out, probabilities, classes = cross_validate(features, n_splits=min(3, photos_per_member))
    scores = metrics(out, probabilities, classes)
    print("Cross-validation (whole photos held out):")
    for name, value in scores.items():
        print(f"  {name:<26} {value:.3f}")
    print("\nConfusion matrix (rows = true member):")
    print(pd.DataFrame(confusion_matrix(out["member"], out["predicted"], labels=classes),
                       index=classes, columns=classes))

    impostors = impostor_test(features)
    if len(impostors):
        print("\nLeave-one-member-out (each member treated as a stranger):")
        print(impostors.drop(columns="similarities").to_string(index=False))

    model = fit(features)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved {MODEL_FILE.relative_to(ROOT)} (trained on all {len(features)} rows)")

    unauthorized = unauthorized_test(model)
    if len(unauthorized):
        print("\nUnauthorized face (and its augmentations):")
        print(unauthorized.to_string(index=False))


if __name__ == "__main__":
    main()
