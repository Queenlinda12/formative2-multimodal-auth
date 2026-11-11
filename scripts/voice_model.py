"""
Task 4 - Voiceprint Verification Model
======================================
Author: Agnes Mbabazi (Audio lead)

Trains the voice model on features/audio_features.csv, evaluates it, and saves:

    models/voice_model.joblib

Run from the repository root, after scripts/audio_pipeline.py:
    python scripts/voice_model.py

Use it from the app (scripts/main.py):
    from voice_model import verify_voice
    verify_voice("audio/processed/sheja_yesapprove.wav")   # -> "sheja", or "unauthorized"

Design decisions (explained in the notebook and report):
  * Two Logistic Regression models on the 34 audio features of every 0.5 s window:
      - speaker model: WHO is talking (agnes, aime, naillah, sheja)
      - phrase model:  WHAT they said (yesapprove or confirmtransaction)
    A clip's answer is the average of its windows' probabilities. Features are standardised
    first (MFCCs are in the hundreds, zero-crossing rate is below 1), and classes are
    weighted so members with longer recordings don't dominate.
  * Evaluation holds out whole recordings: a clip AND its 5 augmented copies are always on
    the same side (StratifiedGroupKFold grouped by `source_clip`). Testing on a pitch-shifted
    copy of a training clip would overstate accuracy.
  * "unauthorized": a classifier always names one of the members, even for a stranger.
    So a voice is only accepted if the speaker model is at least SPEAKER_THRESHOLD sure,
    AND the phrase model is at least PHRASE_THRESHOLD sure it heard an approved phrase.
    A leave-one-member-out test checks that strangers are rejected.
"""

import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_pipeline import (FEATURES_FILE, NUMERIC_COLUMNS, PROCESSED, ROOT, SAMPLE_RATE, UNAUTHORIZED_PREFIX,
                            augment, extract_features, load_clip)

MODEL_FILE = ROOT / "models" / "voice_model.joblib"
UNAUTHORIZED = "unauthorized"
SPEAKER_THRESHOLD = 0.5  # TODO after evaluation
PHRASE_THRESHOLD = 0.5   # TODO after evaluation
RANDOM_STATE = 42


# ----------------------------------------------------------- 1. model
def make_classifier():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE))


def fit(features):
    """Trains the speaker model and the phrase model on window rows."""
    X = features[NUMERIC_COLUMNS].to_numpy()
    return {
        "speaker": make_classifier().fit(X, features["member"]),
        "phrase": make_classifier().fit(X, features["phrase"]),
        "speaker_threshold": SPEAKER_THRESHOLD,
        "phrase_threshold": PHRASE_THRESHOLD,
    }


def clip_probabilities(classifier, windows):
    """One probability per class for a whole clip: the average over its windows."""
    return classifier.predict_proba(windows).mean(axis=0)


def decide(model, windows):
    """Best speaker and phrase guesses, then the two confidence checks that turn strangers into 'unauthorized'."""
    speaker_p = clip_probabilities(model["speaker"], windows)
    phrase_p = clip_probabilities(model["phrase"], windows)
    guess, probability = model["speaker"].classes_[speaker_p.argmax()], float(speaker_p.max())
    phrase, phrase_probability = model["phrase"].classes_[phrase_p.argmax()], float(phrase_p.max())
    voice_ok = probability >= model["speaker_threshold"]
    phrase_ok = phrase_probability >= model["phrase_threshold"]
    if voice_ok and phrase_ok:
        reason = "matches an enrolled member saying an approved phrase"
    elif not voice_ok:
        reason = f"closest member {guess} is only {probability:.2f} likely (needs {model['speaker_threshold']})"
    else:
        reason = f"phrase not recognised clearly ({phrase} at {phrase_probability:.2f}, needs {model['phrase_threshold']})"
    return {"member": guess if voice_ok and phrase_ok else UNAUTHORIZED, "guess": guess,
            "probability": probability, "phrase": phrase, "phrase_probability": phrase_probability,
            "reason": reason}


def windows_of(features):
    """{(source_clip, augmentation): window rows}: one entry per original or augmented clip."""
    return {key: rows for key, rows in features.groupby(["source_clip", "augmentation"], sort=True)}


# ------------------------------------------------------ 2. evaluation
def evaluate(features):
    """Cross-validation with as many folds as every member has recordings for (at most 3)."""
    clips = features.groupby("member")["source_clip"].nunique()
    if clips.min() < 2:
        short = ", ".join(f"{m} has {n}" for m, n in clips.items() if n < 2)
        raise ValueError(f"Every member needs at least 2 recordings to be evaluated ({short}).")
    return cross_validate(features, n_splits=min(3, int(clips.min())))


def cross_validate(features, n_splits=2):
    """One held-out prediction per clip version; each fold holds out whole recordings with their augmentations."""
    X, y, groups = features[NUMERIC_COLUMNS].to_numpy(), features["member"].to_numpy(), features["source_clip"]
    classes = np.unique(y)
    window_p = np.zeros((len(features), len(classes)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for train, test in splitter.split(X, y, groups):
        model = fit(features.iloc[train])
        window_p[test] = model["speaker"].predict_proba(X[test])
        for (clip, aug), w in windows_of(features.iloc[test]).items():
            d = decide(model, w[NUMERIC_COLUMNS].to_numpy())
            rows.append({"source_clip": clip, "augmentation": aug, "member": w["member"].iloc[0],
                         "phrase": w["phrase"].iloc[0], "predicted": d["guess"], "probability": d["probability"],
                         "accepted_as": d["member"], **dict(zip(classes, clip_probabilities(model["speaker"],
                                                                                            w[NUMERIC_COLUMNS].to_numpy())))})
    return pd.DataFrame(rows), classes


def metrics(out, classes):
    return {
        "accuracy": accuracy_score(out["member"], out["predicted"]),
        "f1_macro": f1_score(out["member"], out["predicted"], average="macro"),
        "log_loss": log_loss(out["member"], out[list(classes)].to_numpy(), labels=classes),
        "accepted_correctly": float((out["accepted_as"] == out["member"]).mean()),
    }


def phrase_test(features):
    """Phrase model accuracy on members it has never heard (each member held out in turn)."""
    correct = []
    for train, test in LeaveOneGroupOut().split(features, groups=features["member"]):
        model = fit(features.iloc[train])
        for _, w in windows_of(features.iloc[test]).items():
            p = clip_probabilities(model["phrase"], w[NUMERIC_COLUMNS].to_numpy())
            correct.append(model["phrase"].classes_[p.argmax()] == w["phrase"].iloc[0])
    return float(np.mean(correct))


def impostor_test(features):
    """Leave one member out of training entirely and check their clips come back 'unauthorized'."""
    rows = []
    for member in sorted(features["member"].unique()):
        model = fit(features[features["member"] != member])
        decisions = [decide(model, w[NUMERIC_COLUMNS].to_numpy())
                     for _, w in windows_of(features[features["member"] == member]).items()]
        rows.append({"left_out": member, "clips": len(decisions),
                     "rejected": sum(d["member"] == UNAUTHORIZED for d in decisions),
                     "highest_probability": round(max(d["probability"] for d in decisions), 3),
                     "probabilities": [d["probability"] for d in decisions]})
    return pd.DataFrame(rows, columns=["left_out", "clips", "rejected", "highest_probability", "probabilities"])


def unauthorized_test(model, folder=PROCESSED):
    """Every unauthorized_* clip and its augmentations, through the given model."""
    rows = []
    for path in sorted(folder.glob(f"{UNAUTHORIZED_PREFIX}*.wav")):
        for name, y in augment(load_clip(path)).items():
            d = decide(model, windows_from_audio(y))
            rows.append({"clip": path.name, "augmentation": name, "result": d["member"],
                         "closest_member": d["guess"], "probability": round(d["probability"], 3),
                         "phrase": d["phrase"], "phrase_probability": round(d["phrase_probability"], 3)})
    return pd.DataFrame(rows)


# ------------------------------------------------------- 3. for the app
def windows_from_audio(y):
    return pd.DataFrame(extract_features(y))[NUMERIC_COLUMNS].to_numpy()


def read_any(audio_path):
    """A clip from any format: WAV directly, anything else (e.g. a phone .m4a) through ffmpeg first."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"No audio at {path}")
    if path.suffix.lower() == ".wav":
        return load_clip(path)
    if shutil.which("ffmpeg") is None:
        raise ValueError(f"{path.name} is not a WAV and ffmpeg is not installed to convert it")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "clip.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
                        str(wav)], check=True)
        return load_clip(wav)


@lru_cache(maxsize=1)
def _load():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"{MODEL_FILE} not found. Run: python scripts/voice_model.py")
    return joblib.load(MODEL_FILE)


def voice_details(audio_path):
    """Everything the decision was based on: member, guess, probability, phrase, phrase_probability, reason."""
    model = _load()
    try:
        y = read_any(audio_path)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:  # missing or unreadable: deny, don't crash
        return {"member": UNAUTHORIZED, "guess": None, "probability": 0.0, "phrase": None,
                "phrase_probability": 0.0, "reason": str(exc)}
    return decide(model, windows_from_audio(y))


def verify_voice(audio_path):
    """The member's name if the voice is recognised saying an approved phrase, otherwise "unauthorized"."""
    return voice_details(audio_path)["member"]


def main():
    features = pd.read_csv(FEATURES_FILE)
    members = sorted(features["member"].unique())
    if len(members) < 2:
        raise SystemExit(f"Need recordings of at least 2 members to train; found {members}.")
    print(f"{len(features)} window rows, {features['source_clip'].nunique()} recordings, members: {', '.join(members)}\n")

    try:
        out, classes = evaluate(features)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print("Cross-validation (whole recordings held out, one decision per clip):")
    for name, value in metrics(out, classes).items():
        print(f"  {name:<22} {value:.3f}")
    print("\nConfusion matrix (rows = true member):")
    print(pd.DataFrame(confusion_matrix(out["member"], out["predicted"], labels=classes),
                       index=classes, columns=classes))
    print(f"\nPhrase model accuracy on unseen speakers: {phrase_test(features):.3f}")

    print("\nLeave-one-member-out (each member treated as a stranger):")
    print(impostor_test(features).drop(columns="probabilities").to_string(index=False))

    model = fit(features)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved {MODEL_FILE.relative_to(ROOT)} (trained on all {len(features)} rows)")

    unauthorized = unauthorized_test(model)
    if len(unauthorized):
        print("\nUnauthorized voice (and its augmentations):")
        print(unauthorized.to_string(index=False))


if __name__ == "__main__":
    main()
