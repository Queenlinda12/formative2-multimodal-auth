"""
Task 4 - Voiceprint Verification Model
======================================
Author: Agnes Mbabazi (Audio lead), GitHub @agmbabazi

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
    So a voice is only accepted if the speaker model is sure enough (the speaker threshold),
    AND the phrase model is at least PHRASE_THRESHOLD sure it heard an approved phrase.
  * The speaker threshold is chosen automatically at the Equal Error Rate: the point where
    the False Acceptance Rate (strangers let in, measured by leaving each member out of
    training in turn) equals the False Rejection Rate (members turned away, measured on
    held-out recordings). Retraining on new recordings re-tunes it, with no manual step.
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
DEFAULT_SPEAKER_THRESHOLD = 0.5  # only used until main() tunes it at the Equal Error Rate
PHRASE_THRESHOLD = 0.6  # 0.5 would always pass (two phrases); 0.6 asks for a clear answer
THRESHOLD_GRID = np.round(np.arange(0.30, 0.96, 0.01), 2)
RANDOM_STATE = 42


# ----------------------------------------------------------- 1. model
def make_classifier():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE))


def fit(features, speaker_threshold=DEFAULT_SPEAKER_THRESHOLD):
    """Trains the speaker model and the phrase model on window rows."""
    X = features[NUMERIC_COLUMNS].to_numpy()
    return {
        "speaker": make_classifier().fit(X, features["member"]),
        "phrase": make_classifier().fit(X, features["phrase"]),
        "speaker_threshold": speaker_threshold,
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
                         "phrase_probability": d["phrase_probability"],
                         **dict(zip(classes, clip_probabilities(model["speaker"], w[NUMERIC_COLUMNS].to_numpy())))})
    return pd.DataFrame(rows), classes


def metrics(out, classes):
    return {
        "accuracy": accuracy_score(out["member"], out["predicted"]),
        "f1_macro": f1_score(out["member"], out["predicted"], average="macro"),
        "log_loss": log_loss(out["member"], out[list(classes)].to_numpy(), labels=classes),
    }


def genuine_scores(out):
    """A member's held-out clip only counts as accepted if the RIGHT member was named, so a wrong name scores 0."""
    return np.where(out["predicted"] == out["member"], out["probability"], 0.0)


def error_rates(genuine, impostor, threshold):
    """(FAR, FRR): share of stranger clips let in, share of member clips turned away."""
    return float(np.mean(np.asarray(impostor) >= threshold)), float(np.mean(np.asarray(genuine) < threshold))


def error_rate_table(genuine, impostor, grid=THRESHOLD_GRID):
    return pd.DataFrame([(t, *error_rates(genuine, impostor, t)) for t in grid], columns=["threshold", "FAR", "FRR"])


def choose_threshold(genuine, impostor, grid=THRESHOLD_GRID):
    """The Equal Error Rate threshold: where FAR and FRR are closest (ties go to the stricter threshold)."""
    table = error_rate_table(genuine, impostor, grid)
    gap = (table["FAR"] - table["FRR"]).abs()
    best = table[gap == gap.min()].iloc[-1]
    return float(best["threshold"]), float(best["FAR"]), float(best["FRR"])


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
    """Leave one member out of training entirely: how sure is the model that this 'stranger' is someone it knows?"""
    rows = []
    for member in sorted(features["member"].unique()):
        model = fit(features[features["member"] != member])
        decisions = [decide(model, w[NUMERIC_COLUMNS].to_numpy())
                     for _, w in windows_of(features[features["member"] == member]).items()]
        rows.append({"left_out": member, "clips": len(decisions),
                     "highest_probability": round(max(d["probability"] for d in decisions), 3),
                     "probabilities": [d["probability"] for d in decisions],
                     "phrase_probabilities": [d["phrase_probability"] for d in decisions]})
    return pd.DataFrame(rows, columns=["left_out", "clips", "highest_probability", "probabilities",
                                       "phrase_probabilities"])


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

    # --- verification: choose the speaker threshold from FAR / FRR
    impostors = impostor_test(features)
    genuine = genuine_scores(out)
    impostor = np.concatenate(impostors["probabilities"].to_list())
    threshold, far, frr = choose_threshold(genuine, impostor)
    print("\nFAR / FRR by speaker threshold (FAR from leaving each member out, FRR from held-out recordings):")
    table = error_rate_table(genuine, impostor)
    print(table[table["threshold"].isin(np.round(np.arange(0.4, 0.95, 0.05), 2)) | (table["threshold"] == threshold)]
          .to_string(index=False, float_format="%.2f"))
    print(f"Chosen threshold {threshold:.2f} (Equal Error Rate): FAR {far:.2f}, FRR {frr:.2f}")

    # the full decision, with the phrase check too
    phrase_ok = out["phrase_probability"].to_numpy() >= PHRASE_THRESHOLD
    imp_phrase_ok = np.concatenate(impostors["phrase_probabilities"].to_list()) >= PHRASE_THRESHOLD
    full_far, full_frr = error_rates(np.where(phrase_ok, genuine, 0.0), np.where(imp_phrase_ok, impostor, 0.0), threshold)
    print(f"With the phrase check (>= {PHRASE_THRESHOLD}): FAR {full_far:.2f}, FRR {full_frr:.2f}")

    model = fit(features, speaker_threshold=threshold)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved {MODEL_FILE.relative_to(ROOT)} (trained on all {len(features)} rows, threshold {threshold:.2f})")

    unauthorized = unauthorized_test(model)
    if len(unauthorized):
        print("\nUnauthorized voice (and its augmentations):")
        print(unauthorized.to_string(index=False))


if __name__ == "__main__":
    main()
