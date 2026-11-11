"""
Task 3 - Sound Data Collection and Processing
Author: Agnes Mbabazi

Converts and checks every member's voice recordings, augments them, and saves the
features to features/audio_features.csv (one row per 0.5 s window of each clip).

Run from the repository root:
    python scripts/audio_pipeline.py
"""

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "audio" / "raw"
PROCESSED = ROOT / "audio" / "processed"
AUGMENTED = ROOT / "audio" / "augmented"
FEATURES_FILE = ROOT / "features" / "audio_features.csv"

SAMPLE_RATE = 16_000
TRIM_DB = 30  # anything 30 dB below the loudest point counts as silence
PHRASES = ["yesapprove", "confirmtransaction"]
# name_phrase.wav, or name_phrase_2.wav for extra takes
NAME_PATTERN = re.compile(r"^([a-z]+)_(yesapprove|confirmtransaction)(?:_(\d+))?\.wav$")
UNAUTHORIZED_PREFIX = "unauthorized"
REPORT_COLUMNS = ["file", "member", "phrase", "valid_name", "duplicate_of", "duration_s", "peak", "ok"]
MIN_SECONDS, MAX_SECONDS = 0.5, 5.0
CLIP_PEAK = 0.99  # a peak at 1.0 means the recording is distorted
TARGET_PEAK = 0.9
N_MFCC = 13
SEGMENT_SECONDS, SEGMENT_HOP = 0.5, 0.25
SUMMARY_FEATURES = ["rolloff", "centroid", "zcr", "rms"]
FEATURE_NAMES = [f"mfcc_{i:02d}" for i in range(1, N_MFCC + 1)] + SUMMARY_FEATURES
NUMERIC_COLUMNS = [f"{name}_{stat}" for name in FEATURE_NAMES for stat in ("mean", "std")]
FEATURE_COLUMNS = ["member", "phrase", "augmentation", "source_clip", "segment", *NUMERIC_COLUMNS]


# 1. Reading
def ffmpeg_exe():
    """System ffmpeg if installed, otherwise the copy that comes with the imageio-ffmpeg package."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise SystemExit("ffmpeg not found. Run: pip install -r requirements.txt")


def to_wav(source, target):
    """Convert any audio file (e.g. a phone .m4a) to a 16 kHz mono WAV."""
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), str(target)], check=True)


def convert_m4a(raw_dir=RAW, out_dir=PROCESSED):
    """Convert new or replaced recordings in audio/raw/ to WAV in audio/processed/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for path in sorted(raw_dir.glob("*.m4a")):
        target = out_dir / f"{path.stem.lower()}.wav"
        if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
            continue
        to_wav(path, target)
        converted.append(target.name)
    return converted


def read_audio(path):
    """Load a clip and trim the silence at both ends."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    trimmed, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    return trimmed


def normalize(y):
    """Scale the clip to the same peak volume, so phone distance doesn't matter."""
    peak = np.abs(y).max()
    return y if peak == 0 else (y * (TARGET_PEAK / peak)).astype(np.float32)


def load_clip(path):
    return normalize(read_audio(path))


# 2. Collection check
def team_clips(folder=PROCESSED):
    return [p for p in sorted(folder.glob("*.wav")) if not p.name.startswith(UNAUTHORIZED_PREFIX)]


def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def unique_clips(folder=PROCESSED):
    """{clip: the clip it is an exact copy of, or None}. Copies would leak into both training and test data."""
    seen, result = {}, {}
    for path in team_clips(folder):
        h = file_hash(path)
        result[path] = seen.get(h)
        seen.setdefault(h, path.name)
    return result


def collection_report(folder=PROCESSED):
    """One row per clip: valid name, copy or not, length and peak."""
    rows = []
    copies = unique_clips(folder)
    for path in team_clips(folder):
        match = NAME_PATTERN.match(path.name)
        member = match.group(1) if match else None
        phrase = match.group(2) if match else None
        y = read_audio(path)
        duration_s = len(y) / SAMPLE_RATE
        peak = float(np.abs(y).max())
        ok = (match is not None and copies[path] is None
              and MIN_SECONDS < duration_s < MAX_SECONDS and peak < CLIP_PEAK)
        rows.append([path.name, member, phrase, match is not None, copies[path],
                     round(duration_s, 2), round(peak, 3), ok])
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def missing_phrases(report):
    """{member: [phrases they haven't recorded]}"""
    named = report[report["valid_name"].astype(bool)]
    have = named.groupby("member")["phrase"].apply(set)
    return {m: [p for p in PHRASES if p not in got] for m, got in have.items() if set(PHRASES) - got}


# 3. Augmentation
# Kept small on purpose: stronger settings made one member sound like another (see the notebook).
def pitch_shift(y, semitones):
    return librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=semitones)


def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)


def add_noise(y, snr_db, seed=0):
    """Add white noise snr_db decibels quieter than the voice."""
    noise = np.random.default_rng(seed).standard_normal(len(y))
    noise_power = np.mean(y**2) / 10 ** (snr_db / 10)
    return (y + noise * np.sqrt(noise_power)).astype(np.float32)


AUGMENTATIONS = {
    "pitch_up": lambda y: pitch_shift(y, 1),      # a cold or a different mood
    "pitch_down": lambda y: pitch_shift(y, -1),
    "faster": lambda y: time_stretch(y, 1.1),     # rushing, or speaking carefully
    "slower": lambda y: time_stretch(y, 0.9),
    "noisy": lambda y: add_noise(y, 40),          # background noise in the room
}


def augment(y):
    return {"original": y, **{name: fn(y) for name, fn in AUGMENTATIONS.items()}}


# 4. Feature extraction
def frame_features(y):
    """13 MFCCs, roll-off, centroid, zero-crossing rate and RMS energy for every frame."""
    return np.vstack([
        librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC + 1)[1:],  # skip MFCC 0 (loudness)
        librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, roll_percent=0.85),
        librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(y=y),
    ])


def summarize(frames):
    """Mean and standard deviation of every feature."""
    stats = np.column_stack([frames.mean(axis=1), frames.std(axis=1)]).ravel()
    return dict(zip(NUMERIC_COLUMNS, stats.astype(float)))


def extract_features(y):
    """One feature row per 0.5 s window of the clip."""
    frames = frame_features(y)
    hop = SAMPLE_RATE * 0.032  # librosa moves 512 samples (32 ms) between frames
    width, step = round(SEGMENT_SECONDS * SAMPLE_RATE / hop), round(SEGMENT_HOP * SAMPLE_RATE / hop)
    starts = range(0, max(1, frames.shape[1] - width + 1), step)
    return [summarize(frames[:, s:s + width]) for s in starts]


def build_features(folder=PROCESSED, augmented_dir=AUGMENTED):
    """Augment every team clip, save the augmented copies, and return all feature rows."""
    augmented_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, copy_of in unique_clips(folder).items():
        match = NAME_PATTERN.match(path.name)
        if not match or copy_of:
            continue
        member, phrase, _take = match.groups()
        for name, y in augment(load_clip(path)).items():
            if name != "original":
                sf.write(augmented_dir / f"{path.stem}_{name}.wav", y, SAMPLE_RATE)
            for i, values in enumerate(extract_features(y)):
                # source_clip keeps a recording and its augmented copies together when splitting train/test
                rows.append({"member": member, "phrase": phrase, "augmentation": name,
                             "source_clip": path.name, "segment": i, **values})
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def main():
    converted = convert_m4a()
    if converted:
        print(f"Converted {len(converted)} recordings to WAV: {', '.join(converted)}")
    report = collection_report()
    if report.empty:
        raise SystemExit(f"No team clips in {PROCESSED.relative_to(ROOT)}")
    print(report.to_string(index=False), "\n")
    if not report["ok"].all():
        print(f"Check these clips: {', '.join(report.loc[~report['ok'], 'file'])}")
    copies = report.dropna(subset=["duplicate_of"])
    for _, row in copies.iterrows():
        print(f"Skipped {row['file']}: an exact copy of {row['duplicate_of']} (record a new take instead)")
    for member, missing in missing_phrases(report).items():
        print(f"{member} is missing: {', '.join(missing)}")

    features = build_features()
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_FILE, index=False)
    print(f"Saved {len(features)} rows from {features['source_clip'].nunique()} recordings "
          f"to {FEATURES_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
