"""
Task 3 - Sound Data Collection and Processing
=============================================
Author: Agnes Mbabazi (Audio lead), GitHub @agmbabazi

Checks every member's voice clips, augments them and saves one feature row per
half-second window of every clip, original and augmented:

    audio/processed/<member>_<phrase>[_<take>].wav         16 kHz mono copies of the phone recordings
    audio/augmented/<member>_<phrase>[_<take>]_<augmentation>.wav
    features/audio_features.csv   member, phrase, augmentation, source_clip, segment, mfcc_..., rolloff_..., rms_...

Run from the repository root:
    python scripts/audio_pipeline.py

Design decisions (explained in the notebook and report):
  * Phone recordings (.m4a) are converted once to 16 kHz mono WAV with ffmpeg, so every
    clip has the same format before any processing. The originals in audio/raw/ are never changed.
    ffmpeg comes with the imageio-ffmpeg pip package, so `pip install -r requirements.txt` is enough.
  * Silence at the start and end is trimmed (30 dB below the peak), because people paused
    for very different lengths before speaking (Aime's "yes approve" is 4.0 s raw but 1.25 s
    of speech). Without trimming, the model could learn the length of the pause, not the voice.
  * TODO (you): one line per augmentation on which real-life situation it imitates.
  * Features are measured per 0.5 s window (every 0.25 s), not once per clip. A whole-clip
    average mostly describes WHICH WORDS were said, so with one clip per phrase a model learned
    phrases, not people (cross-validated accuracy 23-35%, chance is 25%). Short windows give the
    model many small pieces of each voice; a clip's decision is the average over its windows.
  * `source_clip` records which original recording a row came from, so the voice model
    can keep a clip and its augmented copies on the same side of a train/test split.
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

SAMPLE_RATE = 16_000  # 16 kHz is plenty for speech and is what most voice models use
TRIM_DB = 30  # quieter than (loudest point - 30 dB) counts as silence; 10 cut into words, 60 kept the dead air
PHRASES = ["yesapprove", "confirmtransaction"]
# name_phrase.wav, or name_phrase_2.wav for extra takes of the same phrase
NAME_PATTERN = re.compile(r"^([a-z]+)_(yesapprove|confirmtransaction)(?:_(\d+))?\.wav$")
UNAUTHORIZED_PREFIX = "unauthorized"
REPORT_COLUMNS = ["file", "member", "phrase", "valid_name", "duplicate_of", "duration_s", "peak", "ok"]
MIN_SECONDS, MAX_SECONDS = 0.5, 5.0  # speech shorter or longer than this means a bad recording
CLIP_PEAK = 0.99  # a peak this close to 1.0 means the recording is distorted
TARGET_PEAK = 0.9  # every clip is scaled to this loudest point before augmenting and extracting features
N_MFCC = 13  # coefficients 1-13; coefficient 0 is overall loudness, which rms already measures
SEGMENT_SECONDS, SEGMENT_HOP = 0.5, 0.25  # window length and step for the feature rows
SUMMARY_FEATURES = ["rolloff", "centroid", "zcr", "rms"]
FEATURE_NAMES = [f"mfcc_{i:02d}" for i in range(1, N_MFCC + 1)] + SUMMARY_FEATURES
NUMERIC_COLUMNS = [f"{name}_{stat}" for name in FEATURE_NAMES for stat in ("mean", "std")]
FEATURE_COLUMNS = ["member", "phrase", "augmentation", "source_clip", "segment", *NUMERIC_COLUMNS]


# ------------------------------------------------------------- 1. reading
def ffmpeg_exe():
    """The ffmpeg program: a system install if there is one, otherwise the copy bundled with imageio-ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise SystemExit("ffmpeg not found. Run: pip install -r requirements.txt (it installs imageio-ffmpeg)")


def to_wav(source, target):
    """Convert any audio file (e.g. a phone .m4a) to a 16 kHz mono WAV."""
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), str(target)], check=True)


def convert_m4a(raw_dir=RAW, out_dir=PROCESSED):
    """Convert every phone recording in raw_dir to a 16 kHz mono WAV in out_dir. Returns the new file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for path in sorted(raw_dir.glob("*.m4a")):
        target = out_dir / f"{path.stem.lower()}.wav"
        if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
            continue  # already converted, and the recording hasn't been replaced since
        to_wav(path, target)
        converted.append(target.name)
    return converted


def read_audio(path):
    """Load a clip as a 1-D float array at SAMPLE_RATE, with the silence at both ends trimmed."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    trimmed, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    return trimmed


def normalize(y):
    """Scale a clip so its loudest point is TARGET_PEAK. Peaks ranged 0.16-0.87, mostly from phone distance."""
    peak = np.abs(y).max()
    return y if peak == 0 else (y * (TARGET_PEAK / peak)).astype(np.float32)


def load_clip(path):
    """What every later step uses: the trimmed clip at a standard volume."""
    return normalize(read_audio(path))


# --------------------------------------------------- 2. collection check
def team_clips(folder=PROCESSED):
    """Every team member's WAV clip (the unauthorized ones are left out)."""
    return [p for p in sorted(folder.glob("*.wav")) if not p.name.startswith(UNAUTHORIZED_PREFIX)]


def file_hash(path):
    """A fingerprint of the file's bytes: two files with the same hash are the same recording, whatever their names."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def unique_clips(folder=PROCESSED):
    """{clip: the earlier clip it is an exact copy of, or None}. A renamed copy adds no new information
    and, worse, would let the same recording sit in both training and test data (fake 100% accuracy)."""
    seen, result = {}, {}
    for path in team_clips(folder):
        h = file_hash(path)
        result[path] = seen.get(h)
        seen.setdefault(h, path.name)
    return result


def collection_report(folder=PROCESSED):
    """One row per clip: is it named right, is it a copy, how long is it, how loud is it."""
    rows = []
    copies = unique_clips(folder)
    for path in team_clips(folder):
        match = NAME_PATTERN.match(path.name)
        member = match.group(1) if match else None
        phrase = match.group(2) if match else None
        y = read_audio(path)
        duration_s = len(y) / SAMPLE_RATE  # length of the speech after trimming
        peak = float(np.abs(y).max())  # loudest sample; 1.0 means the mic clipped (distorted)
        ok = (match is not None and copies[path] is None
              and MIN_SECONDS < duration_s < MAX_SECONDS and peak < CLIP_PEAK)
        rows.append([path.name, member, phrase, match is not None, copies[path],
                     round(duration_s, 2), round(peak, 3), ok])
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def missing_phrases(report):
    """{member: [phrases still missing]} for every member with a gap."""
    named = report[report["valid_name"].astype(bool)]
    have = named.groupby("member")["phrase"].apply(set)
    return {m: [p for p in PHRASES if p not in got] for m, got in have.items() if set(PHRASES) - got}


# ---------------------------------------------------------- 3. augment
# Strengths were picked by measuring how far each one moves a clip's average MFCCs, compared
# with the gap between two different members (min 27.8). Every setting below stays under ~12,
# so an augmented clip still "sounds like" its speaker; +/-4 semitones (34) or 30 dB noise (32)
# would have moved a voice further than the distance to another team member.
def pitch_shift(y, semitones):
    return librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=semitones)


def time_stretch(y, rate):
    # rate > 1 is faster (shorter), rate < 1 is slower (longer); the pitch stays the same
    return librosa.effects.time_stretch(y, rate=rate)


def add_noise(y, snr_db, seed=0):
    """White noise at a given signal-to-noise ratio: 40 dB means the voice is 10,000x stronger than the noise."""
    noise = np.random.default_rng(seed).standard_normal(len(y))
    noise_power = np.mean(y**2) / 10 ** (snr_db / 10)
    return (y + noise * np.sqrt(noise_power)).astype(np.float32)


AUGMENTATIONS = {
    "pitch_up": lambda y: pitch_shift(y, 1),
    "pitch_down": lambda y: pitch_shift(y, -1),
    "faster": lambda y: time_stretch(y, 1.1),
    "slower": lambda y: time_stretch(y, 0.9),
    "noisy": lambda y: add_noise(y, 40),
}


def augment(y):
    """The original plus every augmentation, as {name: audio}."""
    return {"original": y, **{name: fn(y) for name, fn in AUGMENTATIONS.items()}}


# ------------------------------------------------- 4. extract and save
def frame_features(y):
    """Per-frame features, shape (17, n_frames): 13 MFCCs, roll-off, centroid, zero-crossing rate, energy.

    librosa cuts the clip into short overlapping frames (2048 samples = 128 ms, every 32 ms)
    and measures each frame separately.
    """
    return np.vstack([
        librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC + 1)[1:],   # shape of the voice (vocal tract)
        librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, roll_percent=0.85),  # Hz below which 85% of energy sits
        librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE),             # "centre of mass" of the spectrum: brightness
        librosa.feature.zero_crossing_rate(y),                              # how often the wave crosses 0: noisiness
        librosa.feature.rms(y=y),                                           # energy (loudness) of each frame
    ])


def summarize(frames):
    """Mean and standard deviation over time of every per-frame feature: mfcc_01_mean, mfcc_01_std, ..."""
    stats = np.column_stack([frames.mean(axis=1), frames.std(axis=1)]).ravel()
    return dict(zip(NUMERIC_COLUMNS, stats.astype(float)))


def extract_features(y):
    """One feature row per 0.5 s window of the clip (a clip shorter than one window gives one row)."""
    frames = frame_features(y)
    hop = SAMPLE_RATE * 0.032  # librosa's default step between frames is 512 samples = 32 ms
    width, step = round(SEGMENT_SECONDS * SAMPLE_RATE / hop), round(SEGMENT_HOP * SAMPLE_RATE / hop)
    starts = range(0, max(1, frames.shape[1] - width + 1), step)
    return [summarize(frames[:, s:s + width]) for s in starts]


def build_features(folder=PROCESSED, augmented_dir=AUGMENTED):
    """Augments every team clip, saves the copies and returns one feature row per window of every version."""
    augmented_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, copy_of in unique_clips(folder).items():
        match = NAME_PATTERN.match(path.name)
        if not match or copy_of:  # badly named, or an exact copy of a clip already used
            continue
        member, phrase, _take = match.groups()
        for name, y in augment(load_clip(path)).items():
            if name != "original":
                sf.write(augmented_dir / f"{path.stem}_{name}.wav", y, SAMPLE_RATE)
            for i, values in enumerate(extract_features(y)):
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
    print(f"Saved {len(features)} rows ({features['source_clip'].nunique()} clips x {len(AUGMENTATIONS) + 1} "
          f"versions x about {len(features) // len(features.groupby(['source_clip', 'augmentation']))} windows, "
          f"{len(NUMERIC_COLUMNS)} features each) to {FEATURES_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
