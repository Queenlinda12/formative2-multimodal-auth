"""
Task 3 - Sound Data Collection and Processing
=============================================
Author: Agnes Mbabazi (Audio lead)

Checks every member's voice clips, augments them and saves one feature row per
clip, original and augmented:

    audio/processed/<member>_<phrase>.wav                  16 kHz mono copies of the phone recordings
    audio/augmented/<member>_<phrase>_<augmentation>.wav
    features/audio_features.csv   member, phrase, augmentation, source_clip, mfcc_..., rolloff_..., rms_...

Run from the repository root:
    python scripts/audio_pipeline.py

Design decisions (explained in the notebook and report):
  * Phone recordings (.m4a) are converted once to 16 kHz mono WAV with ffmpeg, so every
    clip has the same format before any processing. The originals in audio/raw/ are never changed.
  * Silence at the start and end is trimmed (30 dB below the peak), because people paused
    for very different lengths before speaking (Aime's "yes approve" is 4.0 s raw but 1.25 s
    of speech). Without trimming, the model could learn the length of the pause, not the voice.
  * TODO (you): one line per augmentation on which real-life situation it imitates.
  * `source_clip` records which original recording a row came from, so the voice model
    can keep a clip and its augmented copies on the same side of a train/test split.
"""

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
NAME_PATTERN = re.compile(r"^([a-z]+)_(yesapprove|confirmtransaction)\.wav$")
UNAUTHORIZED_PREFIX = "unauthorized"
REPORT_COLUMNS = ["file", "member", "phrase", "valid_name", "duration_s", "peak", "ok"]


# ------------------------------------------------------------- 1. reading
def convert_m4a(raw_dir=RAW, out_dir=PROCESSED):
    """Convert every phone recording in raw_dir to a 16 kHz mono WAV in out_dir. Returns the new file names."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Run: sudo apt install ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for path in sorted(raw_dir.glob("*.m4a")):
        target = out_dir / f"{path.stem.lower()}.wav"
        if target.exists():
            continue
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(path),
             "-ac", "1", "-ar", str(SAMPLE_RATE), str(target)],
            check=True,
        )
        converted.append(target.name)
    return converted


def read_audio(path):
    """Load a clip as a 1-D float array at SAMPLE_RATE, with the silence at both ends trimmed."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    trimmed, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    return trimmed


# --------------------------------------------------- 2. collection check
def team_clips(folder=PROCESSED):
    """Every team member's WAV clip (the unauthorized ones are left out)."""
    return [p for p in sorted(folder.glob("*.wav")) if not p.name.startswith(UNAUTHORIZED_PREFIX)]


def collection_report(folder=PROCESSED):
    """One row per clip: is it named right, how long is it, how loud is it."""
    rows = []
    for path in team_clips(folder):
        # TODO 2:
        #   - match = NAME_PATTERN.match(path.name) -> member = match.group(1), phrase = match.group(2)
        #     (if there's no match, valid_name is False and member/phrase are None)
        #   - y = read_audio(path)
        #   - duration_s = len(y) / SAMPLE_RATE
        #   - peak = np.abs(y).max()   (1.0 means the recording clipped / distorted)
        #   - ok = valid name AND 0.5 < duration_s < 5 AND peak < 0.99
        #   - rows.append([...]) in the same order as REPORT_COLUMNS
        pass
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def missing_phrases(report):
    """{member: [phrases they have not recorded]}"""
    # TODO 3: look at B's missing_expressions() in image_pipeline.py and do the same with PHRASES
    return {}


# ---------------------------------------------------------- 3. augment
# (next step: pitch_shift, time_stretch, add_noise, AUGMENTATIONS dict)


# ------------------------------------------------- 4. extract and save
# (step after that: extract_features, build_features)


def main():
    converted = convert_m4a()
    if converted:
        print(f"Converted {len(converted)} recordings to WAV: {', '.join(converted)}")
    report = collection_report()
    if report.empty:
        raise SystemExit(f"No team clips in {PROCESSED.relative_to(ROOT)}")
    print(report.to_string(index=False), "\n")
    for member, missing in missing_phrases(report).items():
        print(f"{member} is missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
