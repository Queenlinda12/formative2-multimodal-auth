# Formative 2: Multimodal Authentication & Product Recommendation

A command-line system that checks a user's **face**, then their **voice**, and only after both checks pass shows a **product recommendation** based on their social media and purchase history.

```
Face image ──► Face recognition ──► Product model runs ──► Voice clip ──► Voice verification ──► Show recommended product
                    │ fail                                                      │ fail
                    ▼                                                           ▼
              Access denied                                               Access denied
```

> Status: data, image and audio parts are complete. The product model and command-line app (Task 4, Member D) are in progress. See `TEAM_GUIDE.md` for who does what.

**Repository:** https://github.com/Queenlinda12/formative2-multimodal-auth

## Team

| Member | Role | GitHub |
| --- | --- | --- |
| Sheja Queen Linda | Data lead: merge, cleaning, EDA, feature engineering | [@Queenlinda12](https://github.com/Queenlinda12) |
| Aime Igirimpuhwe | Image lead: face images, image features, face recognition model | [@igaimerca](https://github.com/igaimerca) |
| Agnes Mbabazi | Audio lead: voice samples, audio features, voiceprint model | [@agmbabazi](https://github.com/agmbabazi) |
| Member D | Product model and command-line app | |

## Folder layout

```
data/raw/          customer_social_profiles.xlsx, customer_transactions.xlsx (provided)
data/processed/    merged_dataset.csv, merged_dataset_model_ready.csv
images/raw/        name_neutral.jpg, name_smiling.jpg, name_surprised.jpg
images/augmented/  augmented face images
audio/raw/         phone recordings: name_yesapprove[_take].m4a, name_confirmtransaction[_take].m4a
audio/processed/   the same recordings as 16 kHz mono WAV (made by audio_pipeline.py)
audio/augmented/   augmented voice clips
features/          image_features.csv, audio_features.csv
scripts/           merge_pipeline.py (+ face, voice, product model scripts and main.py)
notebooks/         01_data_merge_eda.ipynb, 02_image_processing_face_model.ipynb,
                   03_audio_processing_voice_model.ipynb (+ the final combined notebook)
models/            saved model files
figures/           plots used in the report
report/            final PDF report
```

## Setup

```bash
git clone https://github.com/Queenlinda12/formative2-multimodal-auth.git
cd formative2-multimodal-auth
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install ffmpeg        # converts phone .m4a recordings (macOS: brew install ffmpeg)
```

### Reproduce everything, in order

```bash
python scripts/merge_pipeline.py      # Task 1: merged dataset
python scripts/image_pipeline.py      # Task 2: image features
python scripts/face_model.py          #         face model
python scripts/audio_pipeline.py      # Task 3: audio features
python scripts/voice_model.py         #         voice model
```

Every output (`data/processed/`, `features/`, `models/`, `audio/processed/`, augmented images and clips) is rebuilt from the raw files by these scripts, so deleting an output and re-running brings it back.

## Task 1: Data merge (Sheja Queen Linda)

```bash
python scripts/merge_pipeline.py                     # rebuilds data/processed/*.csv
jupyter notebook notebooks/01_data_merge_eda.ipynb   # step-by-step walkthrough with checks and plots
```

- The two files use different ID formats (`A151` vs `151`), so both were standardised to an integer `customer_id`.
- Social profiles has several rows per customer (one per platform), so it is summarised to one row per customer before joining. This avoids duplicating purchases.
- An **inner join** keeps only customers who appear in both files: 117 transactions from 61 customers, with no missing values.
- 5 duplicate rows were removed, and 10 missing ratings were filled with the median, with a `rating_missing` flag added.
- EDA: 5 labelled plots (target balance, distributions, outliers, correlations, features vs target), saved in `figures/`.

**For the product model:** use `merged_dataset_model_ready.csv`. Drop `transaction_id`, `customer_id` and `product_category`, and use `target_code` as the label (0 Books, 1 Clothing, 2 Electronics, 3 Groceries, 4 Sports).

## Task 2 and the face model: Images (Aime Igirimpuhwe)

```bash
python scripts/image_pipeline.py    # checks photos, augments them, saves features/image_features.csv
python scripts/face_model.py        # evaluates and saves models/face_model.joblib
jupyter notebook notebooks/02_image_processing_face_model.ipynb   # step-by-step walkthrough with plots
```

- Photos go in `images/raw/` as `name_neutral.jpg`, `name_smiling.jpg`, `name_surprised.jpg`. iPhone `.heic` photos are converted to `.jpg` automatically, upright, and every photo's metadata (including GPS location) is removed before it's committed, since this repo is public.
- Each photo gets 5 augmentations (rotated, flipped, grayscale, brightened, blurred), saved in `images/augmented/`.
- Features are 128-number face embeddings from OpenCV's pretrained SFace model, after the YuNet detector finds and aligns the face (`models/pretrained/`).
- The face model is a Logistic Regression, evaluated so that each photo and its augmented copies are always held out together.
- `predict_face(image_path)` in `scripts/face_model.py` returns the member's name, or `"unknown"` when the face is not close enough to any member (cosine similarity below 0.363), when there is no face or more than one, or when the file can't be read.
- `images/raw/unauthorized_face.jpg` (a public-domain NASA portrait, from scikit-image) is the stranger for the demo.

## Task 3 and the voice model: Audio (Agnes Mbabazi)

```bash
python scripts/audio_pipeline.py    # converts, checks and augments clips, saves features/audio_features.csv
python scripts/voice_model.py       # evaluates, tunes the threshold and saves models/voice_model.joblib
jupyter notebook notebooks/03_audio_processing_voice_model.ipynb   # step-by-step walkthrough with plots
```

**Adding recordings.** Put phone recordings in `audio/raw/` named `name_yesapprove.m4a` and `name_confirmtransaction.m4a`, with extra takes as `name_yesapprove_2.m4a`, `_3`, and so on (first name, lowercase). Then re-run the two scripts. Each take must be a **new recording**: exact copies are detected by file hash and skipped, because a copy adds no information and leaks the same recording into both training and test data.

- **Preprocessing:** every `.m4a` is converted with ffmpeg to 16 kHz mono WAV in `audio/processed/`. Silence more than 30 dB below the peak is trimmed from both ends, and every clip is scaled to the same peak volume, so the model sees the voice, not the pause length or the phone's distance. A collection check flags bad names, clips outside 0.5 to 5 s, clipping and copies.
- **Augmentation:** 5 per recording (pitch ±1 semitone, 1.1× faster, 0.9× slower, background noise at 40 dB SNR), saved in `audio/augmented/`. Each strength was measured to move a voice less than the smallest gap between two members. Stronger settings turned one member into "someone else".
- **Features:** 13 MFCCs, spectral roll-off, spectral centroid, zero-crossing rate and RMS energy, as mean and standard deviation over each 0.5 s window (34 numbers per row), saved in `features/audio_features.csv` with `member`, `phrase`, `augmentation`, `source_clip` and `segment` columns.
- **Model:** two Logistic Regressions, one for *who* is speaking and one for *which* approved phrase. A clip's answer is the average over its windows. Evaluation holds out whole recordings with their augmented copies, and reports accuracy, macro F1 and log loss.
- **Rejecting strangers:** a voice is accepted only if the speaker model is sure enough and the phrase is clear (≥ 0.6). The speaker threshold is chosen on every run at the **Equal Error Rate**, where the False Acceptance Rate (each member left out of training in turn) equals the False Rejection Rate (held-out recordings). `scripts/voice_model.py` prints both, and the notebook plots the curve.
- `verify_voice(audio_path)` in `scripts/voice_model.py` accepts a WAV or phone `.m4a` file and returns the member's name, or `"unauthorized"` when the voice or phrase is not recognised confidently, or the file can't be read. The app must check it is the **same member** that `predict_face` returned. Note that the face model says `"unknown"` and the voice model says `"unauthorized"`.
- `audio/raw/unauthorized_*.m4a` (a speaker outside the team, saying both phrases) is the stranger for the demo. It is never used for training.
- **Limitations:** the main one is the amount of data (2 to 6 recordings per member). Results improved sharply with each real extra take. The phrase check can tell the two approved phrases apart but has never heard a wrong phrase. Background noise (Sheja's room hum, for example) can be learned as part of a voice.

## Task 4: Product model and command-line app

To be completed by Member D.
