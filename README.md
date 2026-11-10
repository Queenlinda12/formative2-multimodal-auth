# Formative 2: Multimodal Authentication & Product Recommendation

A command-line system that checks a user's **face**, then their **voice**, and only after both checks pass shows a **product recommendation** based on their social media and purchase history.

```
Face image ──► Face recognition ──► Product model runs ──► Voice clip ──► Voice verification ──► Show recommended product
                    │ fail                                                      │ fail
                    ▼                                                           ▼
              Access denied                                               Access denied
```

> This README is a draft. Member C (Audio lead), please finish it once the face, voice and app parts are in. See `TEAM_GUIDE.md` for who does what.

**Repository:** https://github.com/Queenlinda12/formative2-multimodal-auth

## Team

| Member | Role | GitHub |
| --- | --- | --- |
| Sheja Queen Linda | Data lead: merge, cleaning, EDA, feature engineering | [@Queenlinda12](https://github.com/Queenlinda12) |
| Member B | Image lead: face images, image features, face recognition model | |
| Member C | Audio lead: voice samples, audio features, voiceprint model | |
| Member D | Product model and command-line app | |

## Folder layout

```
data/raw/          customer_social_profiles.xlsx, customer_transactions.xlsx (provided)
data/processed/    merged_dataset.csv, merged_dataset_model_ready.csv
images/raw/        name_neutral.jpg, name_smiling.jpg, name_surprised.jpg
images/augmented/  augmented face images
audio/raw/         name_yesapprove.wav, name_confirmtransaction.wav
audio/augmented/   augmented voice clips
features/          image_features.csv, audio_features.csv
scripts/           merge_pipeline.py (+ face, voice, product model scripts and main.py)
notebooks/         01_data_merge_eda.ipynb (+ the final combined notebook)
models/            saved model files
figures/           plots used in the report
report/            final PDF report
```

## Setup

```bash
git clone https://github.com/Queenlinda12/formative2-multimodal-auth.git
cd formative2-multimodal-auth
pip install -r requirements.txt
```

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

## Task 2 to 4: Images, audio, models and app

To be completed by Members B, C and D.
