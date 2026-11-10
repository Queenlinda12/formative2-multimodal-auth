"""
Task 1 - Data Merge
===================
Author: Sheja Queen Linda (Data lead), GitHub @Queenlinda12

Cleans customer_social_profiles and customer_transactions, merges them,
engineers features and saves:

    data/processed/merged_dataset.csv              readable, one row per transaction
    data/processed/merged_dataset_model_ready.csv  encoded + scaled, for the product model

Run from the repository root:
    python scripts/merge_pipeline.py

Design decisions (explained in the notebook and report):
  * The two tables use different ID formats: "A151" (social) vs 151 (transactions).
    Both are standardised to an integer `customer_id`.
  * Social profiles has several rows per customer (one per platform). Joining it
    directly to transactions (also several rows per customer) would be a
    many-to-many join that duplicates purchases. So social profiles are first
    AGGREGATED to one row per customer, then joined (many-to-one).
  * INNER join: the model predicts what a customer buys from their social media
    behaviour, so a transaction without any social profile has no social
    features to learn from. Filling them in would invent data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

SOCIAL_FILE = RAW / "customer_social_profiles.xlsx"
TRANS_FILE = RAW / "customer_transactions.xlsx"

PLATFORMS = ["Facebook", "Instagram", "LinkedIn", "TikTok", "Twitter"]
SENTIMENTS = ["Negative", "Neutral", "Positive"]
SENTIMENT_SCORE = {"Negative": -1, "Neutral": 0, "Positive": 1}
TARGET = "product_category"


# ---------------------------------------------------------------- 1. load
def load_raw():
    """Read the two provided Excel files exactly as given."""
    social = pd.read_excel(SOCIAL_FILE)
    trans = pd.read_excel(TRANS_FILE)
    return social, trans


def profile(df, name):
    """Small quality report used before and after cleaning."""
    return pd.DataFrame({
        "dataset": name,
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "nulls": df.isna().sum().values,
        "unique": df.nunique().values,
    })


# --------------------------------------------------------------- 2. clean
def clean_social(social):
    s = social.copy()
    s = s.drop_duplicates()                                   # 5 exact duplicate rows
    s["customer_id"] = s["customer_id_new"].str.strip().str.upper().str.lstrip("A").astype(int)
    s["social_media_platform"] = s["social_media_platform"].str.strip().astype("category")
    s["review_sentiment"] = s["review_sentiment"].str.strip().astype("category")
    s["engagement_score"] = s["engagement_score"].astype(int)
    s["purchase_interest_score"] = s["purchase_interest_score"].astype(float)
    return s.drop(columns="customer_id_new")


def clean_transactions(trans):
    t = trans.copy()
    t = t.drop_duplicates()
    t = t.rename(columns={"customer_id_legacy": "customer_id"})
    t["customer_id"] = t["customer_id"].astype(int)
    t["purchase_date"] = pd.to_datetime(t["purchase_date"])
    t[TARGET] = t[TARGET].str.strip()
    # customer_rating has 10 nulls. Keep the rows (they still have a valid
    # target), record that the rating was missing, and fill with the overall
    # median. The median by product category is NOT used because that would
    # leak the target into a feature.
    t["rating_missing"] = t["customer_rating"].isna().astype(int)
    t["customer_rating"] = t["customer_rating"].fillna(t["customer_rating"].median())
    return t


# ----------------------------------------------------- 3. aggregate social
def aggregate_social(s):
    """One row per customer, so the join with transactions is many-to-one."""
    num = s.groupby("customer_id").agg(
        n_social_profiles=("social_media_platform", "size"),
        n_platforms=("social_media_platform", "nunique"),
        avg_engagement=("engagement_score", "mean"),
        max_engagement=("engagement_score", "max"),
        avg_purchase_interest=("purchase_interest_score", "mean"),
        max_purchase_interest=("purchase_interest_score", "max"),
    )
    s = s.assign(sentiment_score=s["review_sentiment"].astype(str).map(SENTIMENT_SCORE))
    num["avg_sentiment_score"] = s.groupby("customer_id")["sentiment_score"].mean()

    # share of the customer's profiles that are positive / neutral / negative
    sent = pd.crosstab(s["customer_id"], s["review_sentiment"].astype(str), normalize="index")
    sent = sent.reindex(columns=SENTIMENTS, fill_value=0).add_prefix("share_").rename(columns=str.lower)

    # 1 if the customer is active on that platform
    plat = pd.crosstab(s["customer_id"], s["social_media_platform"].astype(str)).clip(upper=1)
    plat = plat.reindex(columns=PLATFORMS, fill_value=0).add_prefix("on_").rename(columns=str.lower)

    # platform where the customer is most engaged
    top = (s.sort_values("engagement_score", ascending=False)
             .groupby("customer_id")["social_media_platform"].first().astype(str)
             .rename("top_platform"))

    agg = num.join([sent, plat, top]).reset_index()
    return agg.round(4)


# ------------------------------------------------------------- 4. merge
def merge(trans_clean, social_agg):
    merged = trans_clean.merge(social_agg, on="customer_id", how="inner", validate="many_to_one")
    return merged.sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True)


def merge_checks(trans_clean, social_agg, merged):
    """Numbers shown in the notebook/report to prove the merge is correct."""
    t_ids, s_ids = set(trans_clean.customer_id), set(social_agg.customer_id)
    left = trans_clean.merge(social_agg, on="customer_id", how="left", indicator=True)
    return {
        "transactions_rows_before": len(trans_clean),
        "social_customers_after_aggregation": len(social_agg),
        "customers_in_both": len(t_ids & s_ids),
        "customers_only_in_transactions": len(t_ids - s_ids),
        "customers_only_in_social": len(s_ids - t_ids),
        "transactions_without_social_profile": int((left["_merge"] == "left_only").sum()),
        "merged_rows": len(merged),
        "merged_customers": merged.customer_id.nunique(),
        "transaction_id_unique_after_merge": bool(merged.transaction_id.is_unique),
        "nulls_after_merge": int(merged.isna().sum().sum()),
        "rows_gained_by_join": len(merged) - len(left[left["_merge"] == "both"]),
    }


# -------------------------------------------------- 5. feature engineering
def engineer(merged):
    df = merged.copy()
    # date features
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dayofweek"] = df["purchase_date"].dt.dayofweek
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)

    # per-customer spending aggregates (amounts only, never the target)
    g = df.groupby("customer_id")["purchase_amount"]
    df["customer_total_spend"] = g.transform("sum")
    df["customer_purchase_count"] = g.transform("count")
    df["customer_avg_spend"] = g.transform("mean").round(2)
    df["amount_vs_customer_avg"] = (df["purchase_amount"] / df["customer_avg_spend"]).round(4)

    # purchase history in time order
    df = df.sort_values(["customer_id", "purchase_date", "transaction_id"])
    df["purchase_number"] = df.groupby("customer_id").cumcount() + 1
    gap = df.groupby("customer_id")["purchase_date"].diff().dt.days
    df["days_since_prev_purchase"] = gap.fillna(-1).astype(int)   # -1 = first purchase
    df = df.sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True)

    # interaction: interest weighted by engagement
    df["engagement_x_interest"] = (df["avg_engagement"] * df["avg_purchase_interest"]).round(4)

    # readable target label + integer code
    classes = sorted(df[TARGET].unique())
    df["target_code"] = df[TARGET].map({c: i for i, c in enumerate(classes)})
    return df


ID_COLS = ["transaction_id", "customer_id", "purchase_date"]
CAT_FEATURES = ["top_platform"]
BINARY_FEATURES = ["rating_missing", "is_weekend"] + [f"on_{p.lower()}" for p in PLATFORMS]


def model_ready(df):
    """One-hot encode categories and standard-scale numeric features."""
    num_features = [c for c in df.columns
                    if c not in ID_COLS + CAT_FEATURES + BINARY_FEATURES + [TARGET, "target_code"]
                    and pd.api.types.is_numeric_dtype(df[c])]
    X = pd.get_dummies(df[CAT_FEATURES], prefix=CAT_FEATURES, dtype=int)
    scaled = pd.DataFrame(StandardScaler().fit_transform(df[num_features]),
                          columns=num_features, index=df.index).round(5)
    out = pd.concat([df[["transaction_id", "customer_id"]], scaled, df[BINARY_FEATURES], X,
                     df[[TARGET, "target_code"]]], axis=1)
    return out, num_features


def outlier_report(df, cols):
    """IQR rule: values outside Q1 - 1.5*IQR .. Q3 + 1.5*IQR."""
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"column": c, "lower_bound": round(lo, 2), "upper_bound": round(hi, 2),
                     "n_outliers": int(((df[c] < lo) | (df[c] > hi)).sum())})
    return pd.DataFrame(rows)


def run(save=True):
    social_raw, trans_raw = load_raw()
    social = clean_social(social_raw)
    trans = clean_transactions(trans_raw)
    social_agg = aggregate_social(social)
    merged = merge(trans, social_agg)
    checks = merge_checks(trans, social_agg, merged)
    final = engineer(merged)
    ready, _ = model_ready(final)
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        final.to_csv(OUT / "merged_dataset.csv", index=False)
        ready.to_csv(OUT / "merged_dataset_model_ready.csv", index=False)
    return final, ready, checks


if __name__ == "__main__":
    final, ready, checks = run()
    print("Merge checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nSaved merged_dataset.csv {final.shape} and merged_dataset_model_ready.csv {ready.shape} to {OUT}")
