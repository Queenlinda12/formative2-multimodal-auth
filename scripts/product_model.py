"""
Task 4 - Product Recommendation Model
======================================
Trains a product model on data/processed/merged_dataset_model_ready.csv,
evaluates it, and saves:

    models/product_model.joblib

Run from the repository root, after scripts/merge_pipeline.py:
    python scripts/product_model.py

Use it from the app (scripts/main.py):
    from product_model import predict_product
    predict_product(customer_id=151)          # -> "Sports", "Electronics", ...
    predict_product(transaction_id=1001)

Design decisions:
  * Random Forest on the engineered columns. The label is target_code
    (0 Books, 1 Clothing, 2 Electronics, 3 Groceries, 4 Sports).
    transaction_id, customer_id and product_category are not features:
    the ids are not behaviour, and product_category is the answer in words.
  * GroupKFold grouped by customer_id. The same customer has several
    transactions, and the spending aggregates are shared across those rows,
    so a random split would test the model on a customer it just trained on.
  * The pattern in this table is weak (a majority guess is already in the
    mid-20%s). The held-out accuracy, macro F1 and log loss are the scores
    that matter; a prediction on a saved row will look better because that
    row was in the final fit.
"""

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from merge_pipeline import OUT, ROOT

DATA_FILE = OUT / "merged_dataset_model_ready.csv"
MODEL_FILE = ROOT / "models" / "product_model.joblib"
# Ids are not behaviour. product_category is the label written out as text.
DROP_COLUMNS = ["transaction_id", "customer_id", "product_category", "target_code"]
N_SPLITS = 5
RANDOM_STATE = 42


# ----------------------------------------------------------- 1. model
def feature_columns(df):
    return [c for c in df.columns if c not in DROP_COLUMNS]


def class_mapping(df):
    """target_code -> product name, taken from the merged table."""
    pairs = df.drop_duplicates("target_code").set_index("target_code")["product_category"].sort_index()
    return {int(code): str(name) for code, name in pairs.items()}


def make_classifier():
    # class_weight keeps the smaller categories (Books, Groceries) from being ignored.
    # Deeper trees and a leaf-size penalty did not beat this under GroupKFold.
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def fit(df):
    columns = feature_columns(df)
    X, y = df[columns].to_numpy(dtype=float), df["target_code"].to_numpy(dtype=int)
    return {
        "classifier": make_classifier().fit(X, y),
        "feature_columns": columns,
        "classes": class_mapping(df),
    }


def align_proba(classifier, X, classes):
    """predict_proba columns lined up with `classes`, even if a fold missed one."""
    proba = classifier.predict_proba(X)
    full = np.zeros((len(X), len(classes)))
    index = {int(label): i for i, label in enumerate(classes)}
    for col, label in enumerate(classifier.classes_):
        full[:, index[int(label)]] = proba[:, col]
    return full


# ------------------------------------------------------ 2. evaluation
def cross_validate(df, n_splits=N_SPLITS):
    """One held-out prediction per row. A customer's transactions stay in a single fold."""
    columns = feature_columns(df)
    X = df[columns].to_numpy(dtype=float)
    y = df["target_code"].to_numpy(dtype=int)
    groups = df["customer_id"].to_numpy()
    classes = np.array(sorted(class_mapping(df)))
    n_splits = min(n_splits, df["customer_id"].nunique())
    splitter = GroupKFold(n_splits=n_splits)
    predicted = np.empty(len(df), dtype=int)
    probabilities = np.zeros((len(df), len(classes)))
    for train, test in splitter.split(X, y, groups):
        classifier = make_classifier().fit(X[train], y[train])
        predicted[test] = classifier.predict(X[test]).astype(int)
        probabilities[test] = align_proba(classifier, X[test], classes)
    return predicted, probabilities, classes, n_splits


def metrics(y, predicted, probabilities, classes):
    return {
        "accuracy": accuracy_score(y, predicted),
        "f1_macro": f1_score(y, predicted, average="macro", labels=classes, zero_division=0),
        "log_loss": log_loss(y, probabilities, labels=classes),
    }


def majority_accuracy(y):
    """Always predicting the most common product."""
    mode = np.bincount(y).argmax()
    return float(np.mean(y == mode))


def base_rate_log_loss(y, classes):
    """Log loss of predicting the overall class frequencies on every row."""
    prior = np.array([(y == code).mean() for code in classes])
    return log_loss(y, np.tile(prior, (len(y), 1)), labels=classes)


# ------------------------------------------------------- 3. for the app
def load_table():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Run: python scripts/merge_pipeline.py")
    return pd.read_csv(DATA_FILE)


@lru_cache(maxsize=1)
def _load():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"{MODEL_FILE} not found. Run: python scripts/product_model.py")
    return joblib.load(MODEL_FILE), load_table()


def resolve_row(table, features, customer_id, transaction_id):
    """The feature row to score, plus the customer and transaction it came from."""
    if features is not None:
        row = features if isinstance(features, pd.Series) else pd.Series(features)
        return row, _as_int(row.get("customer_id")), _as_int(row.get("transaction_id"))
    if transaction_id is not None:
        hits = table[table["transaction_id"] == int(transaction_id)]
        if hits.empty:
            raise ValueError(f"No transaction with transaction_id {transaction_id}.")
        row = hits.iloc[0]
        return row, int(row["customer_id"]), int(transaction_id)
    if customer_id is not None:
        hits = table[table["customer_id"] == int(customer_id)]
        if hits.empty:
            raise ValueError(f"No customer with customer_id {customer_id}.")
        # Latest visit: transaction ids increase with time in the source file.
        row = hits.sort_values("transaction_id").iloc[-1]
        return row, int(customer_id), int(row["transaction_id"])
    raise ValueError("Pass features, a customer_id, or a transaction_id.")


def _as_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    return int(value)


def product_details(features=None, customer_id=None, transaction_id=None):
    """Product name, code, and class probabilities for one customer visit."""
    model, table = _load()
    row, customer, transaction = resolve_row(table, features, customer_id, transaction_id)
    columns = model["feature_columns"]
    missing = [c for c in columns if c not in row.index or pd.isna(row[c])]
    if missing:
        raise ValueError("Missing features: " + ", ".join(missing))
    X = row[columns].to_numpy(dtype=float).reshape(1, -1)
    probabilities = align_proba(model["classifier"], X, sorted(model["classes"]))[0]
    codes = sorted(model["classes"])
    best = int(codes[int(np.argmax(probabilities))])
    return {
        "product": model["classes"][best],
        "target_code": best,
        "probabilities": {model["classes"][code]: float(probabilities[i]) for i, code in enumerate(codes)},
        "customer_id": customer,
        "transaction_id": transaction,
    }


def predict_product(features=None, customer_id=None, transaction_id=None):
    """The product category name, for example "Electronics"."""
    return product_details(features, customer_id, transaction_id)["product"]


def default_customer_id(table=None):
    """A customer with several visits, so the demo is not a one-row profile."""
    table = load_table() if table is None else table
    return int(table["customer_id"].value_counts().index[0])


def main():
    df = load_table()
    y = df["target_code"].to_numpy(dtype=int)
    mapping = class_mapping(df)
    columns = feature_columns(df)
    print(f"{len(df)} transactions, {df['customer_id'].nunique()} customers, "
          f"{len(mapping)} products, {len(columns)} features")
    print("Held out of the features: " + ", ".join(DROP_COLUMNS[:3]) + ". Label: target_code.")
    print("Class counts: " + ", ".join(f"{mapping[code]} {int((y == code).sum())}" for code in sorted(mapping)))

    predicted, probabilities, classes, n_splits = cross_validate(df)
    print(f"\nGroupKFold by customer_id ({n_splits} folds):")
    for name, value in metrics(y, predicted, probabilities, classes).items():
        print(f"  {name:<12} {value:.3f}")
    print(f"  {'majority':<12} {majority_accuracy(y):.3f}  (always predict the most common product)")
    print(f"  {'base rate':<12} {base_rate_log_loss(y, classes):.3f}  (log loss of predicting the class frequencies)")

    names = [mapping[int(code)] for code in classes]
    print("\nConfusion matrix (rows = true product):")
    print(pd.DataFrame(confusion_matrix(y, predicted, labels=classes), index=names, columns=names))

    model = fit(df)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    _load.cache_clear()
    print(f"\nSaved {MODEL_FILE.relative_to(ROOT)} (trained on all {len(df)} rows)")

    customer = default_customer_id(df)
    details = product_details(customer_id=customer)
    print(f"Example: customer {details['customer_id']}, transaction {details['transaction_id']} "
          f"-> {details['product']}")
    print("The example row was included in the saved model. The scores above are the held-out ones.")


if __name__ == "__main__":
    main()
