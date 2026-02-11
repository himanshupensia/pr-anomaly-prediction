"""
src/training/train.py
─────────────────────────────────────────────────────────────────────────────
SAP BTP AI Core – PR Anomaly Detection  |  Training Entry-Point

Reads historical Purchase-Requisition data (CSV), engineers features, trains
an Isolation Forest anomaly-detection model, and serialises the artefact to
disk together with metadata (feature list, thresholds, training stats).

Environment variables (all optional – defaults shown):
    INPUT_DATA_PATH   – path to the input CSV   (default: /app/data/alldata.csv)
    OUTPUT_MODEL_PATH – path for model.joblib   (default: /app/models/model.joblib)
    OUTPUT_META_PATH  – path for metadata.json  (default: /app/models/metadata.json)
    CONTAMINATION     – expected anomaly fraction (default: 0.05)
    N_ESTIMATORS      – Isolation Forest trees   (default: 200)
    RANDOM_STATE      – reproducibility seed     (default: 42)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("pr-anomaly.train")

# ── Configuration (from environment, with defaults) ───────────────────────────
INPUT_DATA_PATH   = os.getenv("INPUT_DATA_PATH",   "../../data/alldata.csv")
OUTPUT_MODEL_PATH = os.getenv("OUTPUT_MODEL_PATH", "../../models/model.joblib")
OUTPUT_META_PATH  = os.getenv("OUTPUT_META_PATH",  "../../models/metadata.json")
CONTAMINATION     = float(os.getenv("CONTAMINATION", "0.05"))
N_ESTIMATORS      = int(os.getenv("N_ESTIMATORS",    "200"))
RANDOM_STATE      = int(os.getenv("RANDOM_STATE",    "42"))

# ── Expected CSV columns ──────────────────────────────────────────────────────
# Adjust column names to match your actual S/4HANA export.
REQUIRED_COLS = [
    # Header fields
    "pr_number",           # BANFN
    "pr_date",             # BADAT
    "pr_type",             # BSART
    "company_code",        # BUKRS
    "plant",               # WERKS
    "purchasing_group",    # EKGRP
    "created_by",          # ERNAM

    # Item fields
    "item_number",         # BNFPO
    "material_group",      # MATKL
    "quantity",            # MENGE
    "unit",                # MEINS
    "net_price",           # PREIS
    "currency",            # WAERS
    "delivery_date",       # LFDAT

    # Cost assignment
    "gl_account",          # SAKNR
    "cost_center",         # KOSTL
    "wbs_element",         # POSID
    "order_number",        # AUFNR
    "profit_center",       # PRCTR

    # Text
    "short_text",          # TXZ01 – item short text
    "header_text",         # TXTHD – header text (may be blank)
]

# Columns used as model features (subset of REQUIRED_COLS, all numeric/encoded)
CATEGORICAL_FEATURES = [
    "pr_type", "company_code", "plant", "purchasing_group",
    "material_group", "unit", "currency",
    "gl_account", "cost_center", "wbs_element", "profit_center",
]

NUMERIC_FEATURES = [
    "quantity", "net_price",
    "day_of_week",      # engineered
    "month",            # engineered
    "text_length",      # engineered – len(short_text)
]


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Derive numeric/encoded features from raw PR data.
    Returns the feature matrix and a dict of fitted LabelEncoders
    (needed at serving time to encode incoming requests consistently).
    """
    df = df.copy()

    # Date-derived features
    df["pr_date"] = pd.to_datetime(df["pr_date"], errors="coerce")
    df["day_of_week"] = df["pr_date"].dt.dayofweek.fillna(-1).astype(int)
    df["month"]       = df["pr_date"].dt.month.fillna(-1).astype(int)

    # Text-length feature
    df["text_length"] = df["short_text"].fillna("").str.len()

    # Encode categoricals
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = df[col].fillna("__UNKNOWN__").astype(str)
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = [c + "_enc" for c in CATEGORICAL_FEATURES] + NUMERIC_FEATURES
    return df[feature_cols], encoders


# ── Training ──────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    log.info("Loading data from %s", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def train(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, LabelEncoder]]:
    """
    Engineers features and fits an Isolation Forest.
    Returns (artefact_dict, encoders) – artefact_dict is what gets serialised.
    """
    log.info("Engineering features …")
    X, encoders = engineer_features(df)

    feature_cols = list(X.columns)
    log.info("Training on %d samples, %d features", len(X), len(feature_cols))

    log.info(
        "Fitting IsolationForest  contamination=%.3f  n_estimators=%d  random_state=%d",
        CONTAMINATION, N_ESTIMATORS, RANDOM_STATE,
    )
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)

    # Decision scores: negative = more anomalous
    scores = model.decision_function(X)
    threshold = float(np.percentile(scores, CONTAMINATION * 100))
    log.info("Anomaly score threshold (%.0f%% percentile): %.6f", CONTAMINATION * 100, threshold)

    artefact = {
        "model":        model,
        "encoders":     encoders,
        "feature_cols": feature_cols,
        "threshold":    threshold,
        "training_stats": {
            "n_samples":        int(len(X)),
            "n_features":       int(len(feature_cols)),
            "contamination":    CONTAMINATION,
            "n_estimators":     N_ESTIMATORS,
            "score_mean":       float(np.mean(scores)),
            "score_std":        float(np.std(scores)),
            "score_min":        float(np.min(scores)),
            "score_max":        float(np.max(scores)),
            "trained_at":       datetime.now(timezone.utc).isoformat(),
        },
    }

    return artefact, encoders


def save_artefacts(artefact: dict[str, Any]) -> None:
    model_path = Path(OUTPUT_MODEL_PATH)
    meta_path  = Path(OUTPUT_META_PATH)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Saving model artefact → %s", model_path)
    joblib.dump(artefact, model_path, compress=3)

    metadata = {
        "model_type":     "IsolationForest",
        "feature_cols":   artefact["feature_cols"],
        "threshold":      artefact["threshold"],
        "training_stats": artefact["training_stats"],
        "schema_version": "1.0",
    }
    log.info("Saving metadata → %s", meta_path)
    meta_path.write_text(json.dumps(metadata, indent=2))

    log.info("✔  Training complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== PR Anomaly Detection – Training Job ===")
    df       = load_data(INPUT_DATA_PATH)
    artefact, _ = train(df)
    save_artefacts(artefact)


if __name__ == "__main__":
    main()
