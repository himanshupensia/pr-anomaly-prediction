"""
src/serving/app.py
─────────────────────────────────────────────────────────────────────────────
SAP BTP AI Core – PR Anomaly Detection  |  Serving / Inference API

Flask application that exposes the trained Isolation Forest model as a REST
endpoint conforming to the SAP AI Core serving specification.

Endpoints
─────────
GET  /v1/health          – liveness / readiness probe
POST /v1/models/pr-anomaly/predict  – single or batch PR scoring

SAP AI Core routes all inference traffic to  POST /v1/models/<model-name>/predict
The model name is configured in serve.yaml (resourcePlan → modelName).

Environment variables
─────────────────────
    MODEL_PATH – absolute path to model.joblib   (default: /app/models/model.joblib)
    PORT       – HTTP port                        (default: 8080)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("pr-anomaly.serve")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.joblib")


# ── Model loading (module-level singleton) ────────────────────────────────────
_artefact: dict[str, Any] | None = None


def _load_model() -> dict[str, Any]:
    global _artefact
    if _artefact is None:
        path = Path(MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        log.info("Loading model artefact from %s …", path)
        _artefact = joblib.load(path)
        log.info(
            "Model ready  |  features: %d  |  threshold: %.6f",
            len(_artefact["feature_cols"]),
            _artefact["threshold"],
        )
    return _artefact


# ── Feature engineering (mirrors train.py logic) ──────────────────────────────
CATEGORICAL_FEATURES = [
    "pr_type", "company_code", "plant", "purchasing_group",
    "material_group", "unit", "currency",
    "gl_account", "cost_center", "wbs_element", "profit_center",
]

NUMERIC_FEATURES = [
    "quantity", "net_price",
    "day_of_week",
    "month",
    "text_length",
]


def _engineer_row(row: dict[str, Any], artefact: dict[str, Any]) -> pd.DataFrame:
    """
    Transform a single PR item payload dict into a feature vector (1-row DataFrame)
    using the encoders fitted during training.
    Unknown categories are mapped to -1.
    """
    encoders     = artefact["encoders"]
    feature_cols = artefact["feature_cols"]

    record: dict[str, Any] = {}

    # Categorical encoding
    for col in CATEGORICAL_FEATURES:
        le  = encoders.get(col)
        val = str(row.get(col, "__UNKNOWN__") or "__UNKNOWN__")
        enc_col = col + "_enc"
        if le is not None:
            classes = list(le.classes_)
            record[enc_col] = int(le.transform([val])[0]) if val in classes else -1
        else:
            record[enc_col] = -1

    # Date-derived numerics
    pr_date = pd.to_datetime(row.get("pr_date"), errors="coerce")
    record["day_of_week"] = int(pr_date.dayofweek) if not pd.isna(pr_date) else -1
    record["month"]       = int(pr_date.month)     if not pd.isna(pr_date) else -1

    # Numeric fields
    record["quantity"]    = float(row.get("quantity", 0)   or 0)
    record["net_price"]   = float(row.get("net_price", 0)  or 0)
    record["text_length"] = len(str(row.get("short_text", "") or ""))

    # Build DataFrame in the exact column order the model expects
    df = pd.DataFrame([record])[feature_cols]
    return df


def _score_items(items: list[dict[str, Any]], artefact: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Score a list of PR item dicts.
    Returns a list of result dicts (one per item).
    """
    model     = artefact["model"]
    threshold = artefact["threshold"]
    results   = []

    for item in items:
        try:
            X           = _engineer_row(item, artefact)
            raw_score   = float(model.decision_function(X)[0])
            prediction  = int(model.predict(X)[0])   # 1 = normal, -1 = anomaly
            is_anomaly  = bool(raw_score < threshold)
            confidence  = float(np.clip((threshold - raw_score) / abs(threshold + 1e-9), 0, 1))

            results.append({
                "pr_number":   item.get("pr_number"),
                "item_number": item.get("item_number"),
                "anomaly":     is_anomaly,
                "score":       round(raw_score, 6),
                "confidence":  round(confidence, 4),
                "label":       "ANOMALY" if is_anomaly else "NORMAL",
            })
        except Exception as exc:  # noqa: BLE001
            log.warning("Scoring error for item %s: %s", item.get("item_number"), exc)
            results.append({
                "pr_number":   item.get("pr_number"),
                "item_number": item.get("item_number"),
                "anomaly":     None,
                "score":       None,
                "confidence":  None,
                "label":       "ERROR",
                "error":       str(exc),
            })

    return results


# ── Flask app factory ─────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)

    # Pre-load model at startup so the first request is fast
    try:
        _load_model()
    except FileNotFoundError as exc:
        log.error("⚠  Could not load model at startup: %s", exc)

    # ── Health endpoint ───────────────────────────────────────────────────────
    @app.get("/v1/health")
    def health() -> Response:
        """
        SAP AI Core liveness / readiness probe.
        Returns 200 when the model is loaded, 503 otherwise.
        """
        try:
            art = _load_model()
            return jsonify({
                "status":       "ok",
                "model_loaded": True,
                "n_features":   len(art["feature_cols"]),
                "threshold":    art["threshold"],
            }), 200
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "detail": str(exc)}), 503

    # ── Predict endpoint ──────────────────────────────────────────────────────
    @app.post("/v1/models/pr-anomaly/predict")
    def predict() -> Response:
        """
        Score one or many PR items for anomalies.

        Request body (JSON):
        ────────────────────
        Single item:
            {
              "pr_number": "1000012345",
              "item_number": "00010",
              "pr_type": "NB",
              "company_code": "1000",
              "plant": "1010",
              "purchasing_group": "001",
              "material_group": "01010",
              "quantity": 10.0,
              "unit": "EA",
              "net_price": 250.00,
              "currency": "EUR",
              "pr_date": "2024-03-15",
              "delivery_date": "2024-04-01",
              "gl_account": "400000",
              "cost_center": "10001010",
              "wbs_element": "",
              "order_number": "",
              "profit_center": "YB10",
              "short_text": "Office supplies – A4 paper",
              "header_text": ""
            }

        Batch:
            { "items": [ <item1>, <item2>, … ] }

        Response body (JSON):
        ──────────────────────
        {
          "predictions": [
            {
              "pr_number":   "1000012345",
              "item_number": "00010",
              "anomaly":     false,
              "score":       0.123456,
              "confidence":  0.0,
              "label":       "NORMAL"
            }
          ]
        }
        """
        try:
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return jsonify({"error": "Invalid or missing JSON body"}), 400

            # Accept both single-item and batch payloads
            if "items" in payload:
                items = payload["items"]
                if not isinstance(items, list):
                    return jsonify({"error": "'items' must be a list"}), 400
            else:
                items = [payload]

            if not items:
                return jsonify({"error": "Empty items list"}), 400

            artefact    = _load_model()
            predictions = _score_items(items, artefact)

            return jsonify({"predictions": predictions}), 200

        except FileNotFoundError as exc:
            log.error("Model not available: %s", exc)
            return jsonify({"error": "Model not loaded", "detail": str(exc)}), 503
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected error during prediction")
            return jsonify({"error": "Internal server error", "detail": str(exc)}), 500

    return app


# ── Dev server entry-point (not used in production – gunicorn handles that) ───

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    create_app().run(host="0.0.0.0", port=port, debug=False)
