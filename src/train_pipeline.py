"""Pipeline sklearn + exportación joblib + reentrenar(ruta_datos)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ARTIFACT,
    FEATURE_COLS,
    METRICS_JSON,
    MODELS_DIR,
    SCORED_TABLE,
    TARGET,
    TRAINING_TABLE,
)
from .data_io import ensure_bogota_cuts, load_city_csv, load_two_cuts
from .features import (
    build_training_table,
    informe_localidad,
    prepare_inference_frame,
    score_final,
)


def make_pipeline(random_state: int = 42) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLS,
            )
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _eval(y_true, y_pred, y_proba) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        out["roc_auc"] = None
    out["report"] = classification_report(y_true, y_pred, zero_division=0)
    # baseline: predecir siempre la clase mayoritaria en train se evalúa fuera
    return out


def train_from_frames(df_a: pd.DataFrame, df_b: pd.DataFrame, artifact: Path = ARTIFACT) -> dict:
    table = build_training_table(df_a, df_b)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TRAINING_TABLE, index=False)

    X = table[FEATURE_COLS]
    y = table[TARGET]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    # Baseline: siempre clase mayoritaria del train
    maj = int(y_train.value_counts().idxmax())
    baseline_pred = np.full_like(y_test, maj)
    baseline_f1 = float(f1_score(y_test, baseline_pred, zero_division=0))

    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    metrics = _eval(y_test, pred_test, proba_test)
    metrics["baseline_majority_f1"] = baseline_f1
    metrics["n_train"] = int(len(X_train))
    metrics["n_val"] = int(len(X_val))
    metrics["n_test"] = int(len(X_test))
    metrics["positive_rate"] = float(y.mean())
    metrics["val_f1"] = float(f1_score(y_val, pipe.predict(X_val), zero_division=0))

    # Score sobre toda la tabla
    p_all = pipe.predict_proba(table[FEATURE_COLS])[:, 1]
    table["P_empeora"] = p_all
    table["score_ahp"] = (
        0.625 * table["C_norm"] + 0.239 * table["J_norm"] + 0.137 * table["F_norm"]
    )
    table["score_final"] = score_final(table, p_all)
    table.to_csv(SCORED_TABLE, index=False)

    payload = {
        "pipeline": pipe,
        "feature_cols": FEATURE_COLS,
        "target": TARGET,
        "metrics": metrics,
        "ahp_weights": {"C": 0.625, "J": 0.239, "F": 0.137},
        "score_mix": {"reglas": 0.7, "ml": 0.3},
    }
    joblib.dump(payload, artifact)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"metrics": metrics, "artifact": str(artifact), "n_rows": len(table)}


def train_bogota_default(force_download: bool = False) -> dict:
    a_path, b_path = ensure_bogota_cuts(force=force_download)
    df_a, df_b = load_two_cuts(a_path, b_path)
    return train_from_frames(df_a, df_b)


def reentrenar(ruta_datos: str | Path, artifact: Path = ARTIFACT) -> dict:
    """
    Reentrena con datos de otra ciudad.
    - Si ruta_datos es carpeta con dos CSV (*dic* / *jun* o corte_a.csv / corte_b.csv): usa join temporal.
    - Si es un solo CSV con columna `empeoro`: entrena directo.
    - Si es un solo CSV sin etiqueta: error explícito (hace falta target o dos cortes).
    """
    ruta = Path(ruta_datos)
    if ruta.is_dir():
        csvs = sorted(ruta.glob("*.csv"))
        if len(csvs) >= 2:
            # Heurística: primer y segundo archivo, o nombres con dic/jun
            named_a = [p for p in csvs if "dic" in p.name.lower() or "corte_a" in p.name.lower() or "2024" in p.name]
            named_b = [p for p in csvs if "jun" in p.name.lower() or "corte_b" in p.name.lower() or "2026" in p.name]
            a = named_a[0] if named_a else csvs[0]
            b = named_b[0] if named_b else csvs[1]
            return train_from_frames(pd.read_csv(a), pd.read_csv(b), artifact=artifact)
        if len(csvs) == 1:
            ruta = csvs[0]
        else:
            raise FileNotFoundError(f"No hay CSV en {ruta}")

    df = load_city_csv(ruta)
    if TARGET in df.columns:
        prep = prepare_inference_frame(df)
        prep[TARGET] = pd.to_numeric(df.loc[prep.index, TARGET], errors="coerce")
        prep = prep.dropna(subset=[TARGET])
        prep[TARGET] = prep[TARGET].astype(int)
        X = prep[FEATURE_COLS]
        y = prep[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        pipe = make_pipeline()
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = _eval(y_test, pred, proba)
        payload = {
            "pipeline": pipe,
            "feature_cols": FEATURE_COLS,
            "target": TARGET,
            "metrics": metrics,
            "ahp_weights": {"C": 0.625, "J": 0.239, "F": 0.137},
            "score_mix": {"reglas": 0.7, "ml": 0.3},
        }
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, artifact)
        METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return {"metrics": metrics, "artifact": str(artifact), "n_rows": len(prep)}

    raise ValueError(
        "Para reentrenar se necesitan dos cortes (carpeta con 2 CSV) "
        "o un CSV con columna 'empeoro'."
    )


def load_artifact(path: Path = ARTIFACT) -> dict:
    return joblib.load(path)


def predict_score(df: pd.DataFrame, artifact: Path = ARTIFACT) -> pd.DataFrame:
    art = load_artifact(artifact)
    pipe = art["pipeline"]
    prep = prepare_inference_frame(df)
    p = pipe.predict_proba(prep[FEATURE_COLS])[:, 1]
    prep["P_empeora"] = p
    prep["score_ahp"] = (
        0.625 * prep["C_norm"] + 0.239 * prep["J_norm"] + 0.137 * prep["F_norm"]
    )
    prep["score_final"] = score_final(prep, p)
    return prep


def generar_informe(df_scored: pd.DataFrame, localidad=None, top_n: int = 10) -> str:
    return informe_localidad(df_scored, localidad=localidad, top_n=top_n)
