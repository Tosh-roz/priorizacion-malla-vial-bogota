"""Evaluación reproducible del modelo para la vista técnica de la app.

Reproduce exactamente la partición de `train_pipeline.train_from_frames` (70/15/15, random_state=42,
estratificada) sobre `training_table.csv` y calcula curvas y diagnósticos sobre el conjunto de TEST,
que el modelo no vio al entrenar. Además verifica que el resultado coincida con `metrics.json`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLS, TARGET

# Etiquetas legibles para las variables del modelo
NOMBRES_VARIABLES = {
    "C_norm": "Condición del pavimento (100 − INDEST)",
    "J_norm": "Jerarquía de la vía",
    "F_norm": "Flujo (proxy por jerarquía)",
    "INDEST": "Índice de estado (INDEST)",
    "area": "Área del tramo",
    "length": "Longitud del tramo",
    "loce_code": "Localidad (código)",
}


def particion_test(table: pd.DataFrame):
    X, y = table[FEATURE_COLS], table[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def evaluar_modelo(art: dict, table: pd.DataFrame, metrics_guardadas: dict | None = None) -> dict:
    pipe = art["pipeline"]
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = particion_test(table)
    p = pipe.predict_proba(X_te)[:, 1]
    y = y_te.to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, thr = precision_recall_curve(y, p)
    imp = pipe.named_steps["clf"].feature_importances_
    out = {
        "n_test": int(len(y)),
        "y": y,
        "p": p,
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "pr": pd.DataFrame({"recall": rec[:-1], "precision": prec[:-1], "umbral": thr}),
        "auc": float(roc_auc_score(y, p)),
        "importancias": (
            pd.DataFrame({"Variable": [NOMBRES_VARIABLES.get(c, c) for c in FEATURE_COLS], "Importancia": imp})
            .sort_values("Importancia", ascending=False)
            .reset_index(drop=True)
        ),
        "f1_val": float(f1_score(y_va, pipe.predict(X_va), zero_division=0)),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
    }
    out["f1_test"] = float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
    if metrics_guardadas and metrics_guardadas.get("f1") is not None:
        out["coincide_con_metrics_json"] = abs(out["f1_test"] - metrics_guardadas["f1"]) < 2e-3
    else:
        out["coincide_con_metrics_json"] = None
    return out


def metricas_en_umbral(y: np.ndarray, p: np.ndarray, umbral: float) -> dict:
    pred = (p >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "marcados": int(pred.sum()),
    }


def tasa_por_indest(table: pd.DataFrame) -> pd.DataFrame:
    """Proporción de tramos que empeoran según INDEST inicial (muestra el efecto de los límites de categoría)."""
    g = table.groupby("INDEST")[TARGET].agg(["mean", "size"]).reset_index()
    g.columns = ["INDEST", "Tasa", "Tramos"]
    return g
