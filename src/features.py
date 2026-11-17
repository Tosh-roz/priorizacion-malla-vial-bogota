"""Ingeniería de features, etiqueta empeoro y score AHP."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AASHTO_ORDER,
    JERARQUIA_SCORE,
    W_COND,
    W_FLUJO,
    W_JER,
    W_ML,
    W_REGLAS,
)


def _norm_aashto(s: pd.Series) -> pd.Series:
    # Normaliza mayúsculas/minúsculas entre cortes IDU (p.ej. BUENO vs Bueno)
    cleaned = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    mapping = {k.lower(): v for k, v in AASHTO_ORDER.items()}
    # alias frecuentes
    mapping.update(
        {
            "muy pobre": 5,
            "muypobre": 5,
            "sin dato": None,
            "sindato": None,
            "n/d": None,
            "nan": None,
        }
    )
    return cleaned.map(mapping)

def _jerarquia_norm(tipo: pd.Series) -> pd.Series:
    t = tipo.astype(str).str.strip()
    # TIPOMALLAEST suele traer códigos/textos; mapeo flexible
    out = pd.Series(np.nan, index=t.index, dtype=float)
    for key, val in JERARQUIA_SCORE.items():
        out = out.where(~t.str.contains(key, case=False, na=False), val)
    # fallback por patrones frecuentes en IDU
    out = out.fillna(
        t.map(
            {
                "1": 100,
                "2": 100,
                "3": 70,
                "4": 40,
                "5": 30,
            }
        )
    )
    return out.fillna(40.0)


def build_training_table(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """
    Une dos cortes por CIV (fallback PK_ID_CALZADA) y crea label empeoro.
    Features se toman del corte A (estado inicial).
    """
    a = df_a.copy()
    b = df_b.copy()
    key = "CIV" if "CIV" in a.columns and "CIV" in b.columns else "PK_ID_CALZADA"
    a = a.dropna(subset=[key]).drop_duplicates(key)
    b = b.dropna(subset=[key]).drop_duplicates(key)

    merged = a.merge(b[[key, "ESTAASHTO", "INDEST"]], on=key, suffixes=("_a", "_b"))
    merged["ord_a"] = _norm_aashto(merged["ESTAASHTO_a"])
    merged["ord_b"] = _norm_aashto(merged["ESTAASHTO_b"])
    valid = merged["ord_a"].notna() & merged["ord_b"].notna()
    merged = merged.loc[valid].copy()
    merged["empeoro"] = (merged["ord_b"] > merged["ord_a"]).astype(int)

    # Features desde corte A
    merged["INDEST"] = pd.to_numeric(merged["INDEST_a"], errors="coerce")
    merged = merged.loc[merged["INDEST"].notna() & (merged["INDEST"] >= 0)].copy()
    merged["C_norm"] = 100.0 - merged["INDEST"].clip(0, 100)
    tipo_col = "TIPOMALLAEST" if "TIPOMALLAEST" in merged.columns else None
    if tipo_col:
        merged["J_norm"] = _jerarquia_norm(merged[tipo_col])
    elif "J_norm" in merged.columns:
        merged["J_norm"] = pd.to_numeric(merged["J_norm"], errors="coerce").fillna(40)
    else:
        merged["J_norm"] = 40.0

    # Flujo: si no hay F_norm, proxy estable por jerarquía (documentado en SUPUESTOS)
    if "F_norm" in merged.columns:
        merged["F_norm"] = pd.to_numeric(merged["F_norm"], errors="coerce").fillna(50)
    else:
        merged["F_norm"] = (merged["J_norm"] * 0.85 + 10).clip(0, 100)

    merged["area"] = pd.to_numeric(merged.get("Shape__Area"), errors="coerce").fillna(0)
    merged["length"] = pd.to_numeric(merged.get("Shape__Length"), errors="coerce").fillna(0)
    merged["loce_code"] = pd.to_numeric(merged.get("LOCEST"), errors="coerce").fillna(-1)
    merged["ESTAASHTO"] = merged["ESTAASHTO_a"]
    merged["TIPOMALLAEST"] = merged[tipo_col] if tipo_col else "Local"
    merged["segment_id"] = merged[key].astype(str)
    return merged.reset_index(drop=True)


def score_ahp(df: pd.DataFrame) -> pd.Series:
    return (
        W_COND * df["C_norm"]
        + W_JER * df["J_norm"]
        + W_FLUJO * df["F_norm"]
    )


def score_final(df: pd.DataFrame, p_empeora: np.ndarray | pd.Series) -> pd.Series:
    reglas = score_ahp(df)
    return W_REGLAS * reglas + W_ML * (100.0 * pd.Series(p_empeora, index=df.index))


def prepare_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza un CSV de una sola ciudad/corte para inferencia o reentrenamiento sin 2do corte."""
    out = df.copy()
    if "INDEST" not in out.columns:
        raise ValueError("Se requiere columna INDEST")
    out["INDEST"] = pd.to_numeric(out["INDEST"], errors="coerce")
    out = out.loc[out["INDEST"].notna() & (out["INDEST"] >= 0)].copy()
    out["C_norm"] = 100.0 - out["INDEST"].clip(0, 100)
    if "J_norm" not in out.columns:
        if "TIPOMALLAEST" in out.columns:
            out["J_norm"] = _jerarquia_norm(out["TIPOMALLAEST"])
        else:
            out["J_norm"] = 40.0
    if "F_norm" not in out.columns:
        out["F_norm"] = (out["J_norm"] * 0.85 + 10).clip(0, 100)
    out["area"] = pd.to_numeric(out.get("Shape__Area"), errors="coerce").fillna(0)
    out["length"] = pd.to_numeric(out.get("Shape__Length"), errors="coerce").fillna(0)
    out["loce_code"] = pd.to_numeric(out.get("LOCEST"), errors="coerce").fillna(-1)
    if "segment_id" not in out.columns:
        key = "CIV" if "CIV" in out.columns else ("PK_ID_CALZADA" if "PK_ID_CALZADA" in out.columns else out.index)
        out["segment_id"] = out[key].astype(str) if not isinstance(key, type(out.index)) else out.index.astype(str)
    return out


def informe_localidad(df_scored: pd.DataFrame, localidad: str | int | None = None, top_n: int = 10) -> str:
    """Informe generativo rule-based anclado a datos (sin alucinación de cifras)."""
    d = df_scored.copy()
    if localidad is not None and "loce_code" in d.columns:
        d = d.loc[d["loce_code"] == float(localidad)]
    if d.empty:
        return "No hay segmentos para los filtros seleccionados."
    d = d.sort_values("score_final", ascending=False).head(top_n)
    lines = [
        f"Informe de priorización — top {len(d)} segmentos",
        f"Score medio: {d['score_final'].mean():.1f} | P(empeora) media: {d['P_empeora'].mean():.3f}",
        "Criterio: 70% reglas AHP (condición/jerarquía/flujo) + 30% ML transversal.",
        "Segmentos prioritarios:",
    ]
    for _, r in d.iterrows():
        lines.append(
            f"- ID {r.get('segment_id', '?')}: score={r['score_final']:.1f}, "
            f"estado={r.get('ESTAASHTO', 'n/d')}, P_empeora={r['P_empeora']:.3f}, "
            f"C={r['C_norm']:.0f}, J={r['J_norm']:.0f}, F={r['F_norm']:.0f}"
        )
    lines.append(
        "Limitaciones: Bitcarrier/Waze/SPT-CPT pueden estar ausentes; "
        "DTM descartado por precisión vertical insuficiente."
    )
    return "\n".join(lines)
