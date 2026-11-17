"""Carga local y descarga parametrizada desde ArcGIS REST."""
from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .config import API_CORTE_A, API_CORTE_B, API_FIELDS, DATA_DIR


def fetch_arcgis_layer(
    layer_url: str,
    outfile: Path,
    fields: list[str] | None = None,
    page: int = 1000,
    max_pages: int = 150,
) -> Path:
    """Descarga atributos (sin geometría) paginando resultOffset."""
    fields = fields or API_FIELDS
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    meta = json.loads(
        urllib.request.urlopen(f"{layer_url}?f=pjson", timeout=60).read().decode()
    )
    oid = next(f["name"] for f in meta["fields"] if f["type"] == "esriFieldTypeOID")
    rows: list[dict] = []
    offset = 0
    for i in range(max_pages):
        params = {
            "where": "1=1",
            "outFields": ",".join(fields),
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": str(offset),
            "resultRecordCount": str(page),
            "orderByFields": f"{oid} ASC",
        }
        url = f"{layer_url}/query?{urllib.parse.urlencode(params)}"
        payload = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    payload = json.loads(r.read().decode())
                if payload.get("error"):
                    raise RuntimeError(payload["error"])
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Fallo descarga página {i} de {layer_url}")
        feats = payload.get("features") or []
        if not feats:
            break
        rows.extend(ft.get("attributes", {}) for ft in feats)
        print(f"fetch {outfile.name}: page {i} +{len(feats)} total={len(rows)}", flush=True)
        if len(feats) < page:
            break
        offset += page
        time.sleep(0.12)

    with outfile.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    return outfile


def ensure_bogota_cuts(force: bool = False) -> tuple[Path, Path]:
    """Garantiza CSVs de los dos cortes IDU en data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    a = DATA_DIR / "bogota_estado_dic2024.csv"
    b = DATA_DIR / "bogota_estado_jun2026.csv"
    if force or not a.exists() or a.stat().st_size < 1000:
        fetch_arcgis_layer(API_CORTE_A, a)
    if force or not b.exists() or b.stat().st_size < 1000:
        fetch_arcgis_layer(API_CORTE_B, b)
    return a, b


def load_city_csv(ruta: str | Path) -> pd.DataFrame:
    """
    Carga CSV de cualquier ciudad.
    Columnas mínimas recomendadas: CIV o PK_ID_CALZADA, INDEST, ESTAASHTO,
    TIPOMALLAEST (o J_norm), opcional F_norm, LOCEST, Shape__Area, Shape__Length.
    """
    path = Path(ruta)
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No hay CSV en {path}")
        # Si hay dos cortes con nombres reconocibles, el caller debe usar build_training_table
        path = csvs[0]
    df = pd.read_csv(path)
    return df


def load_two_cuts(ruta_a: str | Path, ruta_b: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(ruta_a), pd.read_csv(ruta_b)
