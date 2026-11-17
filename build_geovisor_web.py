"""Genera el sitio estático del geovisor (geovisor_web/) para publicarlo en Vercel u otro hosting estático.

Datos por localidad en formato compacto (coordenadas como enteros 1e-5° codificados por deltas), que el navegador
descarga en paralelo y comprimido (gzip/brotli). Uso:  python build_geovisor_web.py
Requiere data/geo/ (ver `python -m src.geo_prep`) y data/scored_segments.csv.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

from src.config import ROOT, SCORED_TABLE
from src.geovisor import construir_datos

OUT = ROOT / "geovisor_web"
DATA = OUT / "data"


def codificar_anillo(ring) -> list[int]:
    """[[lon,lat],...] -> [x0, y0, dx1, dy1, ...] con x = round(lon*1e5)."""
    out, px, py = [], 0, 0
    for lon, lat in ring:
        x, y = round(lon * 1e5), round(lat * 1e5)
        out += [x - px, y - py]
        px, py = x, y
    return out


def main() -> None:
    d = construir_datos(pd.read_csv(SCORED_TABLE), embedded=False)
    DATA.mkdir(parents=True, exist_ok=True)
    for f in DATA.glob("*.json"):
        f.unlink()

    por_loc: dict[int, list] = {}
    for r in d["rows"]:
        civ, score, p, indest, est, tipo, loc, area, rings, barrio = r
        por_loc.setdefault(loc, []).append(
            [civ, score, p, indest, est, tipo, area, barrio, [codificar_anillo(a) for a in rings]])
    for loc, rows in por_loc.items():
        (DATA / f"loc_{loc}.json").write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")

    meta = {
        "estados": d["estados"], "tipos": d["tipos"], "locs": d["locs"],
        "localidades": d["localidades"], "barrios": d["barrios"],
        "archivos": sorted(por_loc), "conteo": {str(k): len(v) for k, v in por_loc.items()},
    }
    (DATA / "meta.json").write_text(json.dumps(meta, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    tam = sum(f.stat().st_size for f in DATA.glob("*.json")) / 1e6
    print(f"{len(d['rows']):,} tramos en {len(por_loc)} archivos; datos sin comprimir: {tam:.1f} MB")


if __name__ == "__main__":
    main()
