"""Prepara los datos geográficos del geovisor: geometría de tramos, barrios (sector catastral) y localidades.

Uso:  python -m src.geo_prep
Fuentes públicas (solo lectura): IDU EstadoMallaVialOD y Catastro Bogotá (Mapa_Referencia, capas 37 y 48).
Salida en data/geo/: tramos.json, barrios.json, localidades.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .config import DATA_DIR, SCORED_TABLE

GEO_DIR = DATA_DIR / "geo"
TRAMOS_URL = (
    "https://services2.arcgis.com/qAnmJmgmnVso8Fzo/arcgis/rest/services/"
    "EstadoMallaVialOD/FeatureServer/0/query"
)
CAT = "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/Mapa_Referencia/Mapa_Referencia/MapServer"


def _poly(rings):
    """Une todas las partes del polígono (los huecos se ignoran; irrelevante para asignar tramos)."""
    return unary_union([Polygon(r).buffer(0) for r in rings if len(r) >= 4])


def _get(url, params, intentos=4):
    for i in range(intentos):
        try:
            return requests.get(url, params=params, timeout=90).json()
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"falló la descarga: {url}")


def _paginar(url, campos, offset_step, extra):
    out, off = [], 0
    while True:
        p = dict(where="1=1", outFields=campos, returnGeometry="true", outSR=4326, geometryPrecision=5,
                 resultOffset=off, resultRecordCount=offset_step, f="json", **extra)
        r = _get(url, p)
        feats = r.get("features", [])
        out += feats
        if not r.get("exceededTransferLimit") or not feats:
            return out
        off += len(feats)


def descargar_tramos() -> dict:
    feats = _paginar(TRAMOS_URL, "CIV", 2000, dict(maxAllowableOffset=0.00004, orderByFields="OBJECTID"))
    return {str(f["attributes"]["CIV"]): f["geometry"]["rings"] for f in feats if f.get("geometry")}


def descargar_capa(idx: int, campos: str) -> list:
    feats = _paginar(f"{CAT}/{idx}/query", campos, 1000, dict(maxAllowableOffset=0.00005, orderByFields="OBJECTID"))
    return [f for f in feats if f.get("geometry")]


def preparar() -> None:
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    print("Tramos IDU…")
    tramos = descargar_tramos()
    (GEO_DIR / "tramos.json").write_text(json.dumps(tramos, separators=(",", ":")), encoding="utf-8")
    print("Localidades…")
    locs = descargar_capa(48, "LOCCODIGO,LOCNOMBRE")
    print("Barrios (sector catastral)…")
    bar = descargar_capa(37, "SCACODIGO,SCANOMBRE")

    loc_polys = []
    for f in locs:
        rings = f["geometry"]["rings"]
        loc_polys.append((str(f["attributes"]["LOCCODIGO"]), f["attributes"]["LOCNOMBRE"], rings))
    (GEO_DIR / "localidades.json").write_text(
        json.dumps([{"cod": c, "nombre": n, "rings": r} for c, n, r in loc_polys], separators=(",", ":")),
        encoding="utf-8")

    # barrio -> localidad (por centroide) y barrio por tramo (centroide del tramo dentro del barrio)
    bar_geoms = [_poly(f["geometry"]["rings"]) for f in bar]
    tree = STRtree(bar_geoms)
    loc_geoms = [_poly(r) for _, _, r in loc_polys]
    ltree = STRtree(loc_geoms)

    def contenedor(pt, geoms, t):
        for i in t.query(pt):
            if geoms[i].contains(pt):
                return int(i)
        return None

    barrios = []
    for f, g in zip(bar, bar_geoms):
        pt = g.representative_point()
        li = contenedor(pt, loc_geoms, ltree)
        barrios.append({"cod": str(f["attributes"]["SCACODIGO"]), "nombre": f["attributes"]["SCANOMBRE"],
                        "loc": loc_polys[li][0] if li is not None else None, "rings": f["geometry"]["rings"]})
    tramo_barrio = {}
    for civ, rings in tramos.items():
        pt = _poly(rings).representative_point()
        bi = contenedor(pt, bar_geoms, tree)
        if bi is not None:
            tramo_barrio[civ] = bi
    for i, b in enumerate(barrios):
        b["n"] = 0
    for bi in tramo_barrio.values():
        barrios[bi]["n"] += 1
    (GEO_DIR / "barrios.json").write_text(json.dumps(barrios, separators=(",", ":")), encoding="utf-8")
    (GEO_DIR / "tramo_barrio.json").write_text(json.dumps(tramo_barrio, separators=(",", ":")), encoding="utf-8")
    print(f"tramos={len(tramos)} con barrio={len(tramo_barrio)} barrios={len(barrios)} localidades={len(loc_polys)}")


if __name__ == "__main__":
    preparar()
