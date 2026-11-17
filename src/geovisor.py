"""Geovisor (MapLibre) de tramos priorizados, filtrable por localidad y barrio.

- `construir_datos(scored, loc, barrio)` arma el paquete de datos (solo lo seleccionado si `embedded=True`).
- `render_html(datos)` devuelve una página HTML autocontenida (para st.components.v1.html o para guardar).
Requiere los archivos de data/geo/ (ver `python -m src.geo_prep`).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, ROOT

GEO_DIR = DATA_DIR / "geo"
PLANTILLA = ROOT / "geovisor" / "plantilla.html"

LOCS = {1: "Usaquén", 2: "Chapinero", 3: "Santa Fe", 4: "San Cristóbal", 5: "Usme", 6: "Tunjuelito", 7: "Bosa",
        8: "Kennedy", 9: "Fontibón", 10: "Engativá", 11: "Suba", 12: "Barrios Unidos", 13: "Teusaquillo",
        14: "Los Mártires", 15: "Antonio Nariño", 16: "Puente Aranda", 17: "La Candelaria",
        18: "Rafael Uribe Uribe", 19: "Ciudad Bolívar", 20: "Sumapaz"}
ESTADOS = ["BUENO", "SATISFACTORIO", "ACEPTABLE", "POBRE", "MUY POBRE", "GRAVE", "FALLADO"]


def geo_disponible() -> bool:
    return all((GEO_DIR / f).exists() for f in ("tramos.json", "barrios.json", "tramo_barrio.json", "localidades.json"))


@lru_cache(maxsize=1)
def _geo() -> dict:
    ld = lambda n: json.loads((GEO_DIR / n).read_text(encoding="utf-8"))
    return {"tramos": ld("tramos.json"), "barrios": ld("barrios.json"),
            "tramo_barrio": ld("tramo_barrio.json"), "localidades": ld("localidades.json")}


_MENORES = {"de", "del", "la", "las", "el", "los", "y", "e"}


def _titulo(s: str) -> str:
    return " ".join(w.lower() if (i and w.lower() in _MENORES) else w.capitalize() if not (len(w) <= 4 and set(w.lower()) <= set("ivxlcdm")) else w.upper()
                    for i, w in enumerate(str(s).split()))


def listar_barrios(scored: pd.DataFrame, loc: int) -> list[tuple[int, str, int]]:
    """Barrios de una localidad con su número de tramos: [(idx_global, nombre, n_tramos)]."""
    g = _geo()
    civ = _col_civ(scored)
    cuenta: dict[int, int] = {}
    for c in scored[civ].astype(str):
        bi = g["tramo_barrio"].get(c)
        if bi is not None and _loc_barrio(g, bi, None) == loc:
            cuenta[bi] = cuenta.get(bi, 0) + 1
    return sorted(((bi, _titulo(g["barrios"][bi]["nombre"]), n) for bi, n in cuenta.items()), key=lambda x: x[1])


def _col_civ(df: pd.DataFrame) -> str:
    return "CIV" if "CIV" in df.columns else "segment_id"


def _loc_barrio(g: dict, bi: int, fallback):
    cod = g["barrios"][bi]["loc"]
    return int(cod) if cod is not None else fallback


def construir_datos(scored: pd.DataFrame, loc: int | None = None, barrio: int | None = None,
                    embedded: bool = False) -> dict:
    """Si `embedded`, solo incluye los tramos de `loc` (y `barrio` si se da). Si no, toda la ciudad."""
    g = _geo()
    civ_col = _col_civ(scored)
    tipos = sorted(scored["TIPOMALLAEST"].astype(str).unique()) if "TIPOMALLAEST" in scored else ["—"]
    rows = []
    for r in scored.itertuples():
        civ = str(getattr(r, civ_col))
        rings = g["tramos"].get(civ)
        if not rings:
            continue
        bi = g["tramo_barrio"].get(civ)
        l = _loc_barrio(g, bi, int(getattr(r, "loce_code", 0))) if bi is not None else int(getattr(r, "loce_code", 0))
        if embedded and ((loc is not None and l != loc) or (barrio is not None and bi != barrio)):
            continue
        est = str(getattr(r, "ESTAASHTO", "")).upper()
        ti = tipos.index(str(getattr(r, "TIPOMALLAEST", "—")))
        rows.append([int(civ) if civ.isdigit() else civ, round(float(r.score_final), 1), round(float(r.P_empeora), 3),
                     int(getattr(r, "INDEST", 0)), ESTADOS.index(est) if est in ESTADOS else -1, ti, l,
                     round(float(getattr(r, "area", 0))), rings, bi if bi is not None else -1])
    # barrios: los usados, o todos los de las localidades presentes (para contexto)
    locs_presentes = {r[6] for r in rows}
    if embedded:
        idxs = [bi for bi, b in enumerate(g["barrios"]) if b["loc"] is not None and int(b["loc"]) in locs_presentes]
    else:
        idxs = list(range(len(g["barrios"])))
    nuevo = {bi: i for i, bi in enumerate(idxs)}
    for row in rows:  # reindexar barrio dentro de la lista exportada
        row[9] = nuevo.get(row[9], -1)
    barrios = [{"n": _titulo(g["barrios"][bi]["nombre"]), "l": int(g["barrios"][bi]["loc"]) if g["barrios"][bi]["loc"] else None,
                "r": g["barrios"][bi]["rings"]} for bi in idxs]
    locs_poly = [{"c": int(x["cod"]), "n": LOCS.get(int(x["cod"]), x["nombre"]), "r": x["rings"]} for x in g["localidades"]]
    return {"rows": rows, "estados": ESTADOS, "tipos": tipos, "locs": {str(k): v for k, v in LOCS.items()},
            "barrios": barrios, "localidades": locs_poly, "embedded": embedded,
            "sel": {"loc": loc, "barrio": nuevo.get(barrio, None) if barrio is not None else None}}


def render_html(datos: dict) -> str:
    html = PLANTILLA.read_text(encoding="utf-8")
    return html.replace("/*__DATA__*/null", json.dumps(datos, separators=(",", ":"), ensure_ascii=False))


def barrio_por_tramo(scored: pd.DataFrame) -> pd.DataFrame:
    """Barrio (sector catastral) y localidad espacial de cada tramo, alineado con el índice de `scored`.

    Devuelve columnas `Barrio` y `Localidad_geo` (NaN si el tramo no cae en ningún sector)."""
    g = _geo()
    civ = scored[_col_civ(scored)].astype(str)
    bi = civ.map(g["tramo_barrio"])
    nombres = pd.Series({i: _titulo(b["nombre"]) for i, b in enumerate(g["barrios"])})
    locs = pd.Series({i: (int(b["loc"]) if b["loc"] is not None else None) for i, b in enumerate(g["barrios"])})
    out = pd.DataFrame(index=scored.index)
    out["Barrio"] = bi.map(nombres)
    out["Localidad_geo"] = bi.map(locs)
    return out
