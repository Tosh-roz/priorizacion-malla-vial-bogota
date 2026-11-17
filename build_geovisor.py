"""Genera geovisor/geovisor.html: versión autónoma (toda la ciudad embebida; se elige localidad/barrio en la página).

Uso:  python build_geovisor.py     (requiere data/geo/, ver `python -m src.geo_prep`)
"""
import pandas as pd

from src.config import ROOT, SCORED_TABLE
from src.geovisor import construir_datos, render_html

html = render_html(construir_datos(pd.read_csv(SCORED_TABLE), embedded=False))
out = ROOT / "geovisor" / "geovisor.html"
out.write_text(html, encoding="utf-8")
print(out, round(len(html) / 1e6, 1), "MB")
