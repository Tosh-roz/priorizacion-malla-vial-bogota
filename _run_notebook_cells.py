"""Ejecuta las celdas de codigo de proyecto.ipynb en orden (verificacion offline)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
nb = nbformat.read(ROOT / "proyecto.ipynb", as_version=4)
globs: dict = {"__name__": "__main__"}
code_cells = [c for c in nb.cells if c.cell_type == "code"]
print(f"Ejecutando {len(code_cells)} celdas de codigo...")
for i, cell in enumerate(code_cells, 1):
    print(f"\n=== CELDA {i}/{len(code_cells)} ===")
    try:
        exec(cell.source, globs)
        print(f"OK celda {i}")
    except Exception as e:
        print(f"FAIL celda {i}: {type(e).__name__}: {e}")
        raise
print("\nTODAS LAS CELDAS OK")
# Marcar notebook como verificado
meta = {"executed_ok": True, "n_code_cells": len(code_cells)}
(ROOT / "models" / "notebook_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
