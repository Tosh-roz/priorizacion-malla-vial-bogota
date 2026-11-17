import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []


def md(t):
    cells.append(nbf.v4.new_markdown_cell(t))


def code(t):
    cells.append(nbf.v4.new_code_cell(t))


md(
    "# Proyecto final — Priorización inteligente de malla vial\n"
    "Pipeline AHP + ML transversal + informe. Ver hoja de ruta y enunciado; "
    "supuestos en `SUPUESTOS.md`."
)

md("## 0. Setup\nImporta rutas del proyecto y librerías.")
code(
    """from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from src.config import DATA_DIR, FEATURE_COLS, TARGET
from src.data_io import ensure_bogota_cuts, load_two_cuts
from src.features import build_training_table, score_ahp, informe_localidad
from src.train_pipeline import train_from_frames, reentrenar, load_artifact
print('ROOT', ROOT)
print('DATA_DIR', DATA_DIR)
"""
)
md(
    "## 1. Datos de entrada\n"
    "Asegura dos cortes IDU (descarga si faltan). El ZIP Itagüí no alcanza (un solo año / FileGDB)."
)
code(
    """paths = ensure_bogota_cuts(force=False)
df_a, df_b = load_two_cuts(*paths)
print('corte A', df_a.shape, paths[0].name)
print('corte B', df_b.shape, paths[1].name)
print(df_a.head(3))
print('ESTAASHTO A:')
print(df_a['ESTAASHTO'].value_counts())
print('nulos A:')
print(df_a.isna().sum())
"""
)

md("## 2. EDA y calidad\nRevisa INDEST inválidos, tipologías y distribución de estado.")
code(
    """print('INDEST describe A')
print(df_a['INDEST'].describe())
print('INDEST < 0:', int((df_a['INDEST'] < 0).sum()))
print(df_a['TIPOMALLAEST'].value_counts())
fig, ax = plt.subplots(figsize=(7,3))
df_a['ESTAASHTO'].value_counts().plot(kind='bar', ax=ax, color='#0E1A2B')
ax.set_title('Distribucion ESTAASHTO - corte A')
plt.tight_layout()
plt.close()
"""
)

md(
    "## 3. Join temporal y etiqueta empeoro\n"
    "Une por CIV/PK y marca deterioro AASHTO (modelo transversal, no serie de tiempo)."
)
code(
    """table = build_training_table(df_a, df_b)
print('filas join valido', len(table))
print('tasa empeoro', round(float(table[TARGET].mean()), 4))
print(table[TARGET].value_counts())
print(table[['segment_id','ESTAASHTO','INDEST','C_norm','J_norm','F_norm',TARGET]].head())
"""
)

md("## 4. Score AHP (reglas)\nPesos Condicion 62.5% / Jerarquia 23.9% / Flujo 13.7%.")
code(
    """table['score_ahp'] = score_ahp(table)
print(table['score_ahp'].describe())
fig, ax = plt.subplots(figsize=(6,3))
ax.hist(table['score_ahp'], bins=40, color='#FFB703', edgecolor='none')
ax.set_title('Distribucion score AHP')
plt.tight_layout()
plt.close()
"""
)

md(
    "## 5. Entrenamiento ML + validacion\n"
    "Pipeline sklearn, train/val/test, metricas y baseline majority. Exporta joblib."
)
code(
    """result = train_from_frames(df_a, df_b)
metrics = result['metrics']
print('artifact:', result['artifact'])
print('n_rows:', result['n_rows'])
keys = ['accuracy','precision','recall','f1','roc_auc','baseline_majority_f1','positive_rate']
print({k: metrics[k] for k in keys})
print(metrics['report'])
"""
)

md("## 6. Score final e informe\nCombina reglas (70%) + P_empeora (30%).")
code(
    """art = load_artifact()
scored = pd.read_csv(DATA_DIR / 'scored_segments.csv')
print(scored[['segment_id','P_empeora','score_ahp','score_final']].head())
print(informe_localidad(scored, localidad=None, top_n=8))
"""
)

md("## 7. reentrenar(ruta_datos)\nAPI para otras ciudades (ejemplo con carpeta data/).")
code(
    """info = reentrenar(DATA_DIR)
print('reentrenar OK', info['n_rows'], 'F1', info['metrics']['f1'])
"""
)

md(
    "## 8. Conclusiones tecnicas\n"
    "- Datos suficientes via IDU abierto; Itagui pendiente de CSV canonico.\n"
    "- ML transversal viable; DTM fuera; SPT/CPT sin fuente publica.\n"
    "- App: streamlit run app.py"
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
out = Path(__file__).resolve().parent / "proyecto.ipynb"
out.write_text(nbf.writes(nb), encoding="utf-8")
print("wrote", out)
