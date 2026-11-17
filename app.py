"""Demo Streamlit — priorización de malla vial (lenguaje claro + vista técnica + geovisor)."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import ARTIFACT, METRICS_JSON, SCORED_TABLE, TRAINING_TABLE
from src.evaluacion import evaluar_modelo, metricas_en_umbral, tasa_por_indest
from src.geovisor import PLANTILLA, barrio_por_tramo, construir_datos, geo_disponible, render_html
from src.train_pipeline import generar_informe, load_artifact, predict_score, reentrenar

# ───────────────────────── constantes ─────────────────────────
LOCALIDADES = {
    1: "Usaquén", 2: "Chapinero", 3: "Santa Fe", 4: "San Cristóbal", 5: "Usme", 6: "Tunjuelito", 7: "Bosa",
    8: "Kennedy", 9: "Fontibón", 10: "Engativá", 11: "Suba", 12: "Barrios Unidos", 13: "Teusaquillo",
    14: "Los Mártires", 15: "Antonio Nariño", 16: "Puente Aranda", 17: "La Candelaria",
    18: "Rafael Uribe Uribe", 19: "Ciudad Bolívar", 20: "Sumapaz",
}
ESTADO_AMIGABLE = {
    "BUENO": ("Bueno", "🟢"), "SATISFACTORIO": ("Satisfactorio", "🟢"), "ACEPTABLE": ("Aceptable", "🟡"),
    "POBRE": ("Pobre", "🟠"), "MUY POBRE": ("Muy pobre", "🟠"), "GRAVE": ("Grave", "🔴"),
    "FALLADO": ("Fallado", "🔴"), "SIN DATO": ("Sin dato", "⚪"),
}
ORDEN_ESTADO = ["Bueno", "Satisfactorio", "Aceptable", "Pobre", "Muy pobre", "Grave", "Fallado"]
COLOR_ESTADO = ["#1a9850", "#66bd63", "#d9ef8b", "#fdae61", "#f46d43", "#d73027", "#7f0000"]
ORDEN_PRIO = ["Urgente", "Alta", "Media", "Baja"]
COLOR_PRIO = ["#d7263d", "#f28c28", "#f4d35e", "#2e9e6b"]
INK, MUTED = "#0E1A2B", "#5B6B82"


def prioridad_label(score: float) -> tuple[str, str]:
    if score >= 65:
        return "Urgente", "🔴"
    if score >= 50:
        return "Alta", "🟠"
    if score >= 35:
        return "Media", "🟡"
    return "Baja", "🟢"


def estado_nombre(val) -> str:
    return ESTADO_AMIGABLE.get(str(val).strip().upper(), (str(val), ""))[0]


def estado_txt(val) -> str:
    nombre, icono = ESTADO_AMIGABLE.get(str(val).strip().upper(), (str(val), "•"))
    return f"{icono} {nombre}"


def localidad_nombre(code) -> str:
    try:
        c = int(float(code))
    except (TypeError, ValueError):
        return "Sin localidad"
    return LOCALIDADES.get(c, f"Localidad {c}")


def preparar_vista(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Localidad"] = out["loce_code"].map(localidad_nombre) if "loce_code" in out.columns else "—"
    out["Estado"] = out["ESTAASHTO"].map(estado_nombre) if "ESTAASHTO" in out.columns else "—"
    out["Estado de la vía"] = out["ESTAASHTO"].map(estado_txt) if "ESTAASHTO" in out.columns else "—"
    out["Tipo de vía"] = out["TIPOMALLAEST"] if "TIPOMALLAEST" in out.columns else "—"
    s = out["score_final"]
    out["Nivel de prioridad"] = np.select([s >= 65, s >= 50, s >= 35], ["Urgente", "Alta", "Media"], "Baja")
    icono = {"Urgente": "🔴", "Alta": "🟠", "Media": "🟡", "Baja": "🟢"}
    out["Prioridad"] = out["Nivel de prioridad"].map(lambda n: f"{icono[n]} {n}")
    out["Puntaje"] = s.round(1)
    out["Riesgo (%)"] = (out["P_empeora"] * 100).round(0)
    out["ID del tramo"] = out["segment_id"].astype(str)
    return out


# ───────────────────────── gráficos ─────────────────────────
def _cfg(chart: alt.Chart) -> alt.Chart:
    return (chart.configure_view(strokeWidth=0).configure_axis(labelColor=INK, titleColor=MUTED, gridColor="#E3E8EF",
                                                               labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=INK, titleColor=MUTED).configure(background="transparent"))


def grafico_estado(df: pd.DataFrame) -> alt.Chart:
    c = df["Estado"].value_counts().reindex(ORDEN_ESTADO).fillna(0).astype(int).rename_axis("Estado").reset_index(name="Tramos")
    c["Porcentaje"] = (100 * c["Tramos"] / max(c["Tramos"].sum(), 1)).round(1)
    base = alt.Chart(c).encode(
        x=alt.X("Estado:N", sort=ORDEN_ESTADO, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Tramos:Q", title="Tramos"),
        tooltip=["Estado", alt.Tooltip("Tramos:Q", format=","), alt.Tooltip("Porcentaje:Q", format=".1f", title="% del total")])
    barras = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        color=alt.Color("Estado:N", scale=alt.Scale(domain=ORDEN_ESTADO, range=COLOR_ESTADO), legend=None))
    txt = base.mark_text(dy=-8, color=INK, fontSize=11).encode(text=alt.Text("Porcentaje:Q", format=".0f"))
    return _cfg((barras + txt).properties(height=300))


def grafico_prioridad(df: pd.DataFrame) -> alt.Chart:
    c = df["Nivel de prioridad"].value_counts().reindex(ORDEN_PRIO).fillna(0).astype(int).rename_axis("Prioridad").reset_index(name="Tramos")
    c["Porcentaje"] = (100 * c["Tramos"] / max(c["Tramos"].sum(), 1)).round(1)
    base = alt.Chart(c).encode(
        x=alt.X("Prioridad:N", sort=ORDEN_PRIO, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Tramos:Q", title="Tramos"),
        tooltip=["Prioridad", alt.Tooltip("Tramos:Q", format=","), alt.Tooltip("Porcentaje:Q", format=".1f", title="% del total")])
    barras = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        color=alt.Color("Prioridad:N", scale=alt.Scale(domain=ORDEN_PRIO, range=COLOR_PRIO), legend=None))
    txt = base.mark_text(dy=-8, color=INK, fontSize=11).encode(text=alt.Text("Porcentaje:Q", format=".0f"))
    return _cfg((barras + txt).properties(height=300))


def grafico_histograma_puntaje(df: pd.DataFrame) -> alt.Chart:
    bandas = pd.DataFrame({"ini": [0, 35, 50, 65], "fin": [35, 50, 65, 100], "Prioridad": ["Baja", "Media", "Alta", "Urgente"]})
    fondo = alt.Chart(bandas).mark_rect(opacity=0.14).encode(
        x="ini:Q", x2="fin:Q", color=alt.Color("Prioridad:N", scale=alt.Scale(domain=ORDEN_PRIO, range=COLOR_PRIO), legend=alt.Legend(orient="top", title=None)))
    hist = alt.Chart(df[["score_final"]]).mark_bar(color="#3B5B8C", opacity=0.9).encode(
        x=alt.X("score_final:Q", bin=alt.Bin(step=2.5), title="Puntaje final (0–100, más alto = reparar antes)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("count():Q", title="Tramos"), tooltip=[alt.Tooltip("count():Q", title="Tramos", format=",")])
    return _cfg((fondo + hist).properties(height=240))


def grafico_ranking(tab: pd.DataFrame, col_nombre: str, alto_por_barra: int = 26) -> alt.Chart:
    long = tab.melt(id_vars=[col_nombre, "% Urgente + Alta", "Tramos"], value_vars=["% Urgente", "% Alta"], var_name="Nivel", value_name="Pct")
    long["Nivel"] = long["Nivel"].map({"% Urgente": "Urgente", "% Alta": "Alta"})
    orden = tab.sort_values("% Urgente + Alta", ascending=False)[col_nombre].tolist()
    barras = alt.Chart(long).mark_bar().encode(
        y=alt.Y(f"{col_nombre}:N", sort=orden, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
        x=alt.X("Pct:Q", title="% de tramos en prioridad Urgente + Alta", stack="zero"),
        color=alt.Color("Nivel:N", scale=alt.Scale(domain=["Urgente", "Alta"], range=[COLOR_PRIO[0], COLOR_PRIO[1]]), legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("Nivel:N", sort="ascending"),
        tooltip=[col_nombre, alt.Tooltip("Nivel:N"), alt.Tooltip("Pct:Q", format=".1f", title="% de sus tramos"),
                 alt.Tooltip("% Urgente + Alta:Q", format=".1f"), alt.Tooltip("Tramos:Q", format=",")])
    tot = alt.Chart(tab).mark_text(align="left", dx=4, color=INK, fontSize=11).encode(
        y=alt.Y(f"{col_nombre}:N", sort=orden), x=alt.X("% Urgente + Alta:Q"), text=alt.Text("% Urgente + Alta:Q", format=".1f"))
    return _cfg((barras + tot).properties(height=max(120, alto_por_barra * len(tab))))


# ───────────────────────── tablas de territorio ─────────────────────────
def indice_territorio(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Índice de criticidad = % de tramos con prioridad Urgente + Alta sobre el total del territorio."""
    g = df.groupby(col)
    t = pd.DataFrame({
        "Tramos": g.size(),
        "Urgentes": g["Nivel de prioridad"].apply(lambda s: (s == "Urgente").sum()),
        "Altas": g["Nivel de prioridad"].apply(lambda s: (s == "Alta").sum()),
        "Puntaje medio": g["score_final"].mean().round(1),
        "Riesgo medio (%)": (100 * g["P_empeora"].mean()).round(0),
    }).reset_index()
    t["% Urgente"] = (100 * t["Urgentes"] / t["Tramos"]).round(1)
    t["% Alta"] = (100 * t["Altas"] / t["Tramos"]).round(1)
    t["% Urgente + Alta"] = (t["% Urgente"] + t["% Alta"]).round(1)
    return t.sort_values(["% Urgente + Alta", "Tramos"], ascending=[False, False]).reset_index(drop=True)


def df_a_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")



def _mezclar(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def _estilo_escala(serie: pd.Series, vmin: float, vmax: float, colores: list[tuple]) -> list[str]:
    """Color de fondo por valor (escala de varios tonos) con texto claro/oscuro según el contraste."""
    out = []
    rango = (vmax - vmin) or 1.0
    for v in serie:
        if pd.isna(v):
            out.append("")
            continue
        t = min(max((v - vmin) / rango, 0.0), 1.0) * (len(colores) - 1)
        i = min(int(t), len(colores) - 2)
        r, g, b = _mezclar(colores[i], colores[i + 1], t - i)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        out.append(f"background-color: rgb({r},{g},{b}); color: {'#FFFFFF' if lum < 140 else '#0E1A2B'}; font-weight: 600")
    return out


ESCALA_RIESGO = [(46, 158, 107), (244, 211, 94), (242, 140, 40), (215, 38, 61)]  # verde → amarillo → naranja → rojo
ESCALA_CONTEO = [(255, 255, 255), (253, 224, 186), (242, 140, 40)]
ESCALA_PUNTAJE = [(46, 158, 107), (244, 211, 94), (242, 140, 40), (215, 38, 61)]


def tabla_territorio_estilizada(t: pd.DataFrame, col_nombre: str):
    """Formato condicional: semáforo en el índice (% Urgente + Alta) y en riesgo/puntaje; degradé en conteos."""
    sty = t.style
    sty = sty.apply(lambda s: _estilo_escala(s, 15, 50, ESCALA_RIESGO), subset=["% Urgente + Alta"])
    sty = sty.apply(lambda s: _estilo_escala(s, 0, max(float(t["% Urgente"].max()), 1), ESCALA_CONTEO), subset=["% Urgente"])
    sty = sty.apply(lambda s: _estilo_escala(s, 0, max(float(t["% Alta"].max()), 1), ESCALA_CONTEO), subset=["% Alta"])
    sty = sty.apply(lambda s: _estilo_escala(s, 25, 50, ESCALA_PUNTAJE), subset=["Puntaje medio"])
    sty = sty.apply(lambda s: _estilo_escala(s, 10, 40, ESCALA_RIESGO), subset=["Riesgo medio (%)"])
    return sty.format({
        "Tramos": "{:,.0f}", "Urgentes": "{:,.0f}", "Altas": "{:,.0f}", "Puntaje medio": "{:.1f}",
        "Riesgo medio (%)": "{:.0f} %", "% Urgente": "{:.1f} %", "% Alta": "{:.1f} %", "% Urgente + Alta": "{:.1f} %",
    })

COL_TABLA_TERR = {
    "Tramos": st.column_config.NumberColumn("Tramos", format="%d"),
    "Urgentes": st.column_config.NumberColumn("Urgentes", format="%d"),
    "Altas": st.column_config.NumberColumn("Altas", format="%d"),
    "% Urgente + Alta": st.column_config.ProgressColumn("% Urgente + Alta", min_value=0, max_value=100, format="%.1f %%"),
    "% Urgente": st.column_config.NumberColumn("% Urgente", format="%.1f %%"),
    "% Alta": st.column_config.NumberColumn("% Alta", format="%.1f %%"),
    "Puntaje medio": st.column_config.NumberColumn("Puntaje medio", format="%.1f"),
    "Riesgo medio (%)": st.column_config.NumberColumn("Riesgo medio", format="%d %%"),
}

# ───────────────────────── página ─────────────────────────
st.set_page_config(page_title="¿Qué vías reparar primero?", page_icon="🛣️", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
  .stApp { background: linear-gradient(180deg, #F7F4EF 0%, #EEF2F7 45%, #E8EDF4 100%); }
  h1, h2, h3, h4 { color: #0E1A2B !important; font-family: "Segoe UI", sans-serif; }
  [data-testid="stCaptionContainer"], .stCaption, small { color: #3F4F66 !important; }
  [data-testid="stWidgetLabel"] p, label { color: #0E1A2B !important; }
  [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #D5DCE6; }
  [data-testid="stSidebar"] * { color: #0E1A2B; }
  [data-testid="stHeader"] { background: rgba(247,244,239,0.9); }
  .hero { background: #0E1A2B; color: #fff; padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1rem; }
  .hero h1 { color: #FFFFFF !important; margin: 0 0 0.4rem 0; font-size: 1.8rem; }
  .hero p { color: #D5DCE6 !important; margin: 0; font-size: 1.05rem; }
  .pill { display: inline-block; background: #FFB703; color: #0E1A2B !important; font-weight: 700; padding: 0.25rem 0.7rem;
          border-radius: 999px; font-size: 0.85rem; margin-bottom: 0.7rem; }
  .card { background: white; border: 1px solid #D5DCE6; border-radius: 16px; padding: 1rem 1.1rem; box-shadow: 0 8px 24px rgba(14,26,43,0.06); height: 100%; }
  .big-num { font-size: 2rem; font-weight: 800; color: #0E1A2B; line-height: 1.1; }
  .mid-num { font-size: 1.35rem; font-weight: 800; color: #0E1A2B; line-height: 1.2; }
  .muted { color: #5B6B82; font-size: 0.92rem; }
  .explain { background: #FFF8E7; border-left: 5px solid #FFB703; padding: 0.9rem 1rem; border-radius: 10px; margin: 0.8rem 0 1.2rem 0; color: #0E1A2B; }
  .hallazgo { background: #EAF2FB; border-left: 5px solid #1F78B4; padding: 0.9rem 1rem; border-radius: 10px; margin: 0.6rem 0 1rem 0; color: #0E1A2B; }
  .alerta { background: #FDECEC; border-left: 5px solid #D7263D; padding: 0.9rem 1rem; border-radius: 10px; margin: 0.6rem 0 1rem 0; color: #0E1A2B; }
  .pasos li { margin-bottom: 0.25rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="pill">Herramienta de apoyo a decisiones</div>
  <h1>¿Qué tramos de vía conviene reparar primero?</h1>
  <p>La app cruza el estado del pavimento, el tipo de vía y un modelo que estima
  qué tramos tienden a empeorar. El resultado es una lista clara de prioridades para Bogotá.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Explorar")
    st.caption("Ajusta lo que quieres ver en la app.")
    top_n = st.slider("¿Cuántos tramos mostrar en el top?", 5, 40, 12)
    modo = st.radio("Nivel de detalle", ["Público general", "Vista técnica"], index=0,
                    help="La vista técnica muestra desempeño del modelo, diagnósticos y supuestos.")
    uploaded = st.file_uploader("Subir otro CSV de tramos (opcional)", type=["csv"],
                                help="Debe incluir al menos la columna INDEST (y, para el mapa, CIV).")

if not ARTIFACT.exists():
    st.error("Aún no hay modelo entrenado. Ejecuta en la carpeta del proyecto:\n\n"
             '`python -c "from src.train_pipeline import train_bogota_default; train_bogota_default()"`')
    st.stop()

art = load_artifact()
metrics = art.get("metrics", {})

try:
    if uploaded is not None:
        scored = predict_score(pd.read_csv(uploaded))
    elif SCORED_TABLE.exists():
        scored = pd.read_csv(SCORED_TABLE)
    elif TRAINING_TABLE.exists():
        scored = predict_score(pd.read_csv(TRAINING_TABLE))
    else:
        st.warning("No hay resultados. Entrena el modelo o sube un CSV.")
        st.stop()
except Exception as e:  # CSV subido con formato inesperado
    st.error(f"No se pudo procesar el CSV: {e}")
    st.stop()

fuente = uploaded.name if uploaded is not None else "bogota_default"


@st.cache_data(show_spinner=False)
def _vista(fuente: str, _scored: pd.DataFrame) -> pd.DataFrame:
    return preparar_vista(_scored)


@st.cache_data(show_spinner=False)
def _con_barrio(fuente: str, _scored: pd.DataFrame) -> pd.DataFrame:
    v = preparar_vista(_scored)
    b = barrio_por_tramo(_scored)
    v["Barrio"] = b["Barrio"]
    v["Localidad (mapa)"] = b["Localidad_geo"].map(lambda x: LOCALIDADES.get(int(x), "—") if pd.notna(x) else None)
    return v


vista = _vista(fuente, scored)

locs_disponibles = sorted(vista["Localidad"].dropna().unique().tolist())
with st.sidebar:
    loc_sel = st.selectbox("Localidad", ["Toda la ciudad"] + locs_disponibles, index=0)

filtrado = vista if loc_sel == "Toda la ciudad" else vista.loc[vista["Localidad"] == loc_sel]
if filtrado.empty:
    st.warning("No hay tramos para ese filtro.")
    st.stop()

# ───────────────────────── resumen ─────────────────────────
n_total = len(filtrado)
criticos = int(filtrado["Estado"].isin(["Pobre", "Muy pobre", "Grave", "Fallado"]).sum())
urgentes = int((filtrado["Nivel de prioridad"] == "Urgente").sum())
alta = int((filtrado["Nivel de prioridad"] == "Alta").sum())
pct_crit = 100.0 * criticos / n_total
pct_ua = 100.0 * (urgentes + alta) / n_total

st.markdown("### En una mirada")
c1, c2, c3, c4 = st.columns(4)
for col, titulo, valor, sub, color in [
    (c1, "Tramos analizados", f"{n_total:,}", loc_sel, INK),
    (c2, "En mal estado hoy", f"{criticos:,}", f"{pct_crit:.1f} % · Pobre, Muy pobre, Grave o Fallado", INK),
    (c3, "Prioridad urgente", f"{urgentes:,}", f"{100 * urgentes / n_total:.1f} % · puntaje ≥ 65", "#B91C1C"),
    (c4, "Prioridad alta", f"{alta:,}", f"{100 * alta / n_total:.1f} % · puntaje 50–65", "#C2410C"),
]:
    col.markdown(f'<div class="card"><div class="muted">{titulo}</div><div class="big-num" style="color:{color}">{valor}</div>'
                 f'<div class="muted">{sub}</div></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="explain">
<b>¿Cómo se decide?</b> Primero se mira el estado actual de la vía (lo más importante),
luego si es una vía principal o local, y un estimado de congestión.
Después un modelo de aprendizaje automático calcula la probabilidad de que el tramo
<em>empeore</em>. Todo se resume en un solo puntaje de 0 a 100: <b>más alto = reparar antes</b>.
</div>
""",
    unsafe_allow_html=True,
)

g1, g2 = st.columns(2)
with g1:
    st.markdown("#### ¿En qué estado están las vías?")
    st.caption("Número de tramos por estado del pavimento (índice AASHTO). Sobre cada barra, el % del total.")
    st.altair_chart(grafico_estado(filtrado), width="stretch")
with g2:
    st.markdown("#### ¿Cuántos tramos hay por prioridad?")
    st.caption("Urgente ≥ 65 · Alta 50–65 · Media 35–50 · Baja < 35. Sobre cada barra, el % del total.")
    st.altair_chart(grafico_prioridad(filtrado), width="stretch")

st.markdown("#### ¿Cómo se reparte el puntaje?")
st.caption("Cada barra agrupa tramos con puntaje similar; las franjas de color son los niveles de prioridad.")
st.altair_chart(grafico_histograma_puntaje(filtrado), width="stretch")

# ───────────────────────── panel de territorios críticos ─────────────────────────
st.markdown("### Panel de territorios críticos")
st.markdown(
    '<div class="muted">Índice de criticidad = tramos en prioridad <b>Urgente + Alta</b> sobre el total de tramos del territorio. '
    "Sirve para comparar zonas de distinto tamaño: un % alto significa que una proporción grande de sus calles necesita intervención.</div>",
    unsafe_allow_html=True,
)
t_loc = indice_territorio(vista, "Localidad")
mejor_loc = t_loc.iloc[0]
ciudad_pct = 100.0 * ((vista["Nivel de prioridad"].isin(["Urgente", "Alta"])).mean())
hay_geo = geo_disponible()
vista_b = _con_barrio(fuente, scored) if hay_geo else None

k1, k2, k3 = st.columns(3)
k1.markdown(f'<div class="card"><div class="muted">Promedio de la ciudad</div><div class="big-num">{ciudad_pct:.1f} %</div>'
            f'<div class="muted">de los tramos en Urgente + Alta</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="card"><div class="muted">Localidad más crítica</div><div class="mid-num">{mejor_loc["Localidad"]}</div>'
            f'<div class="muted">{mejor_loc["% Urgente + Alta"]:.1f} % · {int(mejor_loc["Urgentes"] + mejor_loc["Altas"]):,} de {int(mejor_loc["Tramos"]):,} tramos</div></div>',
            unsafe_allow_html=True)
if hay_geo:
    tb_all = indice_territorio(vista_b.dropna(subset=["Barrio"]), "Barrio")
    tb_ok = tb_all[tb_all["Tramos"] >= 30]
    peor_b = tb_ok.iloc[0]
    k3.markdown(f'<div class="card"><div class="muted">Barrio más crítico (≥ 30 tramos)</div><div class="mid-num">{peor_b["Barrio"]}</div>'
                f'<div class="muted">{peor_b["% Urgente + Alta"]:.1f} % · {int(peor_b["Urgentes"] + peor_b["Altas"]):,} de {int(peor_b["Tramos"]):,} tramos</div></div>',
                unsafe_allow_html=True)
else:
    k3.markdown('<div class="card"><div class="muted">Barrios</div><div class="muted">Falta la geometría (ver sección Geovisor).</div></div>',
                unsafe_allow_html=True)

tab_l, tab_b = st.tabs(["Por localidad", "Por barrio"])
with tab_l:
    st.altair_chart(grafico_ranking(t_loc, "Localidad"), width="stretch")
    st.dataframe(tabla_territorio_estilizada(t_loc, "Localidad"), hide_index=True, width="stretch")
    st.caption("Color: verde = menos críticos, rojo = más críticos (índice, riesgo medio y puntaje medio).")
    st.download_button("Descargar tabla de localidades (CSV)", df_a_csv(t_loc), "indice_localidades.csv", "text/csv")
with tab_b:
    if not hay_geo:
        st.info("Para agrupar por barrio hace falta la geometría: ejecuta `python -m src.geo_prep` (requiere internet).")
    else:
        bc1, bc2, bc3 = st.columns(3)
        loc_b = bc1.selectbox("Localidad", ["Toda la ciudad"] + sorted(vista_b["Localidad (mapa)"].dropna().unique().tolist()),
                              index=0, key="panel_loc")
        min_tr = bc2.slider("Mínimo de tramos por barrio", 5, 200, 30, help="Evita que barrios muy pequeños dominen el ranking.")
        n_b = bc3.slider("Cuántos barrios mostrar", 5, 40, 15)
        base_b = vista_b.dropna(subset=["Barrio"])
        if loc_b != "Toda la ciudad":
            base_b = base_b[base_b["Localidad (mapa)"] == loc_b]
        t_b = indice_territorio(base_b, "Barrio")
        t_b = t_b[t_b["Tramos"] >= min_tr]
        if t_b.empty:
            st.warning("Ningún barrio cumple el mínimo de tramos con ese filtro.")
        else:
            loc_de = base_b.groupby("Barrio")["Localidad (mapa)"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "—")
            t_b.insert(1, "Localidad", t_b["Barrio"].map(loc_de))
            st.caption(f"{len(t_b):,} barrios cumplen el mínimo; se grafican los {min(n_b, len(t_b))} más críticos.")
            st.altair_chart(grafico_ranking(t_b.head(n_b), "Barrio"), width="stretch")
            st.dataframe(tabla_territorio_estilizada(t_b, "Barrio"), hide_index=True, width="stretch")
            st.caption("Color: verde = menos críticos, rojo = más críticos (índice, riesgo medio y puntaje medio).")
            st.download_button("Descargar tabla de barrios (CSV)", df_a_csv(t_b), "indice_barrios.csv", "text/csv")

# ───────────────────────── top de tramos ─────────────────────────
st.markdown("### Top de tramos a intervenir")
st.caption(f"Ámbito: {loc_sel}. Los primeros son los que el sistema recomienda atender primero.")
top = filtrado.sort_values("score_final", ascending=False).head(top_n).reset_index(drop=True)
top.insert(0, "#", range(1, len(top) + 1))
cols_publico = ["#", "Prioridad", "Puntaje", "Estado de la vía", "Tipo de vía", "Localidad", "Riesgo (%)", "ID del tramo"]
st.dataframe(
    top[cols_publico], width="stretch", hide_index=True,
    column_config={
        "Puntaje": st.column_config.ProgressColumn("Puntaje", min_value=0, max_value=100, format="%.1f"),
        "Riesgo (%)": st.column_config.ProgressColumn("Riesgo de empeorar", min_value=0, max_value=100, format="%d %%"),
    },
)
st.download_button("Descargar este top (CSV)", df_a_csv(top[cols_publico]), "top_tramos.csv", "text/csv")

st.markdown("### Los 3 más urgentes (explicados)")
top3 = top.head(3)
cards = st.columns(len(top3) if len(top3) else 1)
for i, (_, row) in enumerate(top3.iterrows()):
    with cards[i]:
        st.markdown(
            f"""
<div class="card">
  <div class="pill">#{int(row['#'])} · {row['Prioridad']}</div>
  <div class="big-num">{row['Puntaje']}</div>
  <div class="muted">puntaje de prioridad</div>
  <p style="margin-top:0.7rem;color:#0E1A2B">
    <b>{row['Estado de la vía']}</b> · {row['Tipo de vía']}<br/>
    {row['Localidad']}<br/>
    Riesgo de empeorar: <b>{int(row['Riesgo (%)'])} %</b>
  </p>
  <div class="muted">ID {row['ID del tramo']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

# ───────────────────────── geovisor ─────────────────────────
@st.cache_data(show_spinner="Preparando el mapa de toda la ciudad…")
def _html_geovisor(fuente: str, version: float, _scored: pd.DataFrame) -> str:
    return render_html(construir_datos(_scored, embedded=False))


st.markdown("### Geovisor de tramos")
st.caption("Mapa de toda la ciudad. Toca una localidad (o elígela en el panel del mapa) y luego un barrio: el mapa se reencuadra. "
           "Puedes cambiar el mapa base, su transparencia y el grosor de los tramos.")
if not hay_geo:
    st.info("Falta la geometría de los tramos. Ejecuta `python -m src.geo_prep` (requiere internet) y recarga.")
else:
    html = _html_geovisor(fuente, PLANTILLA.stat().st_mtime, scored)
    inicial = {v: k for k, v in LOCALIDADES.items()}.get(loc_sel)
    if inicial is not None:
        html = html.replace('"sel":{"loc":null,"barrio":null}', '"sel":{"loc":%d,"barrio":null}' % inicial, 1)
    components.html(html, height=760, scrolling=False)

# ───────────────────────── resumen en lenguaje natural ─────────────────────────
st.markdown("### Resumen en lenguaje natural")
loc_code = {v: k for k, v in LOCALIDADES.items()}.get(loc_sel)
st.markdown(
    f"**Ámbito:** {loc_sel}. Se revisaron **{n_total:,}** tramos; **{criticos:,}** están en mal estado "
    f"y **{urgentes + alta:,}** ({pct_ua:.1f} %) están en prioridad urgente o alta. "
    "Estos son los tramos que el sistema recomienda atender primero:"
)
lineas = []
for _, r in top.head(min(8, top_n)).iterrows():
    lineas.append(
        f"{int(r['#'])}. **Tramo {r['ID del tramo']}** — {r['Localidad']} · {r['Tipo de vía']} · estado **{r['Estado']}** · "
        f"puntaje **{r['Puntaje']}** · riesgo de empeorar **{int(r['Riesgo (%)'])} %**"
    )
st.markdown("\n".join(lineas))
st.caption("Cada cifra sale directamente de la tabla de arriba; el texto no inventa tramos ni valores. "
           "Limitaciones: el flujo es un proxy por jerarquía (sin Bitcarrier/Waze); no se usan SPT/CPT ni DTM.")
with st.expander("Ver el informe técnico generado (texto original anclado a filas)"):
    st.text(generar_informe(filtrado, localidad=loc_code, top_n=min(8, top_n)))

# ───────────────────────── vista técnica ─────────────────────────
@st.cache_data(show_spinner="Evaluando el modelo sobre el conjunto de prueba…")
def _evaluacion(version: float):
    tabla = pd.read_csv(TRAINING_TABLE)
    return evaluar_modelo(art, tabla, metrics), tasa_por_indest(tabla), tabla


if modo == "Vista técnica":
    st.markdown("### Vista técnica")
    if not TRAINING_TABLE.exists():
        st.warning("Falta data/training_table.csv; no se pueden recalcular los diagnósticos del modelo.")
    else:
        ev, tasa_ind, tabla = _evaluacion(ARTIFACT.stat().st_mtime)
        t_rend, t_modelo, t_datos, t_tabla, t_retrain = st.tabs(
            ["Desempeño", "Qué aprendió el modelo", "Datos y supuestos", "Tabla técnica", "Reentrenar"])

        with t_rend:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("F1 (test)", f"{metrics.get('f1', float('nan')):.3f}", help="Media armónica de precisión y recall sobre datos no vistos.")
            m2.metric("Recall", f"{metrics.get('recall', float('nan')):.3f}", help="De los tramos que sí empeoraron, cuántos detectó.")
            m3.metric("Precisión", f"{metrics.get('precision', float('nan')):.3f}", help="De los tramos que marcó como 'empeora', cuántos realmente empeoraron.")
            m4.metric("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}")
            m5.metric("F1 baseline", f"{metrics.get('baseline_majority_f1', float('nan')):.3f}", help="Predecir siempre 'no empeora' (clase mayoritaria).")
            gap = abs(ev["f1_val"] - ev["f1_test"])
            st.caption(f"Sobreajuste: F1 validación {ev['f1_val']:.3f} vs. test {ev['f1_test']:.3f} (diferencia {gap:.3f}). "
                       f"Partición estratificada 70/15/15: {ev['n_train']:,} / {ev['n_val']:,} / {ev['n_test']:,} tramos. "
                       + ("Los diagnósticos de esta pestaña se recalcularon y coinciden con las métricas guardadas." if ev["coincide_con_metrics_json"]
                          else "Atención: los diagnósticos recalculados no coinciden exactamente con las métricas guardadas."))

            cr1, cr2 = st.columns(2)
            with cr1:
                st.markdown("##### Curva ROC")
                diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[5, 4], color="#9AA6B8").encode(x="x:Q", y="y:Q")
                roc = alt.Chart(ev["roc"]).mark_line(color="#1F78B4", strokeWidth=2.5).encode(
                    x=alt.X("fpr:Q", title="Falsos positivos (tasa)"), y=alt.Y("tpr:Q", title="Verdaderos positivos (tasa)"),
                    tooltip=[alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")])
                st.altair_chart(_cfg((diag + roc).properties(height=280)), width="stretch")
                st.caption(f"AUC = {ev['auc']:.3f}. La línea punteada es un modelo que adivina al azar.")
            with cr2:
                st.markdown("##### Precisión vs. recall según el umbral")
                umbral = st.slider("Umbral de decisión (P ≥ umbral ⇒ 'empeora')", 0.05, 0.95, 0.50, 0.05)
                mu = metricas_en_umbral(ev["y"], ev["p"], umbral)
                curva = ev["pr"].melt(id_vars="umbral", value_vars=["precision", "recall"], var_name="Métrica", value_name="Valor")
                curva["Métrica"] = curva["Métrica"].map({"precision": "Precisión", "recall": "Recall"})
                lineas_pr = alt.Chart(curva).mark_line(strokeWidth=2.5).encode(
                    x=alt.X("umbral:Q", title="Umbral", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Valor:Q", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Métrica:N", scale=alt.Scale(range=["#d7263d", "#2e9e6b"]), legend=alt.Legend(orient="top", title=None)))
                marca = alt.Chart(pd.DataFrame({"u": [umbral]})).mark_rule(color=INK, strokeDash=[4, 3]).encode(x="u:Q")
                st.altair_chart(_cfg((lineas_pr + marca).properties(height=280)), width="stretch")
                st.caption(f"Con umbral {umbral:.2f}: precisión {mu['precision']:.2f}, recall {mu['recall']:.2f}, F1 {mu['f1']:.2f}. "
                           "Subir el umbral reduce falsos positivos pero deja pasar más tramos que sí empeoran.")

            cm1, cm2 = st.columns([1, 1])
            with cm1:
                st.markdown("##### Matriz de confusión (test)")
                cm = pd.DataFrame({"Real": ["No empeora", "No empeora", "Empeora", "Empeora"],
                                   "Predicho": ["No empeora", "Empeora", "No empeora", "Empeora"],
                                   "Tramos": [mu["tn"], mu["fp"], mu["fn"], mu["tp"]]})
                hm = alt.Chart(cm).mark_rect().encode(
                    x=alt.X("Predicho:N", sort=["No empeora", "Empeora"], axis=alt.Axis(labelAngle=0)), y=alt.Y("Real:N", sort=["No empeora", "Empeora"]),
                    color=alt.Color("Tramos:Q", scale=alt.Scale(scheme="blues"), legend=None))
                tx = alt.Chart(cm).mark_text(fontSize=18, fontWeight="bold").encode(
                    x=alt.X("Predicho:N", sort=["No empeora", "Empeora"]), y=alt.Y("Real:N", sort=["No empeora", "Empeora"]),
                    text=alt.Text("Tramos:Q", format=","), color=alt.condition(alt.datum.Tramos > cm["Tramos"].max() * 0.5, alt.value("white"), alt.value(INK)))
                st.altair_chart(_cfg((hm + tx).properties(height=260)), width="stretch")
                st.caption(f"Falsos positivos: {mu['fp']:,} (se revisan sin necesidad). Falsos negativos: {mu['fn']:,} (empeoran sin avisar).")
            with cm2:
                st.markdown("##### Probabilidad estimada por clase real")
                dist = pd.DataFrame({"P": ev["p"], "Real": np.where(ev["y"] == 1, "Empeoró", "No empeoró")})
                h = alt.Chart(dist).mark_bar(opacity=0.75).encode(
                    x=alt.X("P:Q", bin=alt.Bin(step=0.05), title="Probabilidad de empeorar estimada"),
                    y=alt.Y("count():Q", stack=None, title="Tramos"),
                    color=alt.Color("Real:N", scale=alt.Scale(domain=["No empeoró", "Empeoró"], range=["#3B5B8C", "#d7263d"]), legend=alt.Legend(orient="top", title=None)))
                st.altair_chart(_cfg(h.properties(height=260)), width="stretch")
                st.caption("Si el modelo separa bien, las dos distribuciones casi no se superponen.")

        with t_modelo:
            imp = ev["importancias"]
            st.markdown("##### Importancia de las variables (Random Forest)")
            gi = alt.Chart(imp).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#3B5B8C").encode(
                y=alt.Y("Variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=340)), x=alt.X("Importancia:Q", title="Importancia relativa", axis=alt.Axis(format="%")),
                tooltip=["Variable", alt.Tooltip("Importancia:Q", format=".1%")])
            st.altair_chart(_cfg(gi.properties(height=280)), width="stretch")
            cond = imp[imp["Variable"].str.contains("INDEST")]["Importancia"].sum()
            st.markdown(
                f'<div class="hallazgo"><b>Qué significa.</b> Las dos variables de condición (<code>INDEST</code> y <code>C_norm = 100 − INDEST</code>) '
                f"son la misma información y juntas suman <b>{cond:.0%}</b> de la importancia. Jerarquía y flujo (proxy de la jerarquía) aportan cerca de 1 % cada una al modelo; "
                "pesan en el puntaje por las reglas AHP, no por el aprendizaje automático.</div>", unsafe_allow_html=True)

            st.markdown("##### ¿Por qué el ROC-AUC es tan alto? Efecto de los límites de categoría")
            tasa_ind = tasa_ind.copy()
            tasa_ind["Franja"] = np.where(tasa_ind["INDEST"].between(71, 73) | tasa_ind["INDEST"].between(86, 88), "1–3 pts sobre un límite", "Resto del rango")
            gt = alt.Chart(tasa_ind).mark_bar().encode(
                x=alt.X("INDEST:O", title="INDEST inicial (corte dic-2024)", axis=alt.Axis(labelOverlap=True)),
                y=alt.Y("Tasa:Q", title="% de tramos que empeoran", axis=alt.Axis(format="%")),
                color=alt.Color("Franja:N", scale=alt.Scale(domain=["1–3 pts sobre un límite", "Resto del rango"], range=["#d7263d", "#9AA6B8"]), legend=alt.Legend(orient="top", title=None)),
                tooltip=["INDEST:O", alt.Tooltip("Tasa:Q", format=".1%"), alt.Tooltip("Tramos:Q", format=",")])
            st.altair_chart(_cfg(gt.properties(height=260)), width="stretch")
            en = tabla[tabla["INDEST"].between(71, 73) | tabla["INDEST"].between(86, 88)]
            fuera = tabla.drop(en.index)
            st.markdown(
                f'<div class="alerta"><b>Lectura honesta del desempeño.</b> Los tramos que empeoran pierden en promedio unos 3 puntos de INDEST entre cortes, '
                f"así que los que están 1–3 puntos por encima del límite de una categoría (71–73, 86–88) casi siempre cruzan a la siguiente: "
                f"empeora el <b>{en['empeoro'].mean():.0%}</b> de ellos frente al <b>{fuera['empeoro'].mean():.1%}</b> del resto, y esas franjas reúnen el "
                f"<b>{en['empeoro'].sum() / tabla['empeoro'].sum():.0%}</b> de los positivos. El modelo detecta bien esa señal, que es real pero cercana a una regla de umbrales: "
                "el ROC-AUC no demuestra que anticipe deterioro por tránsito o suelo.</div>", unsafe_allow_html=True)

        with t_datos:
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("##### Pipeline")
                st.markdown(
                    "<ol class='pasos'><li>Dos cortes públicos del IDU (dic-2024 y jun-2026) unidos por <code>CIV</code>.</li>"
                    "<li>Etiqueta <code>empeoro = 1</code> si la categoría AASHTO empeora entre cortes.</li>"
                    "<li>Variables del corte inicial: C_norm, J_norm, F_norm, INDEST, área, longitud, localidad.</li>"
                    "<li>Random Forest (200 árboles, profundidad 12, <code>class_weight='balanced'</code>) dentro de un Pipeline sklearn.</li>"
                    "<li>Puntaje = 0,7 · AHP + 0,3 · 100 · P(empeora).</li></ol>", unsafe_allow_html=True)
                st.write("**Pesos AHP:**", art.get("ahp_weights"), " **Mezcla del puntaje:**", art.get("score_mix"))
            with d2:
                st.markdown("##### Datos")
                resumen = pd.DataFrame({
                    "Indicador": ["Filas del join (modelo)", "Tasa de positivos (empeoran)", "Tramos train / val / test", "Tramos puntuados en la app"],
                    "Valor": [f"{len(tabla):,}", f"{metrics.get('positive_rate', float('nan')):.1%}",
                              f"{ev['n_train']:,} / {ev['n_val']:,} / {ev['n_test']:,}", f"{len(vista):,}"]})
                st.dataframe(resumen, hide_index=True, width="stretch")
                st.markdown("##### Limitaciones")
                st.markdown("- Flujo: proxy por jerarquía (sin Bitcarrier/Waze vigentes).\n- Subsidencia (DTM) descartada por precisión vertical (0,15 m); sin capa pública SPT/CPT.\n"
                            "- Itagüí: un solo corte, no permite construir la etiqueta.\n- Precisión 0,58: el ranking requiere revisión humana.")
            with st.expander("Métricas completas (JSON y reporte de clasificación)"):
                st.json({k: v for k, v in metrics.items() if k != "report"})
                if "report" in metrics:
                    st.code(metrics["report"])

        with t_tabla:
            tech_cols = [c for c in ["segment_id", "ESTAASHTO", "TIPOMALLAEST", "INDEST", "C_norm", "J_norm", "F_norm", "P_empeora",
                                     "score_ahp", "score_final", "loce_code"] if c in filtrado.columns]
            top_t = filtrado.sort_values("score_final", ascending=False).head(top_n)[tech_cols]
            st.caption(f"Top {len(top_t)} de {loc_sel} con las variables del modelo.")
            st.dataframe(
                top_t, hide_index=True, width="stretch",
                column_config={
                    "P_empeora": st.column_config.ProgressColumn("P(empeora)", min_value=0, max_value=1, format="%.2f"),
                    "score_ahp": st.column_config.NumberColumn("Score AHP", format="%.1f"),
                    "score_final": st.column_config.ProgressColumn("Score final", min_value=0, max_value=100, format="%.1f"),
                    "C_norm": st.column_config.NumberColumn("C_norm", format="%.0f"),
                    "J_norm": st.column_config.NumberColumn("J_norm", format="%.0f"),
                    "F_norm": st.column_config.NumberColumn("F_norm", format="%.0f"),
                })
            st.download_button("Descargar tabla técnica (CSV)", df_a_csv(top_t), "top_tecnico.csv", "text/csv")

        with t_retrain:
            st.write("Modelo:", Path(ARTIFACT).name)
            retrain_path = st.text_input("Ruta para reentrenar (carpeta con 2 CSV o CSV con columna `empeoro`)")
            st.caption("Atención: reentrenar sobrescribe `models/pipeline_priorizacion.joblib` y las tablas de `data/`.")
            if st.button("Reentrenar") and retrain_path.strip():
                with st.spinner("Reentrenando..."):
                    try:
                        info = reentrenar(retrain_path.strip())
                        st.success(f"OK — filas={info['n_rows']} F1={info['metrics'].get('f1')}")
                    except Exception as e:
                        st.error(str(e))
else:
    with st.expander("¿Quieres ver el detalle técnico del modelo?"):
        st.write(f"Desempeño (F1): **{metrics.get('f1', float('nan')):.2f}**. "
                 "Cambia el modo a *Vista técnica* en la barra lateral para ver curvas, matriz de confusión y qué aprendió el modelo.")

st.caption("Datos: estado de la malla vial de Bogotá (IDU), cortes dic-2024 y jun-2026; barrios y localidades: Catastro Bogotá. "
           "Herramienta académica — no reemplaza la decisión oficial de IDU/UAERMV.")
