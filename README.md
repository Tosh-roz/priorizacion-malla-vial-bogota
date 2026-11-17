# Priorización inteligente de la malla vial de Bogotá

App web (Streamlit) que prioriza tramos viales con reglas AHP + Random Forest, con panel de territorios críticos y geovisor por localidad y barrio.
Datos públicos: IDU (estado de la malla vial, cortes dic-2024 y jun-2026) y Catastro Bogotá (barrios y localidades). Proyecto académico, Universidad de los Andes.

## Ejecutar en local
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Publicar (Streamlit Community Cloud)
1. Sube esta carpeta a un repositorio de GitHub (el archivo principal es `app.py`, en la raíz).
2. Entra a https://share.streamlit.io, inicia sesión con GitHub y elige **Create app**.
3. Selecciona el repositorio, la rama y `app.py`, y despliega. El entorno se arma con `requirements.txt`.

La app necesita internet solo para el mapa base del geovisor (OpenStreetMap / Esri).

## Estructura
- `app.py`: interfaz. `src/`: datos, features, modelo, evaluación y geovisor. `geovisor/plantilla.html`: mapa.
- `models/`: modelo entrenado. `data/`: tablas puntuadas y geometría (`data/geo/`).
- `requirements-dev.txt`: notebook, reentrenamiento y preparación de geometría (`python -m src.geo_prep`).
