# SUPUESTOS y diagnóstico de datos

## Diagnóstico del ZIP de Itagüí (5–10 líneas)

1. El ZIP solo contiene `MV_Itagüí_2024_Centro.gdb.rar` + un `.mxd` (mapa ArcMap); no hay CSV/GeoJSON listos.
2. El FileGDB es el inventario/diagnóstico 2024 de la red vial de Itagüí (~4.101 segmentos según metadatos internos).
3. Hay un solo corte temporal (2024). **No hay dos cortes** para construir la etiqueta `empeoro` de la hoja de ruta (Dic24→Jun26).
4. Sí hay QGIS/`ogr2ogr` en la máquina; se exportaron capas a `data/itagui/` (Calzada 4101, Friccion/PCI, Dano, UPCI, etc.).
5. Aun así **no alcanzan solos** para `empeoro`: un solo año 2024 y sin esquema IDU (`INDEST`/`ESTAASHTO` + 2 cortes). Entrenamiento con Bogotá: join **86.580**, tasa≈13,5%.
6. Modelo RF: F1≈0,72 | recall≈0,93 | ROC-AUC≈0,97 | baseline majority F1=0.
7. Itagüí queda como insumos crudos en `data/itagui/`; para `reentrenar` hace falta mapear a columnas canónicas o un segundo corte.
8. DTM fuera; SPT/CPT sin capa pública; flujo como proxy/`F_norm` por jerarquía en el MVP.

## Supuestos de modelado

- Etiqueta: `empeoro = 1` si el orden AASHTO empeora entre cortes unidos por `CIV`/`PK_ID_CALZADA`.
- Score final: `0.7 * score_AHP + 0.3 * 100 * P_empeora` (punto de partida de la hoja de ruta).
- Cliente/usuario demo: planeación de mantenimiento (IDU/equivalente municipal).
- LLM del informe: plantilla determinística + opcional API; sin API key se genera informe rule-based anclado a filas (cumple salida útil; evita dependencia de red).
