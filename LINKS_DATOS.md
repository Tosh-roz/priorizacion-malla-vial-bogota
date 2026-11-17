# LINKS_DATOS — fuentes abiertas usadas / parametrizadas

1. https://services2.arcgis.com/qAnmJmgmnVso8Fzo/arcgis/rest/services/EstadoMallaVialOD/FeatureServer/0 — Estado malla vial urbana Bogotá (corte POT-190 / Dic-2024), ~100.508 registros; campos `CIV`, `INDEST`, `ESTAASHTO`, `TIPOMALLAEST`.
2. https://services2.arcgis.com/qAnmJmgmnVso8Fzo/arcgis/rest/services/EstadoMallaVialOD555/FeatureServer/2 — Estado malla vial urbana Bogotá (corte POT-555 / Jun-2026), ~102.238 registros; mismo esquema para join temporal.
3. https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/Mapa_Referencia/Mapa_Referencia/MapServer/13 — Inventario geométrico malla vial IDECA/Catastro Bogotá (~139k segmentos; jerarquía `MVITCLA`).
4. https://sig.simur.gov.co/arcgis/rest/services/Hosted/V3_Waze_Alerts_Recent/FeatureServer/0 — Alertas Waze SIMUR/SDM (filtro `HAZARD_ON_ROAD_POT_HOLE`); señal ciudadana viva.
5. https://datosabiertos.bogota.gov.co/dataset/estado-superficial-bogota-d-c — Metadatos/CSV “Estado Superficial” IDU (alternativa tabular al FeatureServer).
6. https://geoportal-idu.hub.arcgis.com/ — Geoportal IDU (cortes y tableros oficiales de infraestructura).
7. https://www.datos.gov.co/Transporte/Red-Vial/ie7y-asdn — Red vial nacional INVIAS (escalabilidad a otras ciudades / red nacional).
8. https://itagui.gov.co/transparencia/datos_abiertos — Datos abiertos Alcaldía de Itagüí (punto de partida para exportar CSV municipal compatible).
9. https://sigmi.itagui.gov.co/mapgis/mapa.jsp?aplicacion=1 — SIGMI Itagüí (visor; no descarga directa de la malla completa).
10. https://caminoscomunitarios.invias.gov.co/docs/caja-herramientas/anexos-modulo-tecnico/normas-ensayo/normas_ensayo_100.pdf — Normas INVIAS ensayos in situ (SPT/CPT; referencia del comentario del profe, no dataset).
