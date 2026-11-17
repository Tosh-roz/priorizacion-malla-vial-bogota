"""Rutas y constantes del pipeline (AHP y score final según hoja de ruta)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACT = MODELS_DIR / "pipeline_priorizacion.joblib"
METRICS_JSON = MODELS_DIR / "metrics.json"
TRAINING_TABLE = DATA_DIR / "training_table.csv"
SCORED_TABLE = DATA_DIR / "scored_segments.csv"

# Pesos AHP (hoja de ruta): Condición / Jerarquía / Flujo
W_COND, W_JER, W_FLUJO = 0.625, 0.239, 0.137
W_REGLAS, W_ML = 0.7, 0.3

AASHTO_ORDER = {
    "Bueno": 1,
    "Satisfactorio": 2,
    "Aceptable": 3,
    "Pobre": 4,
    "Muy pobre": 5,
    "Grave": 6,
    "Fallado": 7,
    "Sin dato": None,
}

JERARQUIA_SCORE = {
    "Troncal": 100,
    "Arterial": 100,
    "Intermedia": 70,
    "Local": 40,
    "Rural": 30,
    "Peatonal": 20,
}

FEATURE_COLS = [
    "C_norm",
    "J_norm",
    "F_norm",
    "INDEST",
    "area",
    "length",
    "loce_code",
]

TARGET = "empeoro"

# APIs parametrizadas (ver LINKS_DATOS.md)
API_CORTE_A = (
    "https://services2.arcgis.com/qAnmJmgmnVso8Fzo/arcgis/rest/services/"
    "EstadoMallaVialOD/FeatureServer/0"
)
API_CORTE_B = (
    "https://services2.arcgis.com/qAnmJmgmnVso8Fzo/arcgis/rest/services/"
    "EstadoMallaVialOD555/FeatureServer/2"
)
# Campos comunes a ambos cortes IDU (SUPERFEST no existe en EstadoMallaVialOD555)
API_FIELDS = [
    "OBJECTID",
    "PK_ID_CALZADA",
    "CIV",
    "LOCEST",
    "TIPOMALLAEST",
    "INDEST",
    "ESTAASHTO",
    "Shape__Area",
    "Shape__Length",
]