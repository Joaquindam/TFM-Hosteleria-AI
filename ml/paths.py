from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"
SILVER_SNAP = DATA_DIR / "silver" / "snapshots"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FORECASTING_RESULTS_DIR = RESULTS_DIR / "forecasting_final"
TICKET_RESULTS_DIR = RESULTS_DIR / "ticket_medio_final"

REVENUE_MODEL_PATH = MODELS_DIR / "mejor_modelo_facturacion_diaria.joblib"
TICKET_MODEL_PATH = MODELS_DIR / "mejor_modelo_ticket_medio.joblib"
REVENUE_MODEL_LARGO_PLAZO_PATH = MODELS_DIR / "mejor_modelo_facturacion_diaria_largo_plazo.joblib"
TICKET_MODEL_LARGO_PLAZO_PATH = MODELS_DIR / "mejor_modelo_ticket_medio_largo_plazo.joblib"
SALES_MODEL_PATH = MODELS_DIR / "modelo_prediccion_ventas_articulo.joblib"
NOSHOW_MODEL_PATH = MODELS_DIR / "modelo_prediccion_noshows.joblib"

GOLD_MASTER_TABLE = GOLD_DIR / "tabla_maestra_diaria.parquet"
SALES_PANEL_TABLE = GOLD_DIR / "panel_prediccion_articulos.parquet"
RESERVAS_SILVER = SILVER_SNAP / "reservas_silver.parquet"
FESTIVOS_SILVER = SILVER_SNAP / "festivos_silver.parquet"
TICKETS_SILVER = SILVER_SNAP / "tickets_silver.parquet"
METEO_DIARIA_SILVER = SILVER_SNAP / "meteo_diaria_silver.parquet"
