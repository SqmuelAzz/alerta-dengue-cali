"""Rutas y nombres de archivos del proyecto (sin rutas absolutas)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figuras"

# --- Bronze: una copia por cada fuente, con nombre estable ---------------------------
BRONZE_SIVIGILA_2022 = "sivigila_cali_dengue_2022.csv"
BRONZE_SIVIGILA_2023 = "sivigila_cali_dengue_2023.csv"
BRONZE_GRAVE_2023 = "sivigila_cali_dengue_grave_2023.csv"
BRONZE_MORTALIDAD_2023 = "sivigila_cali_mortalidad_2023.csv"
BRONZE_INS_TEMPLATE = "ins_dengue_cali_{anio}.csv"       # un archivo por año del INS
BRONZE_POBLACION = "poblacion_cali_comunas.csv"
BRONZE_CENTINELA = "centinela_hospitalizaciones.csv"
BRONZE_PRECIPITACION = "ideam_precipitacion.csv"
BRONZE_TEMPERATURA = "ideam_temperatura.csv"
BRONZE_PRECIPITACION_API = "ideam_precipitacion_api.csv"
BRONZE_TEMPERATURA_API = "ideam_temperatura_api.csv"
BRONZE_POBLACION_DANE = "dane_poblacion_cali.csv"
BRONZE_ESTACIONES = "ideam_catalogo_estaciones.csv"
BRONZE_METADATA_NAME = "_ingestion_metadata.json"
BRONZE_QUALITY_NAME = "_quality_report.json"

# --- Silver: tablas limpias, cada una con su grano -----------------------------------
SILVER_CASOS = "casos_dengue.parquet"                # un registro por caso notificado
SILVER_SERIE = "serie_semanal.parquet"               # una fila por semana epidemiológica
SILVER_POBLACION = "poblacion.parquet"               # territorio x año
SILVER_CLIMA = "clima_semanal.parquet"               # estación x semana epidemiológica
SILVER_CENTINELA = "centinela_comunas.parquet"       # comuna x año
SILVER_CLEANING_NAME = "_resumen_limpieza.json"      # qué se descartó y por qué

# --- Gold: resultados que escribe la etapa de modelado --------------------------------
GOLD_INDICADORES_COMUNAS = "indicadores_comunas.parquet"
GOLD_PRONOSTICO = "pronostico_semanal.parquet"
GOLD_EVALUACION_PRONOSTICO = "evaluacion_pronostico.parquet"
GOLD_PERFILES_COMUNAS = "perfiles_comunas.parquet"

# --- Modelos ---------------------------------------------------------------------------
MODELO_RIESGO = "dengue_riesgo_alarma_v1"
MODELO_HOSPITALIZACION = "dengue_hospitalizacion_v1"
MODELO_PRONOSTICO = "dengue_pronostico_semanal_v1"
MODELO_COMUNAS = "dengue_perfiles_comunas_v1"
