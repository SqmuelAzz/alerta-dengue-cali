"""Configuración del proyecto: valores por defecto + .env opcional."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from flows.common.paths import ROOT

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    source_dir: str               # carpeta con los archivos descargados de los portales
    cod_departamento: str         # DIVIPOLA del departamento de procedencia
    cod_municipio: str            # DIVIPOLA del municipio de procedencia
    anio_inicio: int
    anio_fin: int
    semanas_rezago: int           # semanas incompletas antes de la fecha de corte
    canal_anios_base: tuple[int, ...]   # años sin epidemia para el canal endémico
    canal_ventana: int            # media ventana (semanas) para suavizar el canal
    random_state: int
    test_size: float
    cv_folds: int
    anio_validacion: int          # año para elegir el modelo de pronóstico
    anio_prueba_inicio: int       # desde este año los datos son de prueba
    horizonte_semanas: int        # semanas hacia adelante que se pronostican
    n_rezagos: int                # semanas anteriores que usa el pronóstico
    n_grupos_comunas: int         # número de perfiles de comunas (K-means)

    @property
    def source_path(self) -> Path:
        """Ruta de la carpeta de origen: absoluta, o relativa a la raíz del proyecto."""
        path = Path(self.source_dir)
        return path if path.is_absolute() else ROOT / path


def _years(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def load_settings() -> Settings:
    return Settings(
        source_dir=os.getenv("DENGUE_SOURCE_DIR", "Dataset_Dengue"),
        cod_departamento=os.getenv("COD_DEPARTAMENTO", "76"),
        cod_municipio=os.getenv("COD_MUNICIPIO", "001"),
        anio_inicio=int(os.getenv("ANIO_INICIO", "2020")),
        anio_fin=int(os.getenv("ANIO_FIN", "2025")),
        semanas_rezago=int(os.getenv("SEMANAS_REZAGO", "1")),
        canal_anios_base=_years(os.getenv("CANAL_ANIOS_BASE", "2021,2022,2025")),
        canal_ventana=int(os.getenv("CANAL_VENTANA", "2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        cv_folds=int(os.getenv("CV_FOLDS", "5")),
        anio_validacion=int(os.getenv("ANIO_VALIDACION", "2023")),
        anio_prueba_inicio=int(os.getenv("ANIO_PRUEBA_INICIO", "2024")),
        horizonte_semanas=int(os.getenv("HORIZONTE_SEMANAS", "4")),
        n_rezagos=int(os.getenv("N_REZAGOS", "8")),
        n_grupos_comunas=int(os.getenv("N_GRUPOS_COMUNAS", "3")),
    )


settings = load_settings()
