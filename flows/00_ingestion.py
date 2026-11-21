"""Ingesta: archivos descargados de los portales de datos abiertos -> capa Bronze.

Bronze conserva los valores tal como llegan:

- Los CSV (SIVIGILA de Cali, población, centinela e IDEAM) se copian byte a byte.
- Los Excel de microdatos del INS son nacionales (más de 550.000 filas). Se recorren
  fila por fila y se conservan, sin alterar ningún valor, solo las filas cuya
  procedencia es el municipio de análisis.
- El Excel de proyecciones municipales del DANE se filtra igual: solo las filas del
  municipio de análisis.

Las fuentes complementarias que trae `make descargar` (series completas del IDEAM y
proyecciones del DANE) son opcionales: si no están, el pipeline sigue con las demás.

Los metadatos de la ingesta van en un archivo aparte (_ingestion_metadata.json) y el
diagnóstico de calidad en _quality_report.json; nunca se alteran los datos.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_CENTINELA,
    BRONZE_ESTACIONES,
    BRONZE_GRAVE_2023,
    BRONZE_INS_TEMPLATE,
    BRONZE_METADATA_NAME,
    BRONZE_MORTALIDAD_2023,
    BRONZE_POBLACION,
    BRONZE_POBLACION_DANE,
    BRONZE_PRECIPITACION,
    BRONZE_PRECIPITACION_API,
    BRONZE_QUALITY_NAME,
    BRONZE_SIVIGILA_2022,
    BRONZE_SIVIGILA_2023,
    BRONZE_TEMPERATURA,
    BRONZE_TEMPERATURA_API,
    ROOT,
)

log = get_logger(__name__)

PORTAL_CALI = "Datos Abiertos Cali (datos.cali.gov.co)"
PORTAL_INS = "Portal SIVIGILA - INS (portalsivigila.ins.gov.co)"
PORTAL_IDEAM = "IDEAM - Datos Abiertos Colombia (datos.gov.co)"
PORTAL_IDEAM_API = "IDEAM - API de Datos Abiertos Colombia (make descargar)"
PORTAL_DANE = "DANE - Proyecciones de población municipal 2018-2042 (dane.gov.co)"

INS_YEARS = (2020, 2021, 2024, 2025)      # años que el portal del Distrito no publica
INS_COL_DEPARTAMENTO = "COD_DPTO_O"       # departamento de procedencia del caso
INS_COL_MUNICIPIO = "COD_MUN_O"           # municipio de procedencia del caso
DANE_COL_MUNICIPIO = "MPIO"               # código DIVIPOLA del municipio (5 dígitos)
DANE_HOJA = "PobMunicipal"                # hoja con los datos (inicio del nombre)


@dataclass(frozen=True)
class Source:
    """Una fuente: cómo se llama el archivo descargado y cómo queda en Bronze."""

    key: str
    pattern: str              # patrón del nombre del archivo en la carpeta de origen
    bronze_name: str
    origin: str
    kind: str = "csv"         # "csv": copia exacta | "ins_excel" / "dane_excel": filtro
    sep: str = ","
    encoding: str = "utf-8"
    required: bool = True     # las fuentes complementarias pueden faltar


SOURCES: tuple[Source, ...] = (
    Source("sivigila_2022", "bd_dengue.csv", BRONZE_SIVIGILA_2022, PORTAL_CALI,
           sep=";", encoding="latin-1"),
    Source("sivigila_2023", "53.-bd_casos-notificados-en-sivigila-dengue-*.csv",
           BRONZE_SIVIGILA_2023, PORTAL_CALI, sep=";", encoding="latin-1"),
    Source("grave_2023", "54.-bd_casos-notificados-en-sivigila-para-dengue_grave-*.csv",
           BRONZE_GRAVE_2023, PORTAL_CALI, sep=";", encoding="latin-1"),
    Source("mortalidad_2023", "55.-bd_casos-notificados-en-sivigila-mortalidad_dengue-*.csv",
           BRONZE_MORTALIDAD_2023, PORTAL_CALI, sep=";", encoding="latin-1"),
    *(
        Source(f"ins_{anio}", f"Datos_{anio}_210.xlsx", BRONZE_INS_TEMPLATE.format(anio=anio),
               PORTAL_INS, kind="ins_excel")
        for anio in INS_YEARS
    ),
    Source("poblacion", "proyecciones_de_poblaci*n_de_cali_por_comuna*.csv", BRONZE_POBLACION,
           PORTAL_CALI),
    Source("centinela", "14.-unificado-centinela-*.csv", BRONZE_CENTINELA, PORTAL_CALI,
           sep=";", encoding="latin-1"),
    Source("precipitacion", "s54a-sgyg.csv", BRONZE_PRECIPITACION, PORTAL_IDEAM),
    Source("temperatura", "sbwg-7ju4.csv", BRONZE_TEMPERATURA, PORTAL_IDEAM),
    Source("estaciones", "Cat*logo_Nacional_de_Estaciones_del_IDEAM.csv", BRONZE_ESTACIONES,
           PORTAL_IDEAM),
    Source("precipitacion_api", "ideam_precipitacion_cali_*.csv", BRONZE_PRECIPITACION_API,
           PORTAL_IDEAM_API, required=False),
    Source("temperatura_api", "ideam_temperatura_cali_*.csv", BRONZE_TEMPERATURA_API,
           PORTAL_IDEAM_API, required=False),
    Source("poblacion_dane", "PPED-AreaMun-*.xlsx", BRONZE_POBLACION_DANE, PORTAL_DANE,
           kind="dane_excel", required=False),
)


def find_source(source_dir: Path, pattern: str) -> Path:
    """Ubica el archivo de una fuente dentro de la carpeta de origen."""
    matches = sorted(Path(source_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No se encontró '{pattern}' en {source_dir}. Revisa DENGUE_SOURCE_DIR en .env "
            "y la lista de archivos en docs/INSTALACION.md."
        )
    return matches[0]


def read_source(path: Path, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """Lee un CSV completo como texto, sin transformar ni interpretar valores."""
    return pd.read_csv(path, sep=sep, encoding=encoding, dtype=str, keep_default_na=False)


def _indice_hoja(nombres: list[str], hoja: str | None) -> int:
    """Primera hoja cuyo nombre empieza por 'hoja' (o la primera hoja si no se indica)."""
    if hoja is None:
        return 0
    for i, nombre in enumerate(nombres):
        if nombre.startswith(hoja):
            return i
    raise ValueError(f"El Excel no tiene una hoja que empiece por '{hoja}': {nombres}")


def iter_excel_rows(path: Path, hoja: str | None = None) -> Iterator[list]:
    """Recorre las filas de una hoja de un Excel (por defecto la primera).

    Los microdatos del INS guardan cada celda como una fórmula de texto (="valor"); se lee
    el valor ya calculado. Usa python-calamine (rápido) y, si no está instalado, openpyxl.
    """
    try:
        from python_calamine import CalamineWorkbook
    except ImportError:
        from openpyxl import load_workbook

        log.warning("python-calamine no está instalado: se usa openpyxl (más lento)")
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            hoja_excel = workbook.worksheets[_indice_hoja(workbook.sheetnames, hoja)]
            for row in hoja_excel.iter_rows(values_only=True):
                yield list(row)
        finally:
            workbook.close()
        return

    libro = CalamineWorkbook.from_path(str(path))
    sheet = libro.get_sheet_by_index(_indice_hoja(libro.sheet_names, hoja))
    yield from (sheet.iter_rows() if hasattr(sheet, "iter_rows") else sheet.to_python())


def _as_text(value) -> str:
    return "" if value is None else str(value)


def split_procedencia(rows: Iterable[list], cod_departamento: str,
                      cod_municipio: str) -> tuple[list[str], list[list[str]], int]:
    """Separa las filas con procedencia en el municipio de análisis.

    Devuelve (encabezado, filas conservadas, total de filas leídas). Las celdas se
    conservan como texto, sin recortar espacios ni cambiar formatos.
    """
    rows = iter(rows)
    header = [_as_text(cell).strip() for cell in next(rows)]
    i_dpto, i_mun = header.index(INS_COL_DEPARTAMENTO), header.index(INS_COL_MUNICIPIO)
    kept, total = [], 0
    for row in rows:
        total += 1
        if (_as_text(row[i_dpto]).strip() == cod_departamento
                and _as_text(row[i_mun]).strip() == cod_municipio):
            kept.append([_as_text(cell) for cell in row])
    return header, kept, total


def split_municipio_dane(rows: Iterable[list], cod_mpio: str
                         ) -> tuple[list[str], list[list[str]], int]:
    """Filas del municipio en el Excel del DANE, que trae títulos antes del encabezado.

    El encabezado es la primera fila que contiene la columna MPIO. Devuelve (encabezado,
    filas conservadas, total de filas de datos leídas).
    """
    rows = iter(rows)
    for row in rows:
        header = [_as_text(cell).strip() for cell in row]
        if DANE_COL_MUNICIPIO in header:
            break
    else:
        raise ValueError(f"El Excel del DANE no tiene la columna {DANE_COL_MUNICIPIO}")
    i_mpio = header.index(DANE_COL_MUNICIPIO)
    kept, total = [], 0
    for row in rows:
        if not any(_as_text(cell).strip() for cell in row):
            continue
        total += 1
        if _as_text(row[i_mpio]).strip() == cod_mpio:
            kept.append([_as_text(cell) for cell in row])
    return header, kept, total


def _kind_of(values: pd.Series) -> str:
    """Tipo aparente de una columna leída como texto: vacía, número o texto."""
    filled = values[values.str.strip() != ""]
    if filled.empty:
        return "vacía"
    return "número" if pd.to_numeric(filled, errors="coerce").notna().all() else "texto"


def quality_report(df: pd.DataFrame) -> dict:
    """Resumen de calidad: dimensiones, filas vacías, tipos y faltantes por columna.

    Se considera faltante un valor nulo o un texto en blanco. Devuelve un dict con:
    n_rows, n_cols, empty_rows, dtypes ({col: tipo}), nulls ({col: int}) y
    null_pct ({col: float 0-100}).
    """
    n_rows, n_cols = df.shape
    text = df.astype("string").fillna("")
    blank = text.apply(lambda column: column.str.strip() == "")
    nulls = blank.sum()
    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "empty_rows": int(blank.all(axis=1).sum()) if n_cols else 0,
        "dtypes": {col: _kind_of(text[col]) for col in df.columns},
        "nulls": {col: int(n) for col, n in nulls.items()},
        "null_pct": {
            col: (float(n) / n_rows * 100) if n_rows else 0.0 for col, n in nulls.items()
        },
    }


def _ingest_csv(source: Source, path: Path, out_dir: Path) -> dict:
    """Copia exacta del archivo: Bronze y la fuente tienen la misma huella SHA-256."""
    bronze_path = out_dir / source.bronze_name
    shutil.copyfile(path, bronze_path)
    n_rows = len(read_source(bronze_path, source.sep, source.encoding))
    return {"rows_source": n_rows, "rows_bronze": n_rows}


def _ingest_ins_excel(source: Source, path: Path, out_dir: Path, cod_departamento: str,
                      cod_municipio: str) -> dict:
    """Recorre el Excel nacional y guarda en CSV solo las filas del municipio."""
    header, kept, total = split_procedencia(iter_excel_rows(path), cod_departamento,
                                            cod_municipio)
    with (out_dir / source.bronze_name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept)
    return {
        "rows_source": total,
        "rows_bronze": len(kept),
        "filter": f"{INS_COL_DEPARTAMENTO} == '{cod_departamento}' y "
                  f"{INS_COL_MUNICIPIO} == '{cod_municipio}' (procedencia)",
    }


def _ingest_dane_excel(source: Source, path: Path, out_dir: Path, cod_departamento: str,
                       cod_municipio: str) -> dict:
    """Guarda en CSV solo las filas del municipio de las proyecciones del DANE."""
    cod_mpio = f"{cod_departamento}{cod_municipio}"
    header, kept, total = split_municipio_dane(iter_excel_rows(path, DANE_HOJA), cod_mpio)
    with (out_dir / source.bronze_name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept)
    return {"rows_source": total, "rows_bronze": len(kept),
            "filter": f"{DANE_COL_MUNICIPIO} == '{cod_mpio}'"}


def _load_previous(out_dir: Path) -> dict:
    """Metadatos de la ingesta anterior (para no repetir trabajo si la fuente no cambió)."""
    path = out_dir / BRONZE_METADATA_NAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("files", {})
    except (json.JSONDecodeError, AttributeError):
        return {}


def _relative(path: Path) -> str:
    """Ruta relativa a la raíz del proyecto cuando es posible (metadatos portables)."""
    try:
        return str(Path(path).absolute().relative_to(ROOT))
    except ValueError:
        return str(path)


def run(source_dir: Path | None = None, out_dir: Path = BRONZE, force: bool = False,
        cod_departamento: str | None = None, cod_municipio: str | None = None) -> dict:
    """Lleva cada fuente a Bronze y devuelve los metadatos de la ingesta.

    Si una fuente no cambió desde la ingesta anterior (misma huella SHA-256) y su copia
    en Bronze sigue intacta, no se vuelve a procesar, salvo que force=True.
    """
    source_dir = Path(source_dir) if source_dir else settings.source_path
    cod_departamento = cod_departamento or settings.cod_departamento
    cod_municipio = cod_municipio or settings.cod_municipio
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = _load_previous(out_dir)

    files, quality = {}, {}
    for source in SOURCES:
        try:
            path = find_source(source_dir, source.pattern)
        except FileNotFoundError:
            if source.required:
                raise
            log.warning("Fuente complementaria ausente (se omite): %s. Se obtiene con "
                        "'make descargar'.", source.pattern)
            (out_dir / source.bronze_name).unlink(missing_ok=True)
            continue
        sha_source = sha256_of(path)
        bronze_path = out_dir / source.bronze_name
        old = previous.get(source.key, {})
        unchanged = (
            not force
            and bronze_path.exists()
            and old.get("sha256_source") == sha_source
            and old.get("sha256_bronze") == sha256_of(bronze_path)
            and old.get("cod_municipio", cod_municipio) == cod_municipio
            and old.get("cod_departamento", cod_departamento) == cod_departamento
        )
        if unchanged:
            log.info("Sin cambios en la fuente: %s", source.bronze_name)
            counts = {k: old[k] for k in ("rows_source", "rows_bronze", "filter") if k in old}
        elif source.kind == "ins_excel":
            log.info("Leyendo microdatos del INS: %s", path.name)
            counts = _ingest_ins_excel(source, path, out_dir, cod_departamento, cod_municipio)
        elif source.kind == "dane_excel":
            log.info("Leyendo proyecciones del DANE: %s", path.name)
            counts = _ingest_dane_excel(source, path, out_dir, cod_departamento, cod_municipio)
        else:
            log.info("Copiando fuente: %s", path.name)
            counts = _ingest_csv(source, path, out_dir)

        sep, encoding = ((",", "utf-8") if source.kind in ("ins_excel", "dane_excel")
                         else (source.sep, source.encoding))
        bronze = read_source(bronze_path, sep, encoding)
        quality[source.key] = quality_report(bronze)
        files[source.key] = {
            "source_file": path.name,
            "origin": source.origin,
            "bronze_file": source.bronze_name,
            "kind": source.kind,
            "sep": sep,
            "encoding": encoding,
            **counts,
            "columns": list(bronze.columns),
            "sha256_source": sha_source,
            "sha256_bronze": sha256_of(bronze_path),
        }
        if source.kind in ("ins_excel", "dane_excel"):
            files[source.key] |= {"cod_departamento": cod_departamento,
                                  "cod_municipio": cod_municipio}

    metadata = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": _relative(source_dir),
        "n_files": len(files),
        "rows_bronze_total": sum(f["rows_bronze"] for f in files.values()),
        "files": files,
    }
    (out_dir / BRONZE_METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / BRONZE_QUALITY_NAME).write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Bronze listo: %d archivos, %d filas", len(files), metadata["rows_bronze_total"])
    return metadata
