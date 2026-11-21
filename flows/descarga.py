"""Descarga de las fuentes complementarias desde las API de datos abiertos.

Deja los archivos tal como los entrega cada portal en la carpeta de origen
(DENGUE_SOURCE_DIR); desde ahí la ingesta los lleva a Bronze como cualquier otra fuente.

- IDEAM (Datos Abiertos Colombia, API Socrata): lluvia (s54a-sgyg) y temperatura del aire
  (sbwg-7ju4) de las estaciones automáticas de Cali, mes a mes, para todo el periodo.
- DANE: proyecciones oficiales de población municipal por área 2018-2042 (actualización
  posterior a la pandemia), que reemplazan la extrapolación lineal de la población.

Ejecución: make descargar  (o python -m flows.descarga). Necesita conexión a internet.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flows.common.config import settings
from flows.common.logging import get_logger

log = get_logger(__name__)

API_DATOS_GOV = "https://www.datos.gov.co/resource/{dataset}.csv"
IDEAM_DATASETS = {"precipitacion": "s54a-sgyg", "temperatura": "sbwg-7ju4"}
IDEAM_COLUMNAS = ["codigoestacion", "codigosensor", "fechaobservacion", "valorobservado",
                  "nombreestacion", "departamento", "municipio", "zonahidrografica", "latitud",
                  "longitud", "descripcionsensor", "unidadmedida"]
# Estaciones automáticas del IDEAM en Cali con datos en el periodo (catálogo de estaciones)
ESTACIONES_CALI = {
    "0026055120": "Universidad del Valle - AUT",
    "0026085170": "Base Aérea Marco Fidel Suárez - AUT",
    "0026085160": "Siloé - AUT",
    "0026055100": "Farallones - AUT",
}
NOMBRE_IDEAM = "ideam_{variable}_cali_{inicio}-{fin}.csv"
DANE_URL = ("https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/"
            "PPED-AreaMun-2018-2042_VP.xlsx")
NOMBRE_DANE = "PPED-AreaMun-2018-2042_VP.xlsx"
PAGINA = 50_000                     # filas por consulta a la API
REINTENTOS = 4
TIMEOUT_S = 120
HILOS = 6                           # consultas simultáneas a la API


def _get(url: str, timeout: int = TIMEOUT_S) -> bytes:
    """GET con reintentos y espera creciente (la API pública a veces corta la conexión)."""
    for intento in range(1, REINTENTOS + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as respuesta:
                return respuesta.read()
        except OSError as error:
            if intento == REINTENTOS:
                raise
            log.warning("Fallo de red (%s); reintento %d de %d", error, intento, REINTENTOS - 1)
            time.sleep(2 ** intento)
    raise RuntimeError("inalcanzable")


def url_socrata(dataset: str, estacion: str, desde: str, hasta: str, offset: int = 0,
                limite: int = PAGINA) -> str:
    """Consulta SoQL de las lecturas de una estación entre dos fechas (hasta excluido)."""
    parametros = {
        "$select": ",".join(IDEAM_COLUMNAS),
        "$where": (f"codigoestacion='{estacion}' AND fechaobservacion >= '{desde}T00:00:00' "
                   f"AND fechaobservacion < '{hasta}T00:00:00'"),
        "$order": ":id",                  # orden estable para paginar (rápido en la API)
        "$limit": str(limite),
        "$offset": str(offset),
    }
    return API_DATOS_GOV.format(dataset=dataset) + "?" + urllib.parse.urlencode(parametros)


def meses(anio_inicio: int, anio_fin: int) -> list[tuple[str, str]]:
    """Pares (primer día del mes, primer día del mes siguiente) del periodo."""
    limites = [f"{a}-{m:02d}-01" for a in range(anio_inicio, anio_fin + 1) for m in range(1, 13)]
    limites.append(f"{anio_fin + 1}-01-01")
    return list(zip(limites[:-1], limites[1:], strict=True))


def leer_csv(contenido: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(contenido.decode("utf-8"))))


def descargar_estacion(dataset: str, estacion: str, desde: str, hasta: str,
                       obtener=_get) -> list[dict]:
    """Todas las lecturas de una estación en un intervalo, página por página."""
    filas, offset = [], 0
    while True:
        pagina = leer_csv(obtener(url_socrata(dataset, estacion, desde, hasta, offset)))
        filas.extend(pagina)
        if len(pagina) < PAGINA:
            return filas
        offset += PAGINA


def descargar_ideam(destino: Path, anio_inicio: int, anio_fin: int,
                    estaciones: dict[str, str] = ESTACIONES_CALI, obtener=_get) -> dict:
    """Descarga lluvia y temperatura de las estaciones de Cali; un CSV por variable.

    Devuelve un resumen con las lecturas por variable y estación. Los valores no se
    transforman: el archivo tiene las mismas columnas que la descarga manual del portal.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    resumen = {}
    periodos = meses(anio_inicio, anio_fin)
    for variable, dataset in IDEAM_DATASETS.items():
        filas, por_estacion = [], {}
        for codigo, nombre in estaciones.items():
            def consultar(periodo, ds=dataset, est=codigo):
                return descargar_estacion(ds, est, *periodo, obtener)

            with ThreadPoolExecutor(max_workers=HILOS) as hilos:
                partes = hilos.map(consultar, periodos)
                lecturas = [fila for parte in partes for fila in parte]
            filas.extend(lecturas)
            por_estacion[codigo] = len(lecturas)
            log.info("IDEAM %s, %s: %d lecturas", variable, nombre, len(lecturas))
        ruta = destino / NOMBRE_IDEAM.format(variable=variable, inicio=anio_inicio, fin=anio_fin)
        with ruta.open("w", encoding="utf-8", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=IDEAM_COLUMNAS, quoting=csv.QUOTE_ALL,
                                      extrasaction="ignore")
            escritor.writeheader()
            escritor.writerows(filas)
        resumen[variable] = {"archivo": ruta.name, "dataset": dataset, "lecturas": len(filas),
                             "por_estacion": por_estacion}
    return resumen


def descargar_dane(destino: Path, obtener=_get) -> dict:
    """Descarga el Excel oficial de proyecciones municipales del DANE sin modificarlo."""
    ruta = Path(destino) / NOMBRE_DANE
    contenido = obtener(DANE_URL)
    if not contenido.startswith(b"PK"):                  # un .xlsx es un archivo ZIP
        raise ValueError("La respuesta del DANE no es un archivo Excel")
    ruta.write_bytes(contenido)
    log.info("DANE: %s (%.1f MB)", ruta.name, len(contenido) / 1e6)
    return {"archivo": ruta.name, "url": DANE_URL, "bytes": len(contenido)}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fuente", choices=["todas", "ideam", "dane"], default="todas")
    args = parser.parse_args(argv)
    destino = settings.source_path
    resumen = {}
    if args.fuente in ("todas", "dane"):
        resumen["dane"] = descargar_dane(destino)
    if args.fuente in ("todas", "ideam"):
        resumen["ideam"] = descargar_ideam(destino, settings.anio_inicio, settings.anio_fin)
    log.info("Descarga terminada:\n%s", json.dumps(resumen, indent=2, ensure_ascii=False))
    return resumen


if __name__ == "__main__":
    main()
