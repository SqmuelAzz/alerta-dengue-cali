"""Herramientas epidemiológicas compartidas por todas las etapas.

- Semana epidemiológica (calendario del SIVIGILA: semanas de domingo a sábado).
- Canal endémico por el método de cuartiles (Bortman, 1999).
- Tasas por 100.000 habitantes.

Son funciones puras: no leen ni escriben archivos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONAS_CANAL = ["Éxito", "Seguridad", "Alerta", "Brote"]   # de menor a mayor riesgo
GRUPOS_EDAD = ["0-4", "5-14", "15-29", "30-44", "45-64", "65+"]
LIMITES_EDAD = [0, 5, 15, 30, 45, 65, np.inf]             # años cumplidos, [inferior, superior)
SEMANAS_POR_ANIO = 52                                     # referencia para la estacionalidad


def _domingo(fechas: pd.Series) -> pd.Series:
    """Domingo con el que empieza la semana de cada fecha."""
    return fechas - pd.to_timedelta((fechas.dt.day_of_week + 1) % 7, unit="D")


def inicio_anio_epidemiologico(anio: int) -> pd.Timestamp:
    """Domingo de la semana 1: la semana (domingo a sábado) que contiene el 4 de enero.

    Equivale a la regla del calendario epidemiológico: la semana 1 es la primera que
    tiene al menos cuatro días del año nuevo.
    """
    cuatro_enero = pd.Timestamp(year=int(anio), month=1, day=4)
    return cuatro_enero - pd.Timedelta(days=(cuatro_enero.day_of_week + 1) % 7)


def semanas_del_anio(anio: int) -> int:
    """Número de semanas epidemiológicas del año (52 o 53)."""
    return (inicio_anio_epidemiologico(anio + 1) - inicio_anio_epidemiologico(anio)).days // 7


def semana_epidemiologica(fechas: pd.Series) -> pd.DataFrame:
    """Devuelve 'anio_epi' y 'semana_epi' de cada fecha (NaT produce valores nulos).

    El año epidemiológico puede diferir del año calendario en los primeros y últimos
    días del año: el 29 de diciembre de 2019 pertenece a la semana 1 de 2020.
    """
    fechas = pd.to_datetime(fechas).dt.normalize()
    domingo = _domingo(fechas)
    anio = (domingo + pd.Timedelta(days=3)).dt.year          # el miércoles decide el año
    cuatro_enero = pd.to_datetime({"year": anio, "month": 1, "day": 4}, errors="coerce")
    semana = (domingo - _domingo(cuatro_enero)).dt.days // 7 + 1
    return pd.DataFrame(
        {"anio_epi": anio.astype("Int64"), "semana_epi": semana.astype("Int64")},
        index=fechas.index,
    )


def siguientes_semanas(anio: int, semana: int, n: int) -> pd.DataFrame:
    """Las 'n' semanas epidemiológicas que siguen a (anio, semana), con su fecha de inicio."""
    inicio = inicio_anio_epidemiologico(anio) + pd.Timedelta(weeks=int(semana) - 1)
    fechas = pd.Series([inicio + pd.Timedelta(weeks=k) for k in range(1, n + 1)])
    semanas = semana_epidemiologica(fechas)
    return pd.DataFrame({"anio_epi": semanas["anio_epi"].astype(int),
                         "semana_epi": semanas["semana_epi"].astype(int),
                         "fecha_inicio": fechas})


def calendario_epidemiologico(anio_inicio: int, anio_fin: int) -> pd.DataFrame:
    """Una fila por semana epidemiológica entre los dos años (incluidos)."""
    filas = []
    for anio in range(anio_inicio, anio_fin + 1):
        inicio = inicio_anio_epidemiologico(anio)
        for semana in range(1, semanas_del_anio(anio) + 1):
            comienzo = inicio + pd.Timedelta(weeks=semana - 1)
            filas.append((anio, semana, comienzo, comienzo + pd.Timedelta(days=6)))
    return pd.DataFrame(filas, columns=["anio_epi", "semana_epi", "fecha_inicio", "fecha_fin"])


def canal_endemico(serie: pd.DataFrame, anios_base, ventana: int = 2,
                   valor: str = "casos") -> pd.DataFrame:
    """Canal endémico por semana epidemiológica con el método de cuartiles.

    Para cada semana reúne los valores observados en los años base dentro de una
    ventana de ± 'ventana' semanas y calcula el cuartil 1, la mediana y el cuartil 3.
    La ventana da estabilidad cuando hay pocos años base; con ventana=0 es el método
    clásico (un valor por año). Las semanas sin dato (NaN) no participan.

    Devuelve una fila por semana (1 a 53) con: q1, mediana, q3 y n_valores.
    """
    base = serie[serie["anio_epi"].isin(list(anios_base)) & serie[valor].notna()]
    posicion = np.minimum(base["semana_epi"].to_numpy(dtype=int), SEMANAS_POR_ANIO)
    valores = base[valor].to_numpy(dtype=float)
    filas = []
    for semana in range(1, SEMANAS_POR_ANIO + 2):
        distancia = np.abs(posicion - min(semana, SEMANAS_POR_ANIO))
        distancia = np.minimum(distancia, SEMANAS_POR_ANIO - distancia)   # diciembre-enero
        vecinos = valores[distancia <= ventana]
        if len(vecinos):
            q1, mediana, q3 = np.quantile(vecinos, [0.25, 0.5, 0.75])
        else:
            q1 = mediana = q3 = np.nan
        filas.append((semana, q1, mediana, q3, len(vecinos)))
    return pd.DataFrame(filas, columns=["semana_epi", "q1", "mediana", "q3", "n_valores"])


def zona_canal(valor: pd.Series, q1: pd.Series, mediana: pd.Series, q3: pd.Series) -> pd.Series:
    """Ubica cada valor en su zona del canal endémico.

    Éxito: por debajo de Q1. Seguridad: entre Q1 y la mediana. Alerta: entre la mediana
    y Q3. Brote: por encima de Q3. Si falta el valor o el canal, la zona queda nula.
    """
    zonas = np.select(
        [valor < q1, valor < mediana, valor <= q3, valor > q3],
        ZONAS_CANAL,
        default="",
    )
    return pd.Series(zonas, index=valor.index).where(lambda s: s != "", other=None)


def grupo_edad(edad: pd.Series) -> pd.Series:
    """Grupo de edad de cada caso (texto); la edad faltante queda nula."""
    grupo = pd.cut(edad, bins=LIMITES_EDAD, labels=GRUPOS_EDAD, right=False)
    return grupo.astype("object").where(grupo.notna(), None)


def tasa_por_100k(casos, poblacion):
    """Casos por cada 100.000 habitantes."""
    return casos / poblacion * 100_000
