"""Agregaciones: Silver -> Gold.

Cada función responde una pregunta del comité de vigilancia epidemiológica y devuelve
una tabla lista para un informe o un tablero:

- ¿Cuántos casos y qué incidencia tuvo la ciudad cada año?            incidencia_anual
- ¿Cuál es el comportamiento esperado de cada semana?                 canal_endemico
- ¿En qué zona del canal está cada semana? ¿Es inusual?               situacion_semanal
- ¿En qué época del año sube el dengue?                               estacionalidad
- ¿Cuánto se parece una semana a las anteriores?                      autocorrelacion
- ¿A quiénes afecta?                              perfil_demografico, casos_por_estrato,
                                                  casos_por_afiliacion
- ¿Qué comunas concentran la carga?                                   indicadores_comunas
- ¿Qué grupos tienen mayor severidad?                                 severidad_por_grupo
- ¿Cuánto tarda la gente en consultar y el sistema en notificar?      oportunidad_*
- ¿Hay registros inconsistentes?                                      calidad_datos
- ¿Se relaciona el clima con los casos?                               clima_y_casos
- ¿El clima de semanas anteriores anticipa el crecimiento de casos?   clima_rezagos
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from flows.common.config import settings
from flows.common.epidemiologia import GRUPOS_EDAD, canal_endemico, tasa_por_100k, zona_canal
from flows.common.logging import get_logger
from flows.common.paths import (
    GOLD,
    GOLD_INDICADORES_COMUNAS,
    SILVER,
    SILVER_CASOS,
    SILVER_CENTINELA,
    SILVER_CLEANING_NAME,
    SILVER_POBLACION,
    SILVER_SERIE,
)

log = get_logger(__name__)

TRAMOS_CONSULTA = ["0-1", "2-3", "4-5", "6-7", "8 o más"]     # días desde los síntomas
LIMITES_CONSULTA = [0, 2, 4, 6, 8, np.inf]
DIAS_CONSULTA_TARDIA = 3          # consultar después del tercer día se considera tardío
MAX_REZAGO_AUTOCORRELACION = 12
MIN_CASOS_GRUPO = 30              # grupos más pequeños no entran a las pruebas estadísticas
Z_95 = 1.959964                   # cuantil normal para intervalos de confianza del 95 %


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _clean(table: pd.DataFrame) -> pd.DataFrame:
    """Índice como columna normal y nombres de columna en texto (requisito de Parquet)."""
    table = table.reset_index().rename(columns=str)
    table.columns.name = None
    return table


def wilson_interval(exitos: pd.Series, n: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Intervalo de confianza del 95 % de una proporción (método de Wilson)."""
    n = n.astype("float64")
    p = exitos / n
    centro = (p + Z_95**2 / (2 * n)) / (1 + Z_95**2 / n)
    margen = Z_95 * np.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / (1 + Z_95**2 / n)
    return (centro - margen).clip(0, 1), (centro + margen).clip(0, 1)


def cramers_v(tabla: pd.DataFrame, chi2: float) -> float:
    """Tamaño del efecto de una tabla de contingencia (0 = sin asociación, 1 = total)."""
    n = tabla.to_numpy().sum()
    k = min(tabla.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else float("nan")


def _con_clasificacion(casos: pd.DataFrame) -> pd.DataFrame:
    """Casos con clasificación clínica (solo los publica el Distrito)."""
    con = casos[casos["severo"].notna()].copy()
    con["severo"] = con["severo"].astype(bool)
    return con


# ---------------------------------------------------------------------------
# Estimación: incidencia y canal endémico
# ---------------------------------------------------------------------------

def incidencia_anual(serie: pd.DataFrame, casos: pd.DataFrame) -> pd.DataFrame:
    """Casos, población y tasa de incidencia por 100.000 habitantes de cada año.

    'anio_completo' avisa cuando el archivo del año se cortó antes de terminar: su
    total subestima el año real.
    """
    anual = serie.groupby("anio_epi").agg(
        casos=("casos", "sum"),
        hospitalizados=("hospitalizados", "sum"),
        confirmados=("confirmados", "sum"),
        menores_15=("menores_15", "sum"),
        poblacion=("poblacion", "first"),
        semanas=("semana_epi", "size"),
        semanas_con_datos=("cobertura", lambda s: int((s != "sin_datos").sum())),
    ).astype("int64")
    anual["anio_completo"] = anual["semanas"] == anual["semanas_con_datos"]
    anual["tasa_100k"] = tasa_por_100k(anual["casos"], anual["poblacion"])
    for columna in ("hospitalizados", "confirmados", "menores_15"):
        anual[f"pct_{columna}"] = anual[columna] / anual["casos"] * 100
    anual["edad_mediana"] = casos.groupby("anio_epi")["edad"].median()
    anual["fuente"] = casos.groupby("anio_epi")["fuente"].first()
    return _clean(anual)


def canal_endemico_ciudad(serie: pd.DataFrame, anios_base=None,
                          ventana: int | None = None) -> pd.DataFrame:
    """Canal endémico de la ciudad por semana epidemiológica (método de cuartiles).

    Solo usa semanas completas de los años sin epidemia. Con pocos años base, cada
    semana toma también las semanas vecinas (± ventana) para ganar estabilidad.
    """
    anios_base = settings.canal_anios_base if anios_base is None else tuple(anios_base)
    ventana = settings.canal_ventana if ventana is None else ventana
    completas = serie[serie["cobertura"] == "completa"]
    canal = canal_endemico(completas, anios_base, ventana, valor="casos")
    canal["anios_base"] = ", ".join(str(a) for a in anios_base)
    return canal


def situacion_semanal(serie: pd.DataFrame, canal: pd.DataFrame) -> pd.DataFrame:
    """Ubica cada semana en su zona del canal endémico y marca las semanas inusuales.

    'supera_q3' es la señal de anomalía: la semana tuvo más casos que el cuartil 3 del
    canal. Las semanas parciales o sin datos no se clasifican.
    """
    columnas = ["anio_epi", "semana_epi", "fecha_inicio", "fecha_fin", "casos", "casos_modelo",
                "cobertura", "tasa_100k"]
    situacion = serie[columnas].merge(canal[["semana_epi", "q1", "mediana", "q3"]],
                                      on="semana_epi", how="left")
    observado = situacion["casos"].astype("float64").where(situacion["cobertura"] == "completa")
    situacion["zona"] = zona_canal(observado, situacion["q1"], situacion["mediana"],
                                   situacion["q3"])
    situacion["supera_q3"] = (observado > situacion["q3"]).fillna(False).astype(bool)
    situacion["razon_q3"] = observado / situacion["q3"]
    return situacion


# ---------------------------------------------------------------------------
# Descubrimiento: patrones en el tiempo
# ---------------------------------------------------------------------------

def estacionalidad(serie: pd.DataFrame) -> pd.DataFrame:
    """Índice estacional por semana epidemiológica.

    Cada semana se divide por el promedio semanal de su propio año (así un año epidémico
    no domina) y luego se promedia entre años. Un índice de 1,3 significa que esa semana
    suele tener 30 % más casos que una semana promedio del año.
    """
    completas = serie[serie["cobertura"] == "completa"].copy()
    casos = completas["casos"].astype("float64")
    completas["indice"] = casos / casos.groupby(completas["anio_epi"]).transform("mean")
    return _clean(completas.groupby("semana_epi")["indice"].agg(
        indice_medio="mean", indice_min="min", indice_max="max", n_anios="size"))


def autocorrelacion(serie: pd.DataFrame,
                    max_rezago: int = MAX_REZAGO_AUTOCORRELACION) -> pd.DataFrame:
    """Correlación de la serie semanal con ella misma 1, 2, ... semanas atrás."""
    valores = serie["casos_modelo"].astype("float64").reset_index(drop=True)
    filas = [(k, valores.corr(valores.shift(k))) for k in range(1, max_rezago + 1)]
    return pd.DataFrame(filas, columns=["rezago_semanas", "autocorrelacion"])


# ---------------------------------------------------------------------------
# Descubrimiento: población afectada
# ---------------------------------------------------------------------------

def perfil_demografico(casos: pd.DataFrame) -> pd.DataFrame:
    """Casos por grupo de edad y sexo, con el porcentaje del total y de hospitalizados."""
    perfil = casos.groupby(["grupo_edad", "sexo"]).agg(
        casos=("id_caso", "size"), hospitalizados=("hospitalizado", "sum")).reset_index()
    perfil["pct_del_total"] = perfil["casos"] / perfil["casos"].sum() * 100
    perfil["pct_hospitalizados"] = perfil["hospitalizados"] / perfil["casos"] * 100
    orden = perfil["grupo_edad"].map({g: i for i, g in enumerate(GRUPOS_EDAD)})
    return perfil.assign(_orden=orden).sort_values(["_orden", "sexo"]).drop(
        columns="_orden").reset_index(drop=True)


def casos_por_categoria(casos: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Casos por año y categoría de una variable (estrato, afiliación...).

    'pct_anio' se calcula sobre los casos del año que tienen el dato; los casos sin dato
    quedan en la categoría 'Sin dato' para que nada desaparezca del conteo.
    """
    datos = casos.assign(categoria=casos[variable].astype(object).where(
        casos[variable].notna(), "Sin dato").astype(str))
    tabla = datos.groupby(["anio_epi", "categoria"]).agg(
        casos=("id_caso", "size"), hospitalizados=("hospitalizado", "sum")).reset_index()
    con_dato = tabla["casos"].where(tabla["categoria"] != "Sin dato", 0)
    tabla["pct_anio"] = (tabla["casos"] / con_dato.groupby(tabla["anio_epi"]).transform("sum")
                         * 100).where(tabla["categoria"] != "Sin dato")
    tabla["pct_hospitalizados"] = tabla["hospitalizados"] / tabla["casos"] * 100
    return tabla.astype({"anio_epi": int})


def indicadores_comunas(casos: pd.DataFrame, poblacion: pd.DataFrame,
                        centinela: pd.DataFrame) -> pd.DataFrame:
    """Indicadores epidemiológicos y socioeconómicos de cada comuna y corregimiento.

    Usa los años en que la fuente publica la comuna. La tasa es la incidencia acumulada
    del periodo por 100.000 habitantes (casos del periodo / población media). La carga
    histórica es el total de hospitalizaciones por arbovirosis en menores de 5 años del
    sistema centinela por 100.000 habitantes.
    """
    con_comuna = casos[casos["comuna"].notna()]
    anios = sorted(int(a) for a in con_comuna["anio_epi"].unique())
    grupos = con_comuna.groupby("comuna")
    tabla = grupos.agg(
        zona=("zona", "first"),
        casos=("id_caso", "size"),
        hospitalizados=("hospitalizado", "sum"),
        edad_mediana=("edad", "median"),
        estrato_medio=("estrato", "mean"),
    )
    tabla["severos"] = grupos["severo"].sum()
    tabla["con_clasificacion"] = grupos["severo"].count()
    tabla["pct_severos"] = tabla["severos"] / tabla["con_clasificacion"] * 100
    tabla["pct_hospitalizados"] = tabla["hospitalizados"] / tabla["casos"] * 100
    tabla["pct_menores_15"] = grupos["edad"].agg(lambda s: (s < 15).mean() * 100)
    tabla["pct_subsidiado"] = grupos["afiliacion"].agg(
        lambda s: (s == "Subsidiado").mean() * 100)

    territorios = poblacion[poblacion["comuna"].notna()]
    del_periodo = territorios[territorios["anio"].isin(anios)]
    tabla["poblacion"] = del_periodo.groupby("comuna")["poblacion"].mean()
    tabla["territorio"] = territorios.groupby("comuna")["territorio"].first()
    tabla["tasa_100k"] = tasa_por_100k(tabla["casos"], tabla["poblacion"])

    hospitalizaciones = centinela.groupby("comuna")["hosp_arbovirosis"].sum()
    ultima = territorios[territorios["anio"] == centinela["anio"].max()].set_index("comuna")
    tabla["hosp_arbovirosis_100k"] = tasa_por_100k(
        hospitalizaciones.reindex(tabla.index).fillna(0), ultima["poblacion"])
    tabla["periodo"] = f"{anios[0]}-{anios[-1]}" if anios else ""
    columnas = ["territorio", "zona", "periodo", "casos", "poblacion", "tasa_100k", "severos",
                "con_clasificacion", "pct_severos", "hospitalizados", "pct_hospitalizados",
                "edad_mediana", "pct_menores_15", "estrato_medio", "pct_subsidiado",
                "hosp_arbovirosis_100k"]
    tabla = _clean(tabla[columnas].sort_index())
    return tabla.astype({"comuna": int, "casos": int, "severos": int, "hospitalizados": int,
                         "con_clasificacion": int, "estrato_medio": "float64"})


def severidad_por_grupo(casos: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de casos severos (con signos de alarma o graves) en cada grupo.

    Incluye el intervalo de confianza del 95 % y el n: un porcentaje sin su n no dice
    cuánta evidencia lo respalda.
    """
    con = _con_clasificacion(casos)
    con["dias_hasta_consulta"] = pd.cut(
        con["dias_sintomas_consulta"].astype("float64"), bins=LIMITES_CONSULTA,
        labels=TRAMOS_CONSULTA, right=False).astype(object)
    con["estrato"] = con["estrato"].astype(object)
    partes = []
    for variable in ("grupo_edad", "sexo", "afiliacion", "estrato", "zona",
                     "dias_hasta_consulta"):
        grupo = con.dropna(subset=[variable]).groupby(variable)["severo"].agg(
            n="size", severos="sum").reset_index().rename(columns={variable: "categoria"})
        grupo.insert(0, "variable", variable)
        grupo["categoria"] = grupo["categoria"].astype(str)
        partes.append(grupo)
    tabla = pd.concat(partes, ignore_index=True)
    tabla["pct_severos"] = tabla["severos"] / tabla["n"] * 100
    inferior, superior = wilson_interval(tabla["severos"], tabla["n"])
    tabla["ic95_inferior"], tabla["ic95_superior"] = inferior * 100, superior * 100
    return tabla.astype({"n": int, "severos": int})


# ---------------------------------------------------------------------------
# Estimación: oportunidad de la consulta y de la notificación
# ---------------------------------------------------------------------------

def _resumen_oportunidad(grupo) -> pd.DataFrame:
    consulta = grupo["dias_sintomas_consulta"]
    notificacion = grupo["dias_consulta_notificacion"]
    return pd.DataFrame({
        "casos": consulta.size(),
        "mediana_dias_consulta": consulta.median(),
        "p75_dias_consulta": consulta.quantile(0.75),
        "pct_consulta_tardia": consulta.agg(
            lambda s: (s > DIAS_CONSULTA_TARDIA).mean() * 100),
        "mediana_dias_notificacion": notificacion.median(),
        "p90_dias_notificacion": notificacion.quantile(0.90),
    })


def oportunidad_anual(casos: pd.DataFrame) -> pd.DataFrame:
    """Días entre síntomas, consulta y notificación de cada año (medianas y percentiles)."""
    datos = casos.astype({"dias_sintomas_consulta": "float64",
                          "dias_consulta_notificacion": "float64"})
    return _clean(_resumen_oportunidad(datos.groupby("anio_epi"))).astype({"anio_epi": int})


def oportunidad_por_grupo(casos: pd.DataFrame) -> pd.DataFrame:
    """La misma oportunidad por grupo de edad, afiliación y estrato (todos los años)."""
    datos = casos.astype({"dias_sintomas_consulta": "float64",
                          "dias_consulta_notificacion": "float64", "estrato": object})
    partes = []
    for variable in ("grupo_edad", "afiliacion", "estrato"):
        grupo = _clean(_resumen_oportunidad(datos.dropna(subset=[variable]).groupby(variable)))
        grupo = grupo.rename(columns={variable: "categoria"})
        grupo.insert(0, "variable", variable)
        grupo["categoria"] = grupo["categoria"].astype(str)
        partes.append(grupo)
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------
# Anomalías: calidad de los registros
# ---------------------------------------------------------------------------

def calidad_datos(casos: pd.DataFrame, serie: pd.DataFrame,
                  limpieza: dict | None = None) -> pd.DataFrame:
    """Inventario de registros anómalos o incompletos, con su número y porcentaje."""
    n = len(casos)
    distrito = casos["fuente"] != "INS"
    filas = [
        ("Casos con más de 30 días entre dos de sus fechas", casos["intervalo_atipico"].sum(), n),
        ("Casos con fechas en orden imposible", casos["fecha_incoherente"].sum(), n),
        ("Casos cuya semana reportada no coincide con el inicio de síntomas",
         casos["semana_inconsistente"].sum(), n),
        ("Casos sin estrato", casos["estrato"].isna().sum(), n),
        ("Casos del Distrito sin comuna", (casos["estado_comuna"] == "Sin dato").sum(),
         distrito.sum()),
        ("Casos del Distrito con código de comuna no válido",
         (casos["estado_comuna"] == "Código no válido").sum(), distrito.sum()),
        ("Casos del Distrito sin clasificación clínica",
         (distrito & casos["severo"].isna()).sum(), distrito.sum()),
        ("Semanas sin datos (archivo cortado antes de fin de año)",
         (serie["cobertura"] == "sin_datos").sum(), len(serie)),
        ("Semanas parciales por rezago de notificación",
         (serie["cobertura"] == "parcial").sum(), len(serie)),
    ]
    if limpieza:
        filas += [
            ("Filas vacías descartadas en la limpieza",
             limpieza["filas_vacias_descartadas"]["total"], None),
            ("Notificaciones duplicadas eliminadas", limpieza["duplicados_eliminados"],
             limpieza["casos_con_datos"]),
            ("Lecturas de clima fuera de rango", limpieza["clima"]["fuera_de_rango"],
             limpieza["clima"]["lecturas"]),
        ]
    tabla = pd.DataFrame(filas, columns=["hallazgo", "registros", "total"])
    tabla["registros"] = tabla["registros"].astype(int)
    tabla["total"] = tabla["total"].astype("float64")
    tabla["pct"] = tabla["registros"] / tabla["total"] * 100
    return tabla


# ---------------------------------------------------------------------------
# Complemento: clima
# ---------------------------------------------------------------------------

def clima_y_casos(serie: pd.DataFrame) -> pd.DataFrame:
    """Semanas que tienen alguna lectura de clima, junto a sus casos."""
    columnas = [c for c in ["anio_epi", "semana_epi", "fecha_inicio", "casos", "lluvia_mm",
                            "temp_media", "temp_anomalia", "cobertura_lluvia", "cobertura_temp",
                            "estaciones_lluvia", "estaciones_temp"] if c in serie]
    con_clima = serie[serie[["lluvia_mm", "temp_media"]].notna().any(axis=1)]
    return con_clima[columnas].reset_index(drop=True)


def clima_rezagos(serie: pd.DataFrame, rezagos=range(0, 13), horizonte: int = 4) -> pd.DataFrame:
    """Correlación entre el clima de hace k semanas y el crecimiento de los casos.

    Para cada semana t se compara la lluvia o la temperatura de la semana t-k con el
    cambio de los casos entre t y t+horizonte (escala logarítmica). Así se ve si el clima
    previo anticipa los aumentos, no solo si coincide con muchos casos. Solo entran las
    semanas observadas (sin interpolar).
    """
    serie = serie.sort_values(["anio_epi", "semana_epi"]).reset_index(drop=True)
    log_casos = np.log1p(serie["casos"].astype("float64"))
    crecimiento = log_casos.shift(-horizonte) - log_casos
    filas = []
    variables = [v for v in ("lluvia_mm", "temp_media", "temp_anomalia") if v in serie]
    for variable in variables:
        for k in rezagos:
            pares = pd.DataFrame({"clima": serie[variable].shift(k),
                                  "crecimiento": crecimiento}).dropna()
            rho, p = (stats.spearmanr(pares["clima"], pares["crecimiento"])
                      if len(pares) >= 10 else (np.nan, np.nan))
            filas.append({"variable": variable, "rezago_semanas": int(k),
                          "rho": float(rho), "p_value": float(p), "n": int(len(pares))})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Pruebas estadísticas
# ---------------------------------------------------------------------------

def chi2_severidad(casos: pd.DataFrame, variable: str) -> dict:
    """Prueba chi-cuadrado de independencia entre la severidad y una variable.

    H0: la proporción de casos severos es la misma en todas las categorías. Devuelve
    también la V de Cramér: con miles de casos casi todo es 'significativo', y lo que
    importa es el tamaño del efecto.
    """
    con = _con_clasificacion(casos).dropna(subset=[variable])
    tamanos = con[variable].value_counts()
    con = con[con[variable].isin(tamanos[tamanos >= MIN_CASOS_GRUPO].index)]
    tabla = pd.crosstab(con[variable], con["severo"])
    if min(tabla.shape) < 2:          # menos de dos grupos o de dos desenlaces: no hay prueba
        return {"chi2": None, "p_value": None, "dof": None, "n": int(len(con)),
                "cramers_v": None}
    chi2, p, gl, _esperados = stats.chi2_contingency(tabla)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(gl), "n": int(len(con)),
            "cramers_v": cramers_v(tabla, chi2)}


def consulta_vs_severidad(casos: pd.DataFrame) -> dict:
    """¿Consultan más tarde los casos severos? Prueba U de Mann-Whitney (no paramétrica)."""
    con = _con_clasificacion(casos)
    dias = con["dias_sintomas_consulta"].astype("float64")
    severos = dias[con["severo"]].dropna()
    no_severos = dias[~con["severo"]].dropna()
    if severos.empty or no_severos.empty:
        u = p = None
    else:
        u, p = (float(v) for v in stats.mannwhitneyu(severos, no_severos,
                                                      alternative="two-sided"))
    return {"u": u, "p_value": p, "n": int(len(con)),
            "mediana_dias_severos": float(severos.median()),
            "mediana_dias_no_severos": float(no_severos.median()),
            "media_dias_severos": float(severos.mean()),
            "media_dias_no_severos": float(no_severos.mean())}


def correlacion_clima(serie: pd.DataFrame, variable: str) -> dict:
    """Correlación de Spearman entre el clima semanal y los casos (con su n)."""
    pares = serie[serie["cobertura"] == "completa"].dropna(subset=[variable])
    if len(pares) < 3 or pares[variable].nunique() < 2:
        return {"rho": None, "p_value": None, "n": int(len(pares))}
    rho, p = stats.spearmanr(pares[variable], pares["casos"].astype("float64"))
    return {"rho": float(rho), "p_value": float(p), "n": int(len(pares))}


def pruebas_estadisticas(casos: pd.DataFrame, serie: pd.DataFrame) -> dict:
    """Reúne las pruebas del análisis exploratorio en un solo resultado serializable."""
    return {
        "severidad_vs_afiliacion": chi2_severidad(casos, "afiliacion"),
        "severidad_vs_grupo_edad": chi2_severidad(casos, "grupo_edad"),
        "severidad_vs_sexo": chi2_severidad(casos, "sexo"),
        "severidad_vs_estrato": chi2_severidad(casos, "estrato"),
        "dias_consulta_vs_severidad": consulta_vs_severidad(casos),
        "lluvia_vs_casos": correlacion_clima(serie, "lluvia_mm"),
        "temperatura_vs_casos": correlacion_clima(serie, "temp_media"),
        **{f"{variable}_previa_vs_crecimiento": _mas_fuerte(rezagos, variable)
           for rezagos in [clima_rezagos(serie)]
           for variable in ("lluvia_mm", "temp_media", "temp_anomalia")},
    }


def _mas_fuerte(rezagos: pd.DataFrame, variable: str) -> dict:
    """El rezago con la correlación más fuerte (en valor absoluto) de una variable."""
    datos = rezagos[(rezagos["variable"] == variable) & rezagos["rho"].notna()]
    if datos.empty:
        return {"rezago_semanas": None, "rho": None, "p_value": None, "n": 0}
    fila = datos.loc[datos["rho"].abs().idxmax()]
    return {"rezago_semanas": int(fila["rezago_semanas"]), "rho": float(fila["rho"]),
            "p_value": float(fila["p_value"]), "n": int(fila["n"])}


def situacion_actual(situacion: pd.DataFrame) -> dict:
    """Resumen de la última semana completa: lo primero que mira el comité de vigilancia."""
    ultima = situacion[situacion["cobertura"] == "completa"].iloc[-1]
    return {
        "anio_epi": int(ultima["anio_epi"]),
        "semana_epi": int(ultima["semana_epi"]),
        "fecha_inicio": str(pd.Timestamp(ultima["fecha_inicio"]).date()),
        "casos": int(ultima["casos"]),
        "tasa_100k": float(ultima["tasa_100k"]),
        "canal_q1": float(ultima["q1"]),
        "canal_mediana": float(ultima["mediana"]),
        "canal_q3": float(ultima["q3"]),
        "zona": ultima["zona"],
    }


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def build_gold(casos: pd.DataFrame, serie: pd.DataFrame, poblacion: pd.DataFrame,
               centinela: pd.DataFrame, limpieza: dict | None = None) -> dict[str, pd.DataFrame]:
    """Todas las tablas Gold, cada una con el nombre de su archivo (sin extensión)."""
    canal = canal_endemico_ciudad(serie)
    return {
        "incidencia_anual": incidencia_anual(serie, casos),
        "canal_endemico": canal,
        "situacion_semanal": situacion_semanal(serie, canal),
        "estacionalidad": estacionalidad(serie),
        "autocorrelacion": autocorrelacion(serie),
        "perfil_demografico": perfil_demografico(casos),
        "casos_por_estrato": casos_por_categoria(casos, "estrato"),
        "casos_por_afiliacion": casos_por_categoria(casos, "afiliacion"),
        Path(GOLD_INDICADORES_COMUNAS).stem: indicadores_comunas(casos, poblacion, centinela),
        "severidad_por_grupo": severidad_por_grupo(casos),
        "oportunidad_anual": oportunidad_anual(casos),
        "oportunidad_por_grupo": oportunidad_por_grupo(casos),
        "calidad_datos": calidad_datos(casos, serie, limpieza),
        "clima_y_casos": clima_y_casos(serie),
        "clima_rezagos": clima_rezagos(serie),
    }


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    in_dir = Path(in_dir or SILVER)
    casos = pd.read_parquet(in_dir / SILVER_CASOS)
    serie = pd.read_parquet(in_dir / SILVER_SERIE)
    poblacion = pd.read_parquet(in_dir / SILVER_POBLACION)
    centinela = pd.read_parquet(in_dir / SILVER_CENTINELA)
    limpieza_path = in_dir / SILVER_CLEANING_NAME
    limpieza = (json.loads(limpieza_path.read_text(encoding="utf-8"))
                if limpieza_path.exists() else None)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_gold(casos, serie, poblacion, centinela, limpieza)
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "pruebas_estadisticas.json").write_text(
        json.dumps(pruebas_estadisticas(casos, serie), indent=2, ensure_ascii=False),
        encoding="utf-8")
    (out_dir / "situacion_actual.json").write_text(
        json.dumps(situacion_actual(tables["situacion_semanal"]), indent=2, ensure_ascii=False),
        encoding="utf-8")
    log.info("Gold listo: %d tablas y 2 resúmenes en %s", len(tables), out_dir)
    return out_dir
