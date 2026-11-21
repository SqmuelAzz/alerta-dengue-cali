"""Lógica compartida por la API (Flask) y la aplicación web (Streamlit).

Carga los modelos y las tablas Gold una sola vez y expone los servicios del
producto: situación y clima de cada año, pronóstico de casos con semáforo, riesgo
de un caso y perfiles de comunas.
La API y el tablero solo traducen estas respuestas a JSON o a pantallas.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from flows.common.paths import (
    GOLD,
    GOLD_INDICADORES_COMUNAS,
    GOLD_PERFILES_COMUNAS,
    MODELO_COMUNAS,
    MODELO_HOSPITALIZACION,
    MODELO_PRONOSTICO,
    MODELO_RIESGO,
    MODELS,
    SILVER,
    SILVER_CLIMA,
    SILVER_SERIE,
)

ml = importlib.import_module("flows.05_ml_models")
ingenieria = importlib.import_module("flows.01_data_engineering")

SEXOS = ("F", "M")
AFILIACIONES = ("Contributivo", "Subsidiado", "Excepción o especial", "No asegurado",
                "Indeterminado")                   # categorías del régimen en Silver
MAX_SEMANAS_NUEVAS = 12
MAX_CASOS_SEMANA = 20_000


class EntradaInvalida(ValueError):
    """Datos de entrada que no cumplen el formato esperado (respuesta 400 en la API)."""


@dataclass
class Recursos:
    """Modelos y tablas que necesita el producto de datos."""
    pronostico: dict
    riesgo: dict
    serie: pd.DataFrame
    perfiles: pd.DataFrame
    indicadores: pd.DataFrame
    situacion: pd.DataFrame
    canal: pd.DataFrame
    hospitalizacion: dict | None = None          # segundo clasificador (opcional)
    anual: pd.DataFrame | None = None            # incidencia anual (Gold)
    metadatos: dict = field(default_factory=dict)
    clima_rezagos: pd.DataFrame | None = None    # clima de semanas previas vs. casos (Gold)
    cambio_sistema: list[dict] = field(default_factory=list)   # cambio de sensores del IDEAM


def cargar_recursos(models_dir: Path = MODELS, silver_dir: Path = SILVER,
                    gold_dir: Path = GOLD) -> Recursos:
    """Lee los tres modelos y las tablas Gold (falla con un mensaje claro si faltan)."""
    models_dir, silver_dir, gold_dir = Path(models_dir), Path(silver_dir), Path(gold_dir)
    faltantes = [p for p in (models_dir / f"{MODELO_PRONOSTICO}.joblib",
                             models_dir / f"{MODELO_RIESGO}.joblib",
                             silver_dir / SILVER_SERIE, gold_dir / GOLD_PERFILES_COMUNAS)
                 if not p.exists()]
    if faltantes:
        raise FileNotFoundError(
            "Faltan archivos del pipeline; ejecute 'make pipeline' primero: "
            + ", ".join(str(p) for p in faltantes))
    pronostico, meta_p = ml.load_model(models_dir / f"{MODELO_PRONOSTICO}.joblib")
    riesgo, meta_r = ml.load_model(models_dir / f"{MODELO_RIESGO}.joblib")
    hospitalizacion, meta_h = None, {}
    if (models_dir / f"{MODELO_HOSPITALIZACION}.joblib").exists():
        hospitalizacion, meta_h = ml.load_model(models_dir / f"{MODELO_HOSPITALIZACION}.joblib")
    anual = None
    if (gold_dir / "incidencia_anual.parquet").exists():
        anual = pd.read_parquet(gold_dir / "incidencia_anual.parquet")
    clima_rezagos = None
    if (gold_dir / "clima_rezagos.parquet").exists():
        clima_rezagos = pd.read_parquet(gold_dir / "clima_rezagos.parquet")
    cambio = []
    if (silver_dir / SILVER_CLIMA).exists():
        cambio = cambios_de_sistema(pd.read_parquet(silver_dir / SILVER_CLIMA))
    meta_c = json.loads((models_dir / f"{MODELO_COMUNAS}.metadata.json").read_text(
        encoding="utf-8"))
    return Recursos(
        pronostico=pronostico, riesgo=riesgo,
        serie=pd.read_parquet(silver_dir / SILVER_SERIE),
        perfiles=pd.read_parquet(gold_dir / GOLD_PERFILES_COMUNAS),
        indicadores=pd.read_parquet(gold_dir / GOLD_INDICADORES_COMUNAS),
        situacion=pd.read_parquet(gold_dir / "situacion_semanal.parquet"),
        canal=pd.read_parquet(gold_dir / "canal_endemico.parquet"),
        hospitalizacion=hospitalizacion, anual=anual,
        clima_rezagos=clima_rezagos, cambio_sistema=cambio,
        metadatos={"pronostico": meta_p, "riesgo": meta_r, "comunas": meta_c,
                   **({"hospitalizacion": meta_h} if meta_h else {})},
    )


# ---------------------------------------------------------------------------
# Pronóstico
# ---------------------------------------------------------------------------

def validar_casos_recientes(casos) -> list[float]:
    """Lista de conteos semanales nuevos: números enteros entre 0 y 20.000."""
    if casos is None:
        return []
    if not isinstance(casos, list | tuple):
        raise EntradaInvalida("'casos_recientes' debe ser una lista de números")
    if len(casos) > MAX_SEMANAS_NUEVAS:
        raise EntradaInvalida(f"Máximo {MAX_SEMANAS_NUEVAS} semanas nuevas por consulta")
    valores = []
    for valor in casos:
        if isinstance(valor, bool) or not isinstance(valor, int | float):
            raise EntradaInvalida("Cada semana debe ser un número de casos")
        if not 0 <= valor <= MAX_CASOS_SEMANA or float(valor) != int(valor):
            raise EntradaInvalida(f"Los casos deben ser enteros entre 0 y {MAX_CASOS_SEMANA}")
        valores.append(float(valor))
    return valores


def pronosticar(recursos: Recursos, casos_recientes=None) -> pd.DataFrame:
    """Próximas 4 semanas; con 'casos_recientes' se agregan antes las semanas nuevas."""
    serie = ml.agregar_semanas(recursos.serie, validar_casos_recientes(casos_recientes))
    return ml.pronosticar(recursos.pronostico, serie)


def historial_y_pronostico(recursos: Recursos, casos_recientes=None,
                           semanas: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Últimas 'semanas' observadas (con el umbral de brote de cada una) y el pronóstico.

    Las semanas agregadas por el usuario quedan marcadas con 'agregada' para
    distinguirlas de las que vienen de los datos oficiales.
    """
    nuevos = validar_casos_recientes(casos_recientes)
    serie = ml.agregar_semanas(recursos.serie, nuevos)
    tabla = ml.pronosticar(recursos.pronostico, serie)
    historial = serie.sort_values(["anio_epi", "semana_epi"]).tail(semanas).copy()
    historial["agregada"] = False
    if nuevos:
        historial.loc[historial.index[-len(nuevos):], "agregada"] = True
    canal = recursos.pronostico["canal"][["semana_epi", "q1", "mediana", "q3"]]
    historial = historial.merge(canal, on="semana_epi", how="left")
    columnas = ["anio_epi", "semana_epi", "fecha_inicio", "casos", "casos_modelo", "imputada",
                "agregada", "q1", "mediana", "q3"]
    return historial[columnas].reset_index(drop=True), tabla


def resumen_anio(recursos: Recursos, anio: int) -> dict:
    """Casos, semanas en brote e incidencia de un año."""
    datos = recursos.situacion[recursos.situacion["anio_epi"] == anio]
    completas = datos[datos["cobertura"] == "completa"]
    resumen = {"anio_epi": int(anio), "casos": int(datos["casos"].sum()),
               "semanas_con_datos": int(len(completas)),
               "semanas_brote": int(completas["supera_q3"].sum()),
               "semana_pico": None, "casos_pico": None, "tasa_100k": None}
    if not completas.empty:
        pico = completas.loc[completas["casos"].idxmax()]
        resumen.update(semana_pico=int(pico["semana_epi"]), casos_pico=int(pico["casos"]))
    if recursos.anual is not None:
        fila = recursos.anual[recursos.anual["anio_epi"] == anio]
        if not fila.empty:
            resumen["tasa_100k"] = float(fila["tasa_100k"].iloc[0])
    return resumen


# ---------------------------------------------------------------------------
# Clima (IDEAM)
# ---------------------------------------------------------------------------

COLUMNAS_CLIMA = ["anio_epi", "semana_epi", "fecha_inicio", "casos", "lluvia_mm", "temp_media",
                  "temp_anomalia", "estaciones_lluvia", "estaciones_temp"]


def cambios_de_sistema(clima: pd.DataFrame,
                       estaciones=tuple(ingenieria.ESTACIONES_URBANAS)) -> list[dict]:
    """Primera semana con el sistema de medición actual en cada estación urbana."""
    actual = clima[clima["codigo_estacion"].astype(str).isin(estaciones)
                   & (clima["sistema"] == "actual") & (clima["n_obs_temp"] > 0)]
    primeras = (actual.sort_values(["anio_epi", "semana_epi"])
                .groupby("codigo_estacion", as_index=False).first())
    return [{"codigo_estacion": str(f["codigo_estacion"]),
             "estacion": ingenieria.ESTACIONES_URBANAS.get(str(f["codigo_estacion"]),
                                                          f["estacion"]),
             "anio_epi": int(f["anio_epi"]), "semana_epi": int(f["semana_epi"])}
            for _, f in primeras.sort_values(["anio_epi", "semana_epi"]).iterrows()]


def clima_anio(recursos: Recursos, anio: int) -> tuple[pd.DataFrame, dict]:
    """Casos, lluvia y temperatura de cada semana de un año, con su resumen."""
    semanas = recursos.serie[recursos.serie["anio_epi"] == anio]
    tabla = semanas.reindex(columns=COLUMNAS_CLIMA).sort_values("semana_epi")
    tabla = tabla.reset_index(drop=True)
    lluvia, temperatura = tabla["lluvia_mm"].dropna(), tabla["temp_media"].dropna()
    cambios = [c for c in recursos.cambio_sistema if c["anio_epi"] == anio]
    resumen = {
        "anio_epi": int(anio), "semanas": int(len(tabla)),
        "semanas_con_lluvia": int(len(lluvia)), "semanas_con_temperatura": int(len(temperatura)),
        "semanas_sin_clima": int(tabla[["lluvia_mm", "temp_media"]].isna().all(axis=1).sum()),
        "lluvia_semanal_media_mm": None if lluvia.empty else round(float(lluvia.mean()), 1),
        "lluvia_semanal_max_mm": None if lluvia.empty else round(float(lluvia.max()), 1),
        "temperatura_media_c": None if temperatura.empty else round(float(temperatura.mean()), 1),
        "cambio_sistema": cambios,
    }
    return tabla, resumen


def relacion_clima(recursos: Recursos) -> dict:
    """La correlación más fuerte entre el clima de semanas previas y el crecimiento de casos.

    Por variable: rezago (semanas), rho de Spearman, valor p y semanas comparadas.
    """
    if recursos.clima_rezagos is None or recursos.clima_rezagos.empty:
        return {}
    resultado = {}
    for variable, grupo in recursos.clima_rezagos.dropna(subset=["rho"]).groupby("variable"):
        fila = grupo.loc[grupo["rho"].abs().idxmax()]
        resultado[variable] = {"rezago_semanas": int(fila["rezago_semanas"]),
                               "rho": round(float(fila["rho"]), 3),
                               "p_value": round(float(fila["p_value"]), 4),
                               "n": int(fila["n"])}
    return resultado


def ultima_semana(recursos: Recursos) -> dict:
    """Situación de la última semana completa frente al canal endémico."""
    completas = recursos.situacion[recursos.situacion["cobertura"] == "completa"]
    fila = completas.sort_values(["anio_epi", "semana_epi"]).iloc[-1]
    return {"anio_epi": int(fila["anio_epi"]), "semana_epi": int(fila["semana_epi"]),
            "fecha_inicio": str(pd.Timestamp(fila["fecha_inicio"]).date()),
            "casos": int(fila["casos"]), "zona": fila["zona"],
            "canal_q1": float(fila["q1"]), "canal_mediana": float(fila["mediana"]),
            "canal_q3": float(fila["q3"])}


# ---------------------------------------------------------------------------
# Riesgo de un caso
# ---------------------------------------------------------------------------

def _booleano(caso: dict, campo: str) -> float:
    valor = caso.get(campo, False)
    if valor in (None, ""):
        return float("nan")
    if not isinstance(valor, bool | int) or valor not in (0, 1):
        raise EntradaInvalida(f"'{campo}' debe ser verdadero o falso")
    return float(valor)


def validar_caso(caso: dict) -> pd.DataFrame:
    """Convierte un caso (formulario o JSON) en la fila que espera el modelo de riesgo.

    Obligatorios: edad (0-110), sexo (F/M) y días desde el inicio de los síntomas (0-30).
    Opcionales: estrato (1-6), comuna (1-22), afiliación, migrante, desplazamiento y los
    síntomas generales (verdadero/falso).
    """
    if not isinstance(caso, dict):
        raise EntradaInvalida("El cuerpo debe ser un objeto JSON con los datos del caso")
    faltan = [c for c in ("edad", "sexo", "dias_sintomas_consulta") if caso.get(c) is None]
    if faltan:
        raise EntradaInvalida(f"Faltan campos obligatorios: {', '.join(faltan)}")
    edad, dias = caso["edad"], caso["dias_sintomas_consulta"]
    if isinstance(edad, bool) or not isinstance(edad, int | float) or not 0 <= edad <= 110:
        raise EntradaInvalida("'edad' debe ser un número entre 0 y 110")
    if isinstance(dias, bool) or not isinstance(dias, int) or not 0 <= dias <= 30:
        raise EntradaInvalida("'dias_sintomas_consulta' debe ser un entero entre 0 y 30")
    if caso["sexo"] not in SEXOS:
        raise EntradaInvalida("'sexo' debe ser 'F' o 'M'")
    estrato, comuna = caso.get("estrato"), caso.get("comuna")
    if estrato is not None and (isinstance(estrato, bool) or estrato not in range(1, 7)):
        raise EntradaInvalida("'estrato' debe ser un entero entre 1 y 6")
    if comuna is not None and (isinstance(comuna, bool) or comuna not in range(1, 23)):
        raise EntradaInvalida("'comuna' debe ser un entero entre 1 y 22")
    afiliacion = caso.get("afiliacion")
    if afiliacion is not None and afiliacion not in AFILIACIONES:
        raise EntradaInvalida(f"'afiliacion' debe ser una de: {', '.join(AFILIACIONES)}")
    fila = {"edad": float(edad), "dias_sintomas_consulta": dias, "sexo": caso["sexo"],
            "estrato": estrato, "comuna": comuna, "afiliacion": afiliacion}
    fila.update({c: _booleano(caso, c) for c in ml.RIESGO_BINARIAS})
    return ml.preparar_variables_riesgo(pd.DataFrame([fila]))


def _describir(columnas: list[str], fila: pd.Series) -> str:
    """Valor de un grupo de variables en palabras (para explicar el resultado)."""
    columna = columnas[0]
    valor = fila[columna]
    if columna == "edad":
        return f"{valor:.0f} años"
    if columna == "dias_sintomas_consulta":
        return "sin dato" if pd.isna(valor) else f"{valor:.0f} días"
    if columna == "sexo":
        return {"F": "femenino", "M": "masculino"}.get(valor, str(valor).lower())
    if columna in ("estrato", "comuna"):
        return "sin dato" if valor == ml.SIN_DATO else f"{columna} {valor}"
    if columna == "afiliacion":
        return str(valor).lower()
    if len(columnas) > 2:                                   # síntomas generales
        return f"{int(fila[columnas].fillna(0).sum())} de {len(columnas)}"
    marcados = {"migrante": "migrante", "desplazamiento_15d": "viajó"}
    activos = [marcados[c] for c in columnas if fila[c] == 1]
    return " y ".join(activos) if activos else "no"


def explicar(modelo: dict, X: pd.DataFrame) -> list[dict]:
    """Qué características del caso suben o bajan su probabilidad frente a los valores
    habituales de los casos notificados (puntos porcentuales, de mayor a menor peso)."""
    if "fondo" not in modelo:                               # modelos de versiones anteriores
        return []
    variables = modelo.get("variables", ml.RIESGO_VARIABLES)
    tabla = ml.contribuciones(modelo["pipeline"], X, modelo["fondo"], variables)
    caso = X.iloc[0]
    return [{"factor": fila["factor"], "valor": _describir(fila["variables"], caso),
             "puntos": round(fila["puntos"], 1)}
            for _, fila in tabla.iterrows()]


def _clasificar(modelo: dict, X: pd.DataFrame, metadatos: dict) -> dict:
    fila = ml.predecir_riesgo(modelo["pipeline"], X, modelo["umbrales"]).iloc[0]
    por_nivel = metadatos.get("prueba", {}).get("pct_positivos_por_nivel", {})
    probabilidad = float(fila["probabilidad"])
    promedio = modelo.get("prevalencia", metadatos.get("prevalencia"))
    return {"probabilidad": round(probabilidad, 4),
            "nivel_riesgo": fila["nivel_riesgo"],
            "supera_umbral": bool(fila["supera_umbral"]),
            "umbral": round(modelo["umbrales"]["umbral"], 4),
            "pct_positivos_en_nivel_prueba": por_nivel.get(fila["nivel_riesgo"]),
            "promedio": None if promedio is None else round(float(promedio), 4),
            "veces_promedio": None if not promedio else round(probabilidad / promedio, 2),
            "factores": explicar(modelo, X)}


def evaluar_caso(recursos: Recursos, caso: dict) -> dict:
    """Riesgo de signos de alarma o dengue grave de un caso y, si el modelo está
    disponible, su probabilidad de hospitalización."""
    X = validar_caso(caso)
    resultado = _clasificar(recursos.riesgo, X, recursos.metadatos.get("riesgo", {}))
    if recursos.hospitalizacion is not None:
        resultado["hospitalizacion"] = _clasificar(
            recursos.hospitalizacion, X, recursos.metadatos.get("hospitalizacion", {}))
    resultado["advertencia"] = ("Apoyo a la priorización; no reemplaza la valoración "
                                "clínica.")
    return resultado


# ---------------------------------------------------------------------------
# Comunas
# ---------------------------------------------------------------------------

def tabla_comunas(recursos: Recursos) -> pd.DataFrame:
    """Indicadores de las 22 comunas urbanas con su grupo y perfil."""
    perfiles = recursos.perfiles[["comuna", "grupo", "perfil"]]
    urbanas = recursos.indicadores[recursos.indicadores["zona"] == "Urbana"]
    return (urbanas.merge(perfiles, on="comuna", how="left")
            .sort_values("comuna").reset_index(drop=True))


def comuna(recursos: Recursos, numero: int) -> dict:
    tabla = tabla_comunas(recursos)
    fila = tabla[tabla["comuna"] == numero]
    if fila.empty:
        raise KeyError(numero)
    return registros(fila.round(2))[0]


def registros(tabla: pd.DataFrame) -> list[dict]:
    """Filas como diccionarios con tipos que JSON entiende (fechas en texto, NaN a null)."""
    salida = tabla.copy()
    for columna in salida.columns:
        if pd.api.types.is_datetime64_any_dtype(salida[columna]):
            salida[columna] = salida[columna].dt.strftime("%Y-%m-%d")
    salida = salida.astype(object).where(salida.notna(), None)
    return json.loads(json.dumps(salida.to_dict(orient="records"), default=_a_json))


def _a_json(valor):
    if hasattr(valor, "item"):
        return valor.item()
    return str(valor)
