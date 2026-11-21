"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno nuevo.
Silver queda con cinco tablas, cada una con su grano:

- casos_dengue      un registro por caso notificado (2020-2025), con un esquema común
                    para las dos fuentes (SIVIGILA del Distrito y microdatos del INS).
- poblacion         territorio x año, con la proyección 2021-2025 marcada.
- clima_semanal     estación del IDEAM x semana epidemiológica.
- centinela_comunas hospitalizaciones de menores de 5 años por comuna x año.
- serie_semanal     una fila por semana epidemiológica: casos, población, tasa y clima.

El orden de los pasos importa y está fijado en las funciones build_*().
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.config import settings
from flows.common.epidemiologia import (
    calendario_epidemiologico,
    grupo_edad,
    semana_epidemiologica,
    tasa_por_100k,
)
from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_METADATA_NAME,
    SILVER,
    SILVER_CASOS,
    SILVER_CENTINELA,
    SILVER_CLEANING_NAME,
    SILVER_CLIMA,
    SILVER_POBLACION,
    SILVER_SERIE,
)

log = get_logger(__name__)

# --- Fuentes -----------------------------------------------------------------------------
FUENTE_CALI = "SSPD Cali"            # Secretaría de Salud Pública Distrital (2022-2023)
FUENTE_INS = "INS"                   # Instituto Nacional de Salud (2020, 2021, 2024, 2025)
EVENTO_DENGUE = "Dengue"             # evento 210: dengue sin y con signos de alarma
EVENTO_GRAVE = "Dengue grave"        # evento 220
SIVIGILA_KEYS = {                    # archivo de Bronze -> (etiqueta del id, evento)
    "sivigila_2022": ("2022", EVENTO_DENGUE),
    "sivigila_2023": ("2023", EVENTO_DENGUE),
    "grave_2023": ("G2023", EVENTO_GRAVE),
}
MORTALIDAD_KEY = "mortalidad_2023"

# --- Nombres de columnas: las fuentes del Distrito escriben la misma variable distinto ---
RENAME_SIVIGILA = {
    "fecha_nototificacion": "fecha_notificacion",
    "tip_caso": "tipo_caso",
    "Hospitalizado": "hospitalizado",
    "fecha_de_hospitalizacion": "fecha_hospitalizacion",
    "deplazamiento_ultimos_15_dias": "desplazamiento_15d",
    "deplazamiento_ultimos_SI5_dias": "desplazamiento_15d",
    "tipo_seguridad_social": "afiliacion",
    "etnia": "pertenencia_etnica",
    "dolor_retro_ocular": "dolor_retroocular",
    "malgias": "mialgias",
    "hem_mucosa": "hemorragia_mucosas",
    "hemorragia_mucosa": "hemorragia_mucosas",
    "aum_hemato": "aumento_hematocrito",
    "daño_organo": "dano_organo",
    "clasificacion_final": "clasificacion",
}
SINTOMAS_GENERALES = ["fiebre", "cefalea", "dolor_retroocular", "mialgias", "artralgia",
                      "erupcion"]
SIGNOS_ALARMA = ["dolor_abdominal", "vomito", "diarrea", "somnolencia", "hipotension",
                 "hepatomegalia", "hemorragia_mucosas", "hipotermia", "aumento_hematocrito",
                 "caida_plaquetas", "acumulacion_liquidos"]
SIGNOS_GRAVEDAD = ["extravasacion", "hemorragia", "choque", "dano_organo"]
SINTOMAS = SINTOMAS_GENERALES + SIGNOS_ALARMA + SIGNOS_GRAVEDAD

EXPECTED_SIVIGILA = [
    "fecha_notificacion", "edad", "nacionalidad", "sexo", "comuna", "afiliacion",
    "pertenencia_etnica", "estrato", "ciudad_residencia", "fecha_consulta",
    "fecha_inicio_sintomas", "tipo_caso", "hospitalizado", "fecha_hospitalizacion",
    "desplazamiento_15d", "clasificacion", "conducta", *SINTOMAS,
]
EXPECTED_INS = [
    "CONSECUTIVE", "FEC_NOT", "SEMANA", "ANO", "EDAD", "UNI_MED", "SEXO", "TIP_SS", "PER_ETN",
    "estrato", "GP_MIGRANT", "COD_DPTO_R", "COD_MUN_R", "FEC_CON", "INI_SIN", "TIP_CAS",
    "PAC_HOS", "FEC_HOS", "CON_FIN", "AJUSTE",
]
INS_AJUSTES_DESCARTADOS = ["6", "D"]   # 6 = descartado, D = error de digitación

# --- Categorías (orden FIJO: el resultado es reproducible) ------------------------------
MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
         "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}
SI_NO = {"si": True, "1": True, "no": False, "2": False}      # INS: 1 = sí, 2 = no
AFILIACION = {
    "contributivo": "Contributivo", "c": "Contributivo",
    "subsidiado": "Subsidiado", "s": "Subsidiado",
    "excepcion": "Excepción o especial", "p": "Excepción o especial",
    "especial": "Excepción o especial", "e": "Excepción o especial",
    "no asegurado": "No asegurado", "n": "No asegurado",
    "indeterminado": "Indeterminado", "indeterminado/ pendiente": "Indeterminado",
    "i": "Indeterminado",
}
PERTENENCIA_ETNICA = {
    "indigena": "Indígena", "1": "Indígena",
    "rom, gitano": "Rom o gitano", "2": "Rom o gitano",
    "raizal": "Raizal", "3": "Raizal",
    "palenquero": "Palenquero", "4": "Palenquero",
    "negro, mulato, afro colombiano": "Negro, mulato o afrocolombiano",
    "negro, mulato afro colombiano": "Negro, mulato o afrocolombiano",
    "5": "Negro, mulato o afrocolombiano",
    "otro": "Otro", "6": "Otro",
}
TIPO_CASO = {
    "probable": "Probable", "2": "Probable",
    "confirmado por laboratorio": "Confirmado por laboratorio",
    "conf. por laboratorio": "Confirmado por laboratorio",
    "3": "Confirmado por laboratorio",
    "confirmado por nexo epidemiologico": "Confirmado por nexo",
    "conf. nexo epidemiologico": "Confirmado por nexo",
    "5": "Confirmado por nexo",
}
CLASIFICACION = {
    "sin signos de alarma": "Sin signos de alarma",
    "con signos de alarma": "Con signos de alarma",
    "dengue grave": "Dengue grave",
}
CLASES_SEVERAS = ["Con signos de alarma", "Dengue grave"]
CONDUCTA = {
    "ambulatorio": "Ambulatoria", "ambulatoria": "Ambulatoria",
    "observacion": "Observación",
    "hospitalizado piso": "Hospitalización", "hospitalizacion piso": "Hospitalización",
    "remision a hospitalizacion": "Remisión para hospitalización",
    "uci": "UCI", "unidad de cuidados intensivos": "UCI",
}
UNIDAD_EDAD_EN_ANIOS = {"1": 1.0, "2": 1 / 12, "3": 1 / 365, "4": 0.0, "5": 0.0}

COMUNAS_URBANAS = list(range(1, 23))
CORREGIMIENTOS = {                   # código de comuna rural -> corregimiento
    51: "Navarro", 52: "El Hormiguero", 53: "Pance", 54: "La Buitrera", 55: "Villacarmelo",
    56: "Los Andes", 57: "Pichinde", 58: "La Leonera", 59: "Felidia", 60: "El Saladito",
    61: "La Elvira", 62: "La Castilla", 63: "La Paz", 64: "Montebello", 65: "Golondrinas",
}
COMUNAS_VALIDAS = COMUNAS_URBANAS + list(CORREGIMIENTOS)
COMUNA_SIN_DATO = ["sd", "#n/a", "n/a", "sin dato"]   # formas de escribir "no se sabe"
MAX_DIAS_INTERVALO = 30              # más días entre dos fechas del caso se marca atípico
DEDUP_KEY = [
    "evento", "fecha_notificacion", "edad", "sexo", "comuna_original", "afiliacion",
    "estrato", "fecha_consulta", "fecha_inicio_sintomas", "tipo_caso", "hospitalizado",
    "clasificacion",
]
DEATH_KEY = ["edad", "sexo", "comuna", "fecha_inicio_sintomas", "fecha_consulta"]

# --- Población y clima -------------------------------------------------------------------
TERRITORIO_CIUDAD = "TOTAL CALI"
ANIOS_TENDENCIA = 5                  # últimos años observados para proyectar la población
ESTACIONES_URBANAS = {               # estaciones automáticas del IDEAM en la zona urbana
    "0026055120": "Universidad del Valle",
    "0026085170": "Base Aérea Marco Fidel Suárez",
    "0026085160": "Siloé",
}
COBERTURA_MINIMA = 0.5               # fracción de lecturas para que una semana cuente
DANE_AREAS = {"total": "Ciudad", "cabecera municipal": "Cabecera",
              "centros poblados y rural disperso": "Rural"}
AGREGADO_ZONA = {"total cabecera": "Cabecera", "otros": "Cabecera",
                 "total corregimientos": "Rural"}
RANGO_VALIDO = {"lluvia_mm": (0.0, 60.0), "temperatura_c": (5.0, 45.0)}
OBS_ESPERADAS_SEMANA = {"lluvia_mm": 7 * 24 * 6, "temperatura_c": 7 * 24}   # 10 min y 1 h

CASOS_SIN_NULOS = ["id_caso", "fuente", "evento", "fecha_inicio_sintomas", "anio_epi",
                   "semana_epi", "edad", "grupo_edad", "sexo", "hospitalizado", "tipo_caso",
                   "afiliacion", "fallecido"]


# ---------------------------------------------------------------------------
# Utilidades de texto y fechas
# ---------------------------------------------------------------------------

def _norm(text: pd.Series) -> pd.Series:
    """Texto en minúsculas, sin tildes y sin espacios sobrantes (para comparar categorías)."""
    def clean(value):
        if pd.isna(value):
            return None
        plain = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
        return " ".join(plain.lower().split()) or None
    return text.map(clean)


def recode(text: pd.Series, mapping: dict) -> pd.Series:
    """Lleva cada categoría a su nombre canónico; lo que no se reconoce queda nulo."""
    return _norm(text).map(mapping)


def to_bool(text: pd.Series) -> pd.Series:
    """'Si'/'No' (o 1/2 en el INS) -> booleano; cualquier otro valor queda nulo."""
    return _norm(text).map(SI_NO).astype("boolean")


def parse_fecha(text: pd.Series) -> pd.Series:
    """Convierte a fecha los tres formatos de las fuentes.

    '19-abr-2022' (Distrito 2022), '01/01/2023' (Distrito 2023) y
    '10/03/2020 12:00:00 a. m.' (INS). Lo que no es una fecha ('  -   -', vacío) queda NaT.
    """
    partes = text.astype("string").str.strip().str.lower().str.extract(
        r"^(\d{1,2})[-/]([a-z]{3}|\d{1,2})[-/](\d{4})")
    def numero(columna):
        return pd.to_numeric(columna, errors="coerce").astype("float64")

    mes = partes[1].map(MESES).astype("float64").fillna(numero(partes[1]))
    return pd.to_datetime(
        pd.DataFrame({"year": numero(partes[2]), "month": mes, "day": numero(partes[0])}),
        errors="coerce",
    )


def _differs(text: pd.Series, reference: str) -> pd.Series:
    """Verdadero si el texto es distinto de la referencia; nulo si no hay texto."""
    value = _norm(text)
    return (value != reference).where(value.notna()).astype("boolean")


def plain_text(df: pd.DataFrame) -> pd.DataFrame:
    """Deja todas las columnas de texto con el mismo tipo (texto con NaN como faltante).

    Así las tablas se comportan igual al compararlas, agruparlas o pasarlas a scikit-learn,
    sin importar de qué fuente vino cada columna.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col].dtype) or df[col].dtype == object:
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


def to_int_in(text: pd.Series, valid) -> pd.Series:
    """Entero si el valor está en la lista de válidos; si no, nulo."""
    number = pd.to_numeric(text.astype("string").str.strip(), errors="coerce")
    return number.where(number.isin(list(valid))).astype("Int64")


# ---------------------------------------------------------------------------
# Contratos de entrada
# ---------------------------------------------------------------------------

def validate_bronze(df: pd.DataFrame, expected: list[str], nombre: str) -> pd.DataFrame:
    """Contrato de entrada: lanza ValueError si faltan columnas esperadas."""
    faltantes = [c for c in expected if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en Bronze ({nombre}): {faltantes}")
    return df


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Quita las filas sin ningún dato (los archivos del Distrito traen miles)."""
    text = df.astype("string").fillna("")
    con_datos = text.apply(lambda column: column.str.strip() != "").any(axis=1)
    return df[con_datos].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Casos: llevar cada fuente al esquema común
# ---------------------------------------------------------------------------

def harmonize_sivigila(df: pd.DataFrame, etiqueta: str, evento: str = EVENTO_DENGUE,
                       opcionales: tuple[str, ...] = ()) -> pd.DataFrame:
    """Archivo del SIVIGILA del Distrito -> esquema común de casos.

    'etiqueta' identifica el archivo dentro del id del caso (el código original viene
    dañado por notación científica y no sirve como llave). 'opcionales' son columnas que
    ese archivo puede no traer (el de mortalidad no publica clasificación ni conducta).
    """
    df = drop_empty_rows(df).rename(columns=RENAME_SIVIGILA)
    df = df.assign(**{c: pd.NA for c in opcionales if c not in df.columns})
    df = validate_bronze(df, EXPECTED_SIVIGILA, f"SIVIGILA {etiqueta}")
    comuna_texto = df["comuna"].astype("string").str.strip()
    out = pd.DataFrame({
        "id_caso": [f"SSPD-{etiqueta}-{i:05d}" for i in range(1, len(df) + 1)],
        "fuente": FUENTE_CALI,
        "evento": evento,
        "fecha_notificacion": parse_fecha(df["fecha_notificacion"]),
        "fecha_consulta": parse_fecha(df["fecha_consulta"]),
        "fecha_inicio_sintomas": parse_fecha(df["fecha_inicio_sintomas"]),
        "fecha_hospitalizacion": parse_fecha(df["fecha_hospitalizacion"]),
        "anio_reportado": pd.array([pd.NA] * len(df), dtype="Int64"),
        "semana_reportada": pd.array([pd.NA] * len(df), dtype="Int64"),
        "edad": pd.to_numeric(df["edad"], errors="coerce").astype("float64"),
        "sexo": df["sexo"].astype("string").str.strip().str.upper(),
        "estrato": to_int_in(df["estrato"], range(1, 7)),
        "comuna_original": comuna_texto.where(comuna_texto != ""),
        "comuna": to_int_in(df["comuna"], COMUNAS_VALIDAS),
        "afiliacion": recode(df["afiliacion"], AFILIACION),
        "pertenencia_etnica": recode(df["pertenencia_etnica"], PERTENENCIA_ETNICA),
        "tipo_caso": recode(df["tipo_caso"], TIPO_CASO),
        "hospitalizado": to_bool(df["hospitalizado"]),
        "migrante": _differs(df["nacionalidad"], "colombia"),
        "residencia_cali": ~_differs(df["ciudad_residencia"], "cali"),
        "desplazamiento_15d": to_bool(df["desplazamiento_15d"]),
        "clasificacion": recode(df["clasificacion"], CLASIFICACION),
        "conducta": recode(df["conducta"], CONDUCTA),
        "fallecido": False,
    })
    for sintoma in SINTOMAS:
        out[sintoma] = to_bool(df[sintoma])
    return out


def drop_discarded_ins(df: pd.DataFrame) -> pd.DataFrame:
    """Quita los registros del INS que el propio SIVIGILA anuló.

    AJUSTE = 6 (caso descartado) o D (error de digitación) significa que la notificación
    no corresponde a un caso de dengue; las demás categorías (0 sin ajuste, 3 confirmado,
    5 y 7 otras actualizaciones) son casos válidos.
    """
    ajuste = df["AJUSTE"].astype("string").str.strip().str.upper()
    return df[~ajuste.isin(INS_AJUSTES_DESCARTADOS)].reset_index(drop=True)


def harmonize_ins(df: pd.DataFrame) -> pd.DataFrame:
    """Microdatos del INS (evento 210) -> esquema común de casos.

    El INS no publica comuna, clasificación clínica ni síntomas: esas columnas quedan nulas.
    La edad se lleva a años con la unidad de medida (1 = años, 2 = meses, 3 = días).
    """
    df = validate_bronze(df, EXPECTED_INS, "INS")
    text = df.astype("string").apply(lambda column: column.str.strip())
    edad = pd.to_numeric(text["EDAD"], errors="coerce") * text["UNI_MED"].map(
        UNIDAD_EDAD_EN_ANIOS).astype("float64")
    vacio_texto = pd.array([pd.NA] * len(df), dtype="string")
    vacio_bool = pd.array([pd.NA] * len(df), dtype="boolean")
    out = pd.DataFrame({
        "id_caso": "INS-" + text["CONSECUTIVE"],
        "fuente": FUENTE_INS,
        "evento": EVENTO_DENGUE,
        "fecha_notificacion": parse_fecha(text["FEC_NOT"]),
        "fecha_consulta": parse_fecha(text["FEC_CON"]),
        "fecha_inicio_sintomas": parse_fecha(text["INI_SIN"]),
        "fecha_hospitalizacion": parse_fecha(text["FEC_HOS"]),
        "anio_reportado": pd.to_numeric(text["ANO"], errors="coerce").astype("Int64"),
        "semana_reportada": pd.to_numeric(text["SEMANA"], errors="coerce").astype("Int64"),
        "edad": edad.astype("float64"),
        "sexo": text["SEXO"].str.upper(),
        "estrato": to_int_in(text["estrato"], range(1, 7)),
        "comuna_original": vacio_texto,
        "comuna": pd.array([pd.NA] * len(df), dtype="Int64"),
        "afiliacion": recode(text["TIP_SS"], AFILIACION),
        "pertenencia_etnica": recode(text["PER_ETN"], PERTENENCIA_ETNICA),
        "tipo_caso": recode(text["TIP_CAS"], TIPO_CASO),
        "hospitalizado": to_bool(text["PAC_HOS"]),
        "migrante": to_bool(text["GP_MIGRANT"]),
        "residencia_cali": ((text["COD_DPTO_R"] == settings.cod_departamento)
                            & (text["COD_MUN_R"] == settings.cod_municipio)).astype("boolean"),
        "desplazamiento_15d": vacio_bool,
        "clasificacion": vacio_texto,
        "conducta": vacio_texto,
        "fallecido": (text["CON_FIN"] == "2").fillna(False).astype(bool),
    })
    for sintoma in SINTOMAS:
        out[sintoma] = vacio_bool
    return out.reset_index(drop=True)


def mark_deaths(casos: pd.DataFrame, muertes: pd.DataFrame) -> pd.DataFrame:
    """Marca como fallecido el caso que corresponde a cada registro de mortalidad.

    El archivo de mortalidad repite un caso ya notificado como dengue grave; se integra
    por edad, sexo, comuna y fechas en lugar de contarlo dos veces. Si un registro de
    mortalidad no encuentra su caso, se agrega como un caso de dengue grave.
    """
    casos = casos.copy()
    llaves = muertes[DEATH_KEY].drop_duplicates()
    pares = casos.reset_index().merge(llaves, on=DEATH_KEY, how="inner")
    casos.loc[pares["index"], "fallecido"] = True
    sin_caso = muertes.merge(casos[DEATH_KEY].drop_duplicates(), on=DEATH_KEY, how="left",
                             indicator=True)
    sin_caso = sin_caso[sin_caso["_merge"] == "left_only"].drop(columns="_merge")
    if len(sin_caso):
        casos = pd.concat([casos, sin_caso.assign(fallecido=True, evento=EVENTO_GRAVE)],
                          ignore_index=True)
    return casos


def drop_duplicate_cases(casos: pd.DataFrame) -> pd.DataFrame:
    """Elimina notificaciones repetidas del Distrito con una llave compuesta.

    Solo aplica a la fuente del Distrito (su código de caso viene dañado); los
    microdatos del INS ya traen un consecutivo único por caso.
    """
    repetido = (casos["fuente"] == FUENTE_CALI) & casos.duplicated(subset=DEDUP_KEY,
                                                                    keep="first")
    return casos[~repetido].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Casos: variables derivadas
# ---------------------------------------------------------------------------

def add_epi_week(casos: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'anio_epi' y 'semana_epi' según la fecha de inicio de síntomas.

    Es el criterio del SIVIGILA: la semana reportada por el INS coincide con la del
    inicio de síntomas en el 99,9 % de los casos. Cuando la fuente trae la semana
    (INS) se respeta el dato oficial y 'semana_inconsistente' marca las diferencias.
    """
    casos = casos.copy()
    calculada = semana_epidemiologica(casos["fecha_inicio_sintomas"])
    casos["anio_epi"] = casos["anio_reportado"].fillna(calculada["anio_epi"])
    casos["semana_epi"] = casos["semana_reportada"].fillna(calculada["semana_epi"])
    casos["semana_inconsistente"] = (
        casos["semana_reportada"].notna()
        & ((casos["anio_reportado"] != calculada["anio_epi"])
           | (casos["semana_reportada"] != calculada["semana_epi"]))
    ).fillna(False).astype(bool)
    return casos.drop(columns=["anio_reportado", "semana_reportada"])


def filter_period(casos: pd.DataFrame, anio_inicio: int, anio_fin: int) -> pd.DataFrame:
    """Conserva los casos cuyo año epidemiológico está en el periodo de análisis."""
    en_periodo = casos["anio_epi"].between(anio_inicio, anio_fin).fillna(False)
    return casos[en_periodo.astype(bool)].reset_index(drop=True)


def add_age_group(casos: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'grupo_edad' (0-4, 5-14, 15-29, 30-44, 45-64, 65+)."""
    casos = casos.copy()
    casos["grupo_edad"] = grupo_edad(casos["edad"])
    return casos


def add_delays(casos: pd.DataFrame) -> pd.DataFrame:
    """Agrega los días entre inicio de síntomas, consulta y notificación (oportunidad)."""
    casos = casos.copy()
    casos["dias_sintomas_consulta"] = (
        casos["fecha_consulta"] - casos["fecha_inicio_sintomas"]).dt.days.astype("Int64")
    casos["dias_consulta_notificacion"] = (
        casos["fecha_notificacion"] - casos["fecha_consulta"]).dt.days.astype("Int64")
    return casos


def add_quality_flags(casos: pd.DataFrame) -> pd.DataFrame:
    """Marca los registros anómalos en lugar de borrarlos.

    - fecha_incoherente: la consulta es anterior a los síntomas o la notificación anterior
      a la consulta.
    - intervalo_atipico: más de 30 días entre dos fechas del mismo caso.
    - estado_comuna: 'Válida', 'Sin dato', 'Código no válido' o 'No disponible' (INS).
    """
    casos = casos.copy()
    dias = casos[["dias_sintomas_consulta", "dias_consulta_notificacion"]]
    casos["fecha_incoherente"] = (dias < 0).any(axis=1).fillna(False).astype(bool)
    casos["intervalo_atipico"] = (dias > MAX_DIAS_INTERVALO).any(axis=1).fillna(False).astype(
        bool)
    original = _norm(casos["comuna_original"])
    casos["estado_comuna"] = np.select(
        [casos["comuna"].notna().to_numpy(), (casos["fuente"] == FUENTE_INS).to_numpy(),
         (original.isna() | original.isin(COMUNA_SIN_DATO)).to_numpy()],
        ["Válida", "No disponible", "Sin dato"],
        default="Código no válido",
    )
    return casos


def add_zone(casos: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'zona': Urbana (comunas 1 a 22), Rural (corregimientos 51 a 65) o Sin dato."""
    casos = casos.copy()
    comuna = casos["comuna"]
    casos["zona"] = np.select(
        [comuna.isin(COMUNAS_URBANAS).fillna(False).to_numpy(dtype=bool),
         comuna.isin(list(CORREGIMIENTOS)).fillna(False).to_numpy(dtype=bool)],
        ["Urbana", "Rural"], default="Sin dato",
    )
    return casos


def add_severity(casos: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'severo': con signos de alarma o dengue grave (nulo si no hay clasificación)."""
    casos = casos.copy()
    severo = casos["clasificacion"].isin(CLASES_SEVERAS).astype("boolean")
    casos["severo"] = severo.where(casos["clasificacion"].notna())
    return casos


def add_symptom_count(casos: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'n_sintomas_generales': cuántos de los seis síntomas generales presenta."""
    casos = casos.copy()
    generales = casos[SINTOMAS_GENERALES]
    total = generales.astype("float64").sum(axis=1, min_count=1)
    casos["n_sintomas_generales"] = total.astype("Int64")
    return casos


def validate_casos(casos: pd.DataFrame) -> pd.DataFrame:
    """Contrato de salida de casos: sin nulos en columnas clave, id único y semanas válidas."""
    nulos = casos[CASOS_SIN_NULOS].isnull().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        raise ValueError(f"Silver (casos) tiene nulos: {nulos.to_dict()}")
    if casos["id_caso"].duplicated().any():
        raise ValueError("Silver (casos) tiene id_caso duplicados")
    if not casos["semana_epi"].between(1, 53).all():
        raise ValueError("Silver (casos) tiene semanas epidemiológicas fuera de 1-53")
    return casos


def union_sources(bronze: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une todas las fuentes de casos ya llevadas al esquema común (con los fallecidos)."""
    partes = [harmonize_sivigila(bronze[key], etiqueta, evento)
              for key, (etiqueta, evento) in SIVIGILA_KEYS.items() if key in bronze]
    partes += [harmonize_ins(drop_discarded_ins(validate_bronze(df, EXPECTED_INS, key)))
               for key, df in sorted(bronze.items()) if key.startswith("ins_")]
    casos = pd.concat(partes, ignore_index=True)
    if MORTALIDAD_KEY in bronze:
        muertes = harmonize_sivigila(bronze[MORTALIDAD_KEY], "M2023", EVENTO_GRAVE,
                                     opcionales=("clasificacion", "conducta"))
        casos = mark_deaths(casos, muertes)
    return casos


def build_casos(bronze: dict[str, pd.DataFrame], anio_inicio: int | None = None,
                anio_fin: int | None = None) -> pd.DataFrame:
    """Orden deliberado: los duplicados se quitan antes de calcular nada, y la semana
    epidemiológica va antes de filtrar el periodo (el filtro usa el año epidemiológico)."""
    anio_inicio = settings.anio_inicio if anio_inicio is None else anio_inicio
    anio_fin = settings.anio_fin if anio_fin is None else anio_fin
    return (
        union_sources(bronze)
        .pipe(drop_duplicate_cases)
        .pipe(add_epi_week)
        .pipe(filter_period, anio_inicio, anio_fin)
        .pipe(add_age_group)
        .pipe(add_delays)
        .pipe(add_quality_flags)
        .pipe(add_zone)
        .pipe(add_severity)
        .pipe(add_symptom_count)
        .pipe(plain_text)
        .astype({"hospitalizado": bool})
        .pipe(validate_casos)
    )


# ---------------------------------------------------------------------------
# Población
# ---------------------------------------------------------------------------

def tidy_poblacion(df: pd.DataFrame) -> pd.DataFrame:
    """Formato ancho (un año por columna) -> largo: territorio, anio, poblacion."""
    df = df.rename(columns={df.columns[0]: "territorio"})
    largo = df.melt(id_vars="territorio", var_name="anio", value_name="poblacion")
    largo["territorio"] = largo["territorio"].astype("string").str.strip()
    largo["anio"] = pd.to_numeric(largo["anio"], errors="coerce")
    largo["poblacion"] = pd.to_numeric(largo["poblacion"], errors="coerce")
    largo = largo.dropna(subset=["anio", "poblacion"])
    return largo.astype({"anio": int, "poblacion": int}).reset_index(drop=True)


def add_territory_type(poblacion: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'tipo' (Ciudad, Comuna, Corregimiento o Agregado) y el código de 'comuna'."""
    poblacion = poblacion.copy()
    nombre = _norm(poblacion["territorio"])
    por_corregimiento = {_norm(pd.Series([n]))[0]: c for c, n in CORREGIMIENTOS.items()}
    numero = pd.to_numeric(nombre.str.extract(r"^comuna (\d+)$")[0], errors="coerce")
    poblacion["comuna"] = numero.fillna(nombre.map(por_corregimiento)).astype("Int64")
    poblacion["tipo"] = np.select(
        [(nombre == _norm(pd.Series([TERRITORIO_CIUDAD]))[0]).to_numpy(dtype=bool),
         numero.notna().to_numpy(),
         nombre.isin(list(por_corregimiento)).to_numpy(dtype=bool)],
        ["Ciudad", "Comuna", "Corregimiento"], default="Agregado",
    )
    return poblacion


def project_poblacion(poblacion: pd.DataFrame, anio_fin: int,
                      anios_tendencia: int = ANIOS_TENDENCIA) -> pd.DataFrame:
    """Proyecta la población hasta 'anio_fin' con la tendencia lineal de los últimos años.

    La fuente llega a 2020. Las filas proyectadas quedan marcadas en 'proyectada' para
    que nunca se confundan con el dato oficial.
    """
    poblacion = poblacion.assign(proyectada=False)
    nuevas = []
    for territorio, grupo in poblacion.groupby("territorio", sort=False):
        base = grupo.sort_values("anio").tail(anios_tendencia)
        ultimo = int(base["anio"].max())
        if ultimo >= anio_fin or len(base) < 2:
            continue
        pendiente, intercepto = np.polyfit(base["anio"], base["poblacion"], 1)
        for anio in range(ultimo + 1, anio_fin + 1):
            nuevas.append({
                **grupo.iloc[0].to_dict(), "territorio": territorio, "anio": anio,
                "poblacion": max(int(round(pendiente * anio + intercepto)), 0),
                "proyectada": True,
            })
    return pd.concat([poblacion, pd.DataFrame(nuevas)], ignore_index=True).astype(
        {"anio": int, "poblacion": int, "proyectada": bool, "comuna": "Int64"})


def validate_poblacion(poblacion: pd.DataFrame) -> pd.DataFrame:
    """Contrato de salida: una fila por territorio y año, población positiva y la ciudad."""
    if poblacion.duplicated(subset=["territorio", "anio"]).any():
        raise ValueError("Silver (población) repite territorio y año")
    if (poblacion["poblacion"] <= 0).any():
        raise ValueError("Silver (población) tiene valores no positivos")
    if not (poblacion["tipo"] == "Ciudad").any():
        raise ValueError(f"Silver (población) no trae el total '{TERRITORIO_CIUDAD}'")
    return poblacion


def tidy_poblacion_dane(df: pd.DataFrame) -> pd.DataFrame:
    """Proyecciones del DANE del municipio: anio, zona (Ciudad, Cabecera o Rural), poblacion."""
    area = _norm(df["ÁREA GEOGRÁFICA"]).map(DANE_AREAS)
    dane = pd.DataFrame({
        "anio": pd.to_numeric(df["AÑO"], errors="coerce"),
        "zona": area,
        "poblacion": pd.to_numeric(df["TOTAL"], errors="coerce"),
    }).dropna()
    if dane.duplicated(subset=["anio", "zona"]).any():
        raise ValueError("Bronze (DANE) repite año y área: revisar el filtro del municipio")
    return dane.astype({"anio": int, "poblacion": int}).reset_index(drop=True)


def zona_territorio(poblacion: pd.DataFrame) -> pd.Series:
    """A qué total del DANE pertenece cada territorio: Ciudad, Cabecera o Rural."""
    nombre = _norm(poblacion["territorio"])
    return pd.Series(np.select(
        [(poblacion["tipo"] == "Ciudad").to_numpy(), (poblacion["tipo"] == "Comuna").to_numpy(),
         (poblacion["tipo"] == "Corregimiento").to_numpy()],
        ["Ciudad", "Cabecera", "Rural"], default=None), index=poblacion.index).fillna(
        nombre.map(AGREGADO_ZONA))


def _totales_por_zona(poblacion: pd.DataFrame) -> pd.DataFrame:
    """Población de Planeación por año en la ciudad, la cabecera y la zona rural.

    Usa las filas de total del archivo; si no están, suma comunas o corregimientos.
    """
    nombre = _norm(poblacion["territorio"])
    totales = {}
    for zona, total, tipo in (("Ciudad", None, "Ciudad"), ("Cabecera", "total cabecera", "Comuna"),
                              ("Rural", "total corregimientos", "Corregimiento")):
        filas = poblacion[nombre == total] if total else pd.DataFrame()
        if filas.empty:
            filas = poblacion[poblacion["tipo"] == tipo]
        totales[zona] = filas.groupby("anio")["poblacion"].sum()
    return pd.DataFrame(totales)


def adjust_to_dane(poblacion: pd.DataFrame, dane: pd.DataFrame, anio_fin: int) -> pd.DataFrame:
    """Población oficial del DANE desde su primer año, repartida entre los territorios.

    El total de la ciudad, de la cabecera y de la zona rural son las proyecciones del DANE
    (censo 2018, actualización posterior a la pandemia). Cada comuna recibe el total de la
    cabecera multiplicado por su participación en la proyección de Planeación Distrital
    del mismo año, y cada corregimiento, el total rural por su participación entre los
    corregimientos. Después de 2020 (último año de Planeación) se conserva la participación
    de 2020 y la fila queda marcada en 'proyectada' (la participación es supuesta, el
    total es oficial). Antes del primer año del DANE se
    usa la proyección de Planeación sin cambios.
    """
    poblacion = poblacion.assign(zona=zona_territorio(poblacion), proyectada=False,
                                 fuente_poblacion="Planeación Distrital")
    ultimo_dap = int(poblacion["anio"].max())
    totales_dap = _totales_por_zona(poblacion)
    oficial = dane.pivot(index="anio", columns="zona", values="poblacion")
    nuevas = []
    for anio in range(int(oficial.index.min()), anio_fin + 1):
        if anio not in oficial.index:
            raise ValueError(f"Las proyecciones del DANE no incluyen el año {anio}")
        base = min(anio, ultimo_dap)
        filas = poblacion[(poblacion["anio"] == base) & poblacion["zona"].notna()].copy()
        factor = filas["zona"].map(oficial.loc[anio] / totales_dap.loc[base])
        filas["poblacion"] = (filas["poblacion"] * factor).round().astype(int)
        filas["anio"] = anio
        oficial_dane = ((filas["tipo"] == "Ciudad")
                        | _norm(filas["territorio"]).isin(["total cabecera",
                                                           "total corregimientos"]))
        filas["proyectada"] = (anio > ultimo_dap) & ~oficial_dane
        filas["fuente_poblacion"] = np.where(oficial_dane, "DANE",
                                             "DANE con participación de Planeación")
        nuevas.append(filas)
    anteriores = poblacion[poblacion["anio"] < oficial.index.min()]
    return (pd.concat([anteriores, *nuevas], ignore_index=True)
            .drop(columns="zona").astype({"anio": int, "poblacion": int, "comuna": "Int64"})
            .sort_values(["anio", "territorio"]).reset_index(drop=True))


def build_poblacion(bronze: pd.DataFrame, anio_fin: int | None = None,
                    dane: pd.DataFrame | None = None) -> pd.DataFrame:
    """Población por territorio y año.

    Con las proyecciones del DANE (fuente complementaria) los totales son los oficiales;
    sin ellas, se extiende la tendencia lineal de Planeación Distrital.
    """
    anio_fin = settings.anio_fin if anio_fin is None else anio_fin
    base = bronze.pipe(tidy_poblacion).pipe(add_territory_type)
    if dane is not None and len(dane):
        poblacion = adjust_to_dane(base, tidy_poblacion_dane(dane), anio_fin)
    else:
        poblacion = project_poblacion(base, anio_fin).assign(fuente_poblacion=lambda d: np.where(
            d["proyectada"], "Tendencia lineal de Planeación", "Planeación Distrital"))
    return poblacion.pipe(plain_text).pipe(validate_poblacion)


# ---------------------------------------------------------------------------
# Clima (IDEAM)
# ---------------------------------------------------------------------------

def tidy_clima(precipitacion: pd.DataFrame, temperatura: pd.DataFrame) -> pd.DataFrame:
    """Une lluvia y temperatura en formato largo: estación, fecha, variable y valor."""
    partes = []
    for variable, df in (("lluvia_mm", precipitacion), ("temperatura_c", temperatura)):
        sensor = (df["descripcionsensor"] if "descripcionsensor" in df
                  else pd.Series("", index=df.index))
        partes.append(pd.DataFrame({
            "codigo_estacion": df["codigoestacion"].astype("string").str.strip(),
            "estacion": df["nombreestacion"].astype("string").str.split().str.join(" "),
            "fecha_hora": pd.to_datetime(df["fechaobservacion"], errors="coerce"),
            "variable": variable,
            "valor": pd.to_numeric(df["valorobservado"], errors="coerce"),
            "sistema": sistema_medicion(sensor),
        }))
    return pd.concat(partes, ignore_index=True)


def sistema_medicion(descripcion: pd.Series) -> pd.Series:
    """Sistema con que el IDEAM publicó la lectura, según cómo describe el sensor.

    Hasta 2024 el sensor aparece como 'Precipitacion' o 'Temp Aire 2 m'; desde la segunda
    mitad de 2024, como 'PRECIPITACIÓN' o 'TEMPERATURA DEL AIRE A 2 m'. El cambio coincide
    con un salto en el nivel de la temperatura de algunas estaciones (hasta 3 °C), así que
    cada sistema se trata como una serie distinta al calcular anomalías.
    """
    texto = descripcion.astype("string").fillna("")
    nuevo = texto.str.isupper() | texto.str.upper().str.startswith("TEMPERATURA DEL AIRE")
    return pd.Series(np.where(nuevo, "actual", "anterior"), index=descripcion.index)


def flag_climate_anomalies(clima: pd.DataFrame) -> pd.DataFrame:
    """Marca las lecturas sin valor, sin fecha, repetidas o fuera del rango físico válido."""
    clima = clima.copy()
    minimo = clima["variable"].map({k: v[0] for k, v in RANGO_VALIDO.items()})
    maximo = clima["variable"].map({k: v[1] for k, v in RANGO_VALIDO.items()})
    clima["fuera_de_rango"] = ((clima["valor"] < minimo) | (clima["valor"] > maximo)).fillna(
        False).astype(bool)
    clima["repetida"] = clima.duplicated(subset=["codigo_estacion", "fecha_hora", "variable"])
    clima["valida"] = (clima["valor"].notna() & clima["fecha_hora"].notna()
                       & ~clima["fuera_de_rango"] & ~clima["repetida"])
    return clima


def tidy_estaciones(df: pd.DataFrame) -> pd.DataFrame:
    """Catálogo del IDEAM: código, categoría, tecnología, estado, altitud y coordenadas."""
    return pd.DataFrame({
        "codigo_estacion": df["Codigo"].astype("string").str.strip(),
        "categoria": df["Categoria"].astype("string").str.strip(),
        "tecnologia": df["Tecnologia"].astype("string").str.strip(),
        "estado": df["Estado"].astype("string").str.strip(),
        "altitud_m": pd.to_numeric(df["Altitud"], errors="coerce"),
        "latitud": pd.to_numeric(df["LATITUD"], errors="coerce"),
        "longitud": pd.to_numeric(df["LONGITUD"], errors="coerce"),
    }).drop_duplicates(subset="codigo_estacion").reset_index(drop=True)


def weekly_climate(clima: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Resume el clima por estación y semana epidemiológica.

    Lluvia acumulada y temperatura media, mínima y máxima, con el número de lecturas
    válidas y la cobertura (lecturas recibidas / esperadas en una semana completa). Una
    cobertura baja significa que la semana no es comparable con las demás.
    """
    clima = clima[clima["valida"]].copy()
    clima[["anio_epi", "semana_epi"]] = semana_epidemiologica(clima["fecha_hora"])
    llaves = ["codigo_estacion", "estacion", "sistema", "anio_epi", "semana_epi"]
    lluvia = (clima[clima["variable"] == "lluvia_mm"].groupby(llaves)["valor"]
              .agg(lluvia_mm="sum", n_obs_lluvia="size").reset_index())
    temperatura = (clima[clima["variable"] == "temperatura_c"].groupby(llaves)["valor"]
                   .agg(temp_media="mean", temp_min="min", temp_max="max", n_obs_temp="size")
                   .reset_index())
    semanal = lluvia.merge(temperatura, on=llaves, how="outer")
    semanal[["n_obs_lluvia", "n_obs_temp"]] = semanal[["n_obs_lluvia", "n_obs_temp"]].fillna(
        0).astype(int)
    semanal["cobertura_lluvia"] = semanal["n_obs_lluvia"] / OBS_ESPERADAS_SEMANA["lluvia_mm"]
    semanal["cobertura_temp"] = semanal["n_obs_temp"] / OBS_ESPERADAS_SEMANA["temperatura_c"]
    semanal = semanal.merge(estaciones, on="codigo_estacion", how="left")
    return semanal.sort_values(llaves).reset_index(drop=True).astype(
        {"anio_epi": int, "semana_epi": int})


def union_clima(bronze: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lluvia y temperatura de todas las descargas del IDEAM (manual y por la API).

    Las lecturas que aparecen en ambas descargas se marcan luego como repetidas.
    """
    lluvia = [bronze[k] for k in ("precipitacion", "precipitacion_api") if k in bronze]
    temperatura = [bronze[k] for k in ("temperatura", "temperatura_api") if k in bronze]
    return (pd.concat(lluvia, ignore_index=True), pd.concat(temperatura, ignore_index=True))


def build_clima(precipitacion: pd.DataFrame, temperatura: pd.DataFrame,
                estaciones: pd.DataFrame) -> pd.DataFrame:
    return (
        tidy_clima(precipitacion, temperatura)
        .pipe(flag_climate_anomalies)
        .pipe(weekly_climate, tidy_estaciones(estaciones))
        .pipe(plain_text)
    )


# ---------------------------------------------------------------------------
# Centinela de hospitalizaciones en menores de 5 años
# ---------------------------------------------------------------------------

def build_centinela(df: pd.DataFrame) -> pd.DataFrame:
    """Hospitalizaciones por comuna y año (suma de los barrios con comuna válida)."""
    limpio = pd.DataFrame({
        "anio": pd.to_numeric(df["ano"], errors="coerce"),
        "comuna": to_int_in(df["comuna"], COMUNAS_VALIDAS),
        "hospitalizaciones": pd.to_numeric(df["tot_hosp"], errors="coerce"),
        "hosp_respiratoria": pd.to_numeric(df["num_hosp_ERA"], errors="coerce"),
        "hosp_asma": pd.to_numeric(df["num_hosp_asma"], errors="coerce"),
        "hosp_diarreica": pd.to_numeric(df["num_hosp_EDA"], errors="coerce"),
        "hosp_arbovirosis": pd.to_numeric(df["num_hosp_deng_chik_zika"], errors="coerce"),
    }).dropna(subset=["anio", "comuna"])
    resumen = limpio.groupby(["anio", "comuna"], as_index=False).sum(numeric_only=True)
    resumen["n_barrios"] = limpio.groupby(["anio", "comuna"]).size().to_numpy()
    return resumen.astype({c: int for c in resumen.columns})


# ---------------------------------------------------------------------------
# Serie semanal de la ciudad
# ---------------------------------------------------------------------------

def weekly_counts(casos: pd.DataFrame) -> pd.DataFrame:
    """Casos de dengue (evento 210) por semana epidemiológica, con sus desgloses.

    El dengue grave (evento 220) se cuenta aparte en 'casos_graves' porque solo está
    disponible en 2023: sumarlo haría que los años no fueran comparables.
    """
    dengue = casos[casos["evento"] == EVENTO_DENGUE]
    llaves = ["anio_epi", "semana_epi"]
    conteo = dengue.groupby(llaves).agg(
        casos=("id_caso", "size"),
        hospitalizados=("hospitalizado", "sum"),
        confirmados=("tipo_caso", lambda s: int((s != "Probable").sum())),
        menores_15=("edad", lambda s: int((s < 15).sum())),
    ).reset_index()
    graves = (casos[casos["evento"] == EVENTO_GRAVE].groupby(llaves).size()
              .rename("casos_graves").reset_index())
    return conteo.merge(graves, on=llaves, how="outer").astype(
        {"anio_epi": int, "semana_epi": int})


def add_coverage(serie: pd.DataFrame, casos: pd.DataFrame,
                 semanas_rezago: int) -> pd.DataFrame:
    """Agrega 'cobertura' de cada semana: completa, parcial o sin_datos.

    Cada año tiene una fecha de corte: la última notificación de su archivo. Una semana
    que no había terminado en esa fecha no tiene datos; las 'semanas_rezago' anteriores
    están incompletas porque los casos tardan en notificarse. En las semanas sin datos
    los conteos quedan nulos (no son cero casos).
    """
    serie = serie.copy()
    corte = casos.groupby("anio_epi")["fecha_notificacion"].max()
    corte = serie["anio_epi"].map(corte)
    sin_datos = corte.isna() | (serie["fecha_fin"] > corte)
    parcial = ~sin_datos & (serie["fecha_fin"] > corte - pd.Timedelta(weeks=semanas_rezago))
    serie["cobertura"] = np.select([sin_datos, parcial], ["sin_datos", "parcial"],
                                   default="completa")
    conteos = ["casos", "hospitalizados", "confirmados", "menores_15"]
    serie[conteos] = serie[conteos].fillna(0)
    serie.loc[sin_datos, conteos] = np.nan
    # El dengue grave solo se conoce en los años cuyo archivo lo publica
    con_graves = serie.groupby("anio_epi")["casos_graves"].transform("count") > 0
    serie["casos_graves"] = serie["casos_graves"].fillna(0).where(con_graves & ~sin_datos)
    return serie.astype({c: "Int64" for c in [*conteos, "casos_graves"]})


def impute_gaps(serie: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'casos_modelo': la serie sin huecos que usan los modelos.

    Las semanas parciales o sin datos se rellenan con una interpolación lineal en escala
    logarítmica entre las semanas completas vecinas, y quedan marcadas en 'imputada'.
    'casos' conserva siempre el dato observado.
    """
    serie = serie.copy()
    serie["imputada"] = serie["cobertura"] != "completa"
    observado = np.log1p(serie["casos"].astype("float64").where(~serie["imputada"]))
    relleno = observado.interpolate(method="linear", limit_direction="both")
    serie["casos_modelo"] = np.expm1(relleno).round(1)
    return serie


def add_population(serie: pd.DataFrame, poblacion: pd.DataFrame) -> pd.DataFrame:
    """Agrega la población de la ciudad del año y la tasa semanal por 100.000 habitantes."""
    ciudad = poblacion[poblacion["tipo"] == "Ciudad"].set_index("anio")["poblacion"]
    serie = serie.copy()
    serie["poblacion"] = serie["anio_epi"].map(ciudad)
    serie["tasa_100k"] = tasa_por_100k(serie["casos"].astype("float64"), serie["poblacion"])
    return serie


def clima_ciudad(clima: pd.DataFrame, estaciones=tuple(ESTACIONES_URBANAS),
                 cobertura_minima: float = COBERTURA_MINIMA) -> pd.DataFrame:
    """Clima semanal de la ciudad: promedio de las estaciones urbanas con datos suficientes.

    Una estación cuenta en una semana si recibió al menos la mitad de las lecturas
    esperadas. Como faltan lecturas, la lluvia de cada estación se lleva a la semana
    completa (lluvia / cobertura) antes de promediar.

    'temp_anomalia' es la diferencia con la temperatura media de la misma estación y el
    mismo sistema de medición, promediada entre estaciones: así el cambio de sistema del
    IDEAM no se confunde con un cambio del clima.
    """
    llaves = ["anio_epi", "semana_epi"]
    urbanas = clima[clima["codigo_estacion"].isin(list(estaciones))].copy()
    cob_lluvia = urbanas["cobertura_lluvia"].clip(upper=1)
    cob_temp = urbanas["cobertura_temp"].clip(upper=1)
    urbanas["lluvia_semana"] = (urbanas["lluvia_mm"] / cob_lluvia).where(
        cob_lluvia >= cobertura_minima)
    urbanas["temp_semana"] = urbanas["temp_media"].where(cob_temp >= cobertura_minima)
    urbanas["cob_lluvia"] = cob_lluvia.where(cob_lluvia >= cobertura_minima)
    urbanas["cob_temp"] = cob_temp.where(cob_temp >= cobertura_minima)
    urbanas["temp_anomalia"] = urbanas["temp_semana"] - urbanas.groupby(
        ["codigo_estacion", "sistema"])["temp_semana"].transform("mean")
    ciudad = urbanas.groupby(llaves).agg(
        lluvia_mm=("lluvia_semana", "mean"), temp_media=("temp_semana", "mean"),
        temp_anomalia=("temp_anomalia", "mean"),
        cobertura_lluvia=("cob_lluvia", "mean"), cobertura_temp=("cob_temp", "mean"),
        estaciones_lluvia=("lluvia_semana", "count"), estaciones_temp=("temp_semana", "count"),
    ).reset_index()
    ciudad["lluvia_mm"] = ciudad["lluvia_mm"].round(1)
    ciudad["temp_media"] = ciudad["temp_media"].round(2)
    ciudad["temp_anomalia"] = ciudad["temp_anomalia"].round(2)
    return ciudad


def add_climate(serie: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    """Agrega el clima semanal de la ciudad (nulo en las semanas sin lecturas suficientes)."""
    serie = serie.merge(clima_ciudad(clima), on=["anio_epi", "semana_epi"], how="left")
    conteos = ["estaciones_lluvia", "estaciones_temp"]
    serie[conteos] = serie[conteos].fillna(0).astype(int)
    return serie


def validate_serie(serie: pd.DataFrame) -> pd.DataFrame:
    """Contrato de salida: semanas consecutivas, sin repetir, y serie de modelado completa."""
    if serie.duplicated(subset=["anio_epi", "semana_epi"]).any():
        raise ValueError("Silver (serie) repite semanas")
    if not (serie["fecha_inicio"].diff().dropna() == pd.Timedelta(weeks=1)).all():
        raise ValueError("Silver (serie) no tiene semanas consecutivas")
    if serie[["casos_modelo", "poblacion"]].isnull().any().any():
        raise ValueError("Silver (serie) tiene nulos en casos_modelo o poblacion")
    if serie.loc[serie["cobertura"] != "sin_datos", "casos"].isnull().any():
        raise ValueError("Silver (serie) tiene semanas con datos y conteo nulo")
    return serie


def build_serie(casos: pd.DataFrame, poblacion: pd.DataFrame, clima: pd.DataFrame,
                anio_inicio: int | None = None, anio_fin: int | None = None,
                semanas_rezago: int | None = None) -> pd.DataFrame:
    """Orden deliberado: la cobertura va antes de imputar, y la tasa usa el dato observado."""
    anio_inicio = settings.anio_inicio if anio_inicio is None else anio_inicio
    anio_fin = settings.anio_fin if anio_fin is None else anio_fin
    semanas_rezago = settings.semanas_rezago if semanas_rezago is None else semanas_rezago
    return (
        calendario_epidemiologico(anio_inicio, anio_fin)
        .merge(weekly_counts(casos), on=["anio_epi", "semana_epi"], how="left")
        .pipe(add_coverage, casos, semanas_rezago)
        .pipe(impute_gaps)
        .pipe(add_population, poblacion)
        .pipe(add_climate, clima)
        .pipe(plain_text)
        .pipe(validate_serie)
    )


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def build_silver(bronze: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Construye las cinco tablas de Silver a partir de los archivos de Bronze."""
    casos = build_casos(bronze)
    poblacion = build_poblacion(bronze["poblacion"], dane=bronze.get("poblacion_dane"))
    clima = build_clima(*union_clima(bronze), bronze["estaciones"])
    return {
        SILVER_CASOS: casos,
        SILVER_POBLACION: poblacion,
        SILVER_CLIMA: clima,
        SILVER_CENTINELA: build_centinela(bronze["centinela"]),
        SILVER_SERIE: build_serie(casos, poblacion, clima),
    }


def cleaning_summary(bronze: dict[str, pd.DataFrame], silver: dict[str, pd.DataFrame]) -> dict:
    """Deja constancia de lo que se descartó o marcó: nada se pierde en silencio."""
    casos, serie = silver[SILVER_CASOS], silver[SILVER_SERIE]
    con_fuente = [k for k in (*SIVIGILA_KEYS, MORTALIDAD_KEY) if k in bronze]
    vacias = {k: int(len(bronze[k]) - len(drop_empty_rows(bronze[k]))) for k in con_fuente}
    descartados_ins = sum(len(df) - len(drop_discarded_ins(df))
                          for key, df in bronze.items() if key.startswith("ins_"))
    unidos = union_sources(bronze)
    sin_duplicados = drop_duplicate_cases(unidos)
    clima = flag_climate_anomalies(tidy_clima(*union_clima(bronze)))
    centinela = bronze["centinela"]
    return {
        "filas_vacias_descartadas": {**vacias, "total": sum(vacias.values())},
        "registros_ins_anulados": int(descartados_ins),
        "casos_con_datos": int(len(unidos)),
        "duplicados_eliminados": int(len(unidos) - len(sin_duplicados)),
        "casos_fuera_del_periodo": int(len(sin_duplicados) - len(casos)),
        "casos_en_silver": int(len(casos)),
        "fallecidos_integrados": int(casos["fallecido"].sum()),
        "registros_marcados": {
            "fecha_incoherente": int(casos["fecha_incoherente"].sum()),
            "intervalo_atipico": int(casos["intervalo_atipico"].sum()),
            "semana_inconsistente": int(casos["semana_inconsistente"].sum()),
            "comuna_sin_dato": int((casos["estado_comuna"] == "Sin dato").sum()),
            "comuna_codigo_no_valido": int((casos["estado_comuna"] == "Código no válido").sum()),
            "estrato_sin_dato": int(casos["estrato"].isna().sum()),
            "sin_clasificacion_clinica_distrito": int(
                ((casos["fuente"] == FUENTE_CALI) & casos["clasificacion"].isna()).sum()),
        },
        "semanas": {
            "total": int(len(serie)),
            **{k: int(v) for k, v in serie["cobertura"].value_counts().items()},
        },
        "clima": {
            "lecturas": int(len(clima)),
            "fuera_de_rango": int(clima["fuera_de_rango"].sum()),
            "repetidas": int(clima["repetida"].sum()),
            "semanas_con_lluvia": int(serie["lluvia_mm"].notna().sum()),
            "semanas_con_temperatura": int(serie["temp_media"].notna().sum()),
            "semanas_con_lluvia_y_temperatura": int(
                serie[["lluvia_mm", "temp_media"]].notna().all(axis=1).sum()),
        },
        "poblacion": {
            fuente: sorted(int(a) for a in grupo["anio"].unique())
            for fuente, grupo in silver[SILVER_POBLACION].groupby("fuente_poblacion")
        },
        "centinela_filas_sin_comuna_valida": int(
            to_int_in(centinela["comuna"], COMUNAS_VALIDAS).isna().sum()),
    }


def read_bronze(in_dir: Path = BRONZE) -> dict[str, pd.DataFrame]:
    """Lee todos los archivos de Bronze como texto, guiado por los metadatos de la ingesta."""
    in_dir = Path(in_dir)
    meta_path = in_dir / BRONZE_METADATA_NAME
    if not meta_path.exists():
        raise FileNotFoundError(f"No existe {meta_path}: ejecuta primero la ingesta (make ingest)")
    files = json.loads(meta_path.read_text(encoding="utf-8"))["files"]
    return {
        key: pd.read_csv(in_dir / f["bronze_file"], sep=f["sep"], encoding=f["encoding"],
                         dtype=str, keep_default_na=False)
        for key, f in files.items()
    }


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = read_bronze(in_dir)
    silver = build_silver(bronze)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in silver.items():
        table.to_parquet(out_dir / name, index=False)
        log.info("Silver: %s con %d filas y %d columnas", name, *table.shape)
    resumen = cleaning_summary(bronze, silver)
    (out_dir / SILVER_CLEANING_NAME).write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Silver listo: %d casos, %d duplicados eliminados, %d filas vacías descartadas",
             resumen["casos_en_silver"], resumen["duplicados_eliminados"],
             resumen["filas_vacias_descartadas"]["total"])
    return out_dir / SILVER_CASOS
