"""Modelado: Silver y Gold -> modelos entrenados y guardados con sus metadatos.

Tres modelos responden las preguntas del producto de datos:

1. Pronóstico (regresión): ¿cuántos casos habrá en las próximas 1 a 4 semanas? Con el
   canal endémico se convierte en la alerta de brote (clasificación de cada semana).
2. Riesgo (clasificación): ¿este caso tendrá signos de alarma o será grave? Solo usa
   la información disponible en la primera consulta.
3. Perfiles de comunas (agrupamiento K-means): ¿cómo se organizan las 22 comunas?

El modelo de riesgo tiene un segundo uso con el mismo diseño: ¿el paciente requerirá
hospitalización? La clasificación clínica y los síntomas solo vienen en los casos del
Distrito (2022-2023), pero la hospitalización está en todas las fuentes, así que este
segundo modelo se entrena con los casos de 2020 a 2025 y solo con las variables que
traen todas las fuentes.

Validación temporal: los modelos se eligen con datos anteriores al periodo de prueba
y el periodo de prueba se usa una sola vez (pronóstico: se elige con 2023 y se prueba
con 2024-2025; signos de alarma: el 20 % más reciente de los casos es la prueba;
hospitalización: se entrena con 2020-2023 y se prueba con 2024-2025).
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flows.common.config import settings
from flows.common.epidemiologia import (
    canal_endemico,
    grupo_edad,
    siguientes_semanas,
    zona_canal,
)
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import (
    GOLD,
    GOLD_EVALUACION_PRONOSTICO,
    GOLD_INDICADORES_COMUNAS,
    GOLD_PERFILES_COMUNAS,
    GOLD_PRONOSTICO,
    MODELO_COMUNAS,
    MODELO_HOSPITALIZACION,
    MODELO_PRONOSTICO,
    MODELO_RIESGO,
    MODELS,
    SILVER,
    SILVER_CASOS,
    SILVER_SERIE,
)

log = get_logger(__name__)

# ===========================================================================
# Decisiones congeladas (resultado del cuaderno 04; justificadas en
# reports/problema_ml.md). Se cambian aquí, bajo revisión, no en el cuaderno.
# ===========================================================================
PRONOSTICO_MODELO_POR_HORIZONTE = {      # elegido en la validación (2023), ver abajo
    1: "ingenuo",                        # a una semana, nada mejora "igual que la última"
    2: "tendencia_25",                   # tendencia de 4 semanas amortiguada al 25 %
    3: "tendencia_25",
    4: "tendencia_50",
}
TOLERANCIA_EMPATE = 0.01                 # modelos a menos del 1 % del mejor MAE empatan
NIVEL_INTERVALO = 0.80                   # intervalo de pronóstico del 80 %
RIESGO_ESTIMADOR = "random_forest"
RIESGO_PARAMS = {                        # mejores hiperparámetros del GridSearchCV
    "model__max_depth": 10,
    "model__max_features": 0.2,
    "model__min_samples_leaf": 20,
    "model__n_estimators": 400,
}
HOSPITALIZACION_PARAMS = {               # GridSearchCV con los casos 2020-2023
    "model__max_depth": 10,
    "model__max_features": 0.5,
    "model__min_samples_leaf": 20,
    "model__n_estimators": 400,
}
RIESGO_UMBRAL = "youden"                 # punto de corte de la curva ROC fuera de muestra

# ===========================================================================
# 1. Pronóstico de casos semanales y alerta de brote
# ===========================================================================
TENDENCIA_REZAGO = 4                     # semanas con las que se mide la tendencia reciente
AMORTIGUAMIENTOS = (0.25, 0.5, 0.75)     # candidatos: cuánto de la tendencia se extrapola


VARIABLES_CASOS = "^(?:nivel|cambio_|sem_)"   # variables que usan los modelos sin clima
SEMANAS_CLIMA = 4                          # ventana del promedio de temperatura


def _features_origen(log_casos: np.ndarray, semanas_objetivo: np.ndarray,
                     n_rezagos: int, temperatura=None) -> pd.DataFrame:
    """Variables de cada semana de origen t (solo información conocida en t).

    - nivel: log(1 + casos) de la semana t.
    - cambio_k: cuánto cambió el log de los casos en las k semanas anteriores (k < n_rezagos).
    - sem_sin, sem_cos: posición en el año de la semana que se quiere pronosticar
      (perfil estacional; no depende de los datos, así que no filtra información).
    - temp_4s: anomalía de temperatura de la ciudad (IDEAM, diferencia con la media de cada
      estación y sistema de medición) promedio de las 4 semanas que terminan en t; nula si
      hay menos de dos semanas con lecturas suficientes.
    """
    nivel = pd.Series(log_casos)
    tabla = {"nivel": nivel}
    for k in range(1, n_rezagos):
        tabla[f"cambio_{k}"] = nivel - nivel.shift(k)
    angulo = 2 * np.pi * (np.minimum(semanas_objetivo, 52) - 1) / 52
    tabla["sem_sin"], tabla["sem_cos"] = np.sin(angulo), np.cos(angulo)
    temperatura = (pd.Series(np.nan, index=nivel.index) if temperatura is None
                   else pd.Series(np.asarray(temperatura, dtype=float)))
    tabla["temp_4s"] = temperatura.rolling(SEMANAS_CLIMA, min_periods=2).mean()
    return pd.DataFrame(tabla)


def _temperatura(serie: pd.DataFrame):
    if "temp_anomalia" not in serie:
        return None
    return serie["temp_anomalia"].astype("float64").to_numpy()


def dataset_pronostico(serie: pd.DataFrame, horizonte: int,
                       n_rezagos: int | None = None) -> tuple[pd.DataFrame, pd.Series,
                                                              pd.DataFrame]:
    """Una fila por semana de origen t: variables en t y objetivo en t + horizonte.

    El objetivo es el cambio relativo, log(1 + casos[t+h]) - log(1 + casos[t]): así un
    mismo modelo sirve para años de 2.700 y de 35.000 casos. Devuelve (X, y, meta);
    'meta' trae la semana objetivo, el caso observado y si alguna semana fue imputada.
    """
    n_rezagos = settings.n_rezagos if n_rezagos is None else n_rezagos
    serie = serie.sort_values(["anio_epi", "semana_epi"]).reset_index(drop=True)
    log_casos = np.log1p(serie["casos_modelo"].astype("float64").to_numpy())
    semana_obj = serie["semana_epi"].shift(-horizonte).to_numpy(dtype="float64")
    X = _features_origen(log_casos, semana_obj, n_rezagos, _temperatura(serie))
    y = pd.Series(log_casos).shift(-horizonte) - log_casos
    imputada = serie["imputada"].astype(bool)
    meta = pd.DataFrame({
        "anio_origen": serie["anio_epi"], "semana_origen": serie["semana_epi"],
        "anio_objetivo": serie["anio_epi"].shift(-horizonte),
        "semana_objetivo": serie["semana_epi"].shift(-horizonte),
        "fecha_objetivo": serie["fecha_inicio"].shift(-horizonte),
        "casos_objetivo": serie["casos"].astype("float64").shift(-horizonte),
        "log_base": log_casos,
        # Una semana imputada puede usar datos posteriores (interpolación): ni el origen
        # ni el objetivo pueden ser imputados para entrenar o evaluar.
        "usable": ~imputada & ~imputada.shift(-horizonte, fill_value=True),
    })
    valido = X.filter(regex=VARIABLES_CASOS).notna().all(axis=1) & y.notna()
    return (X[valido].reset_index(drop=True), y[valido].reset_index(drop=True),
            meta[valido].reset_index(drop=True).astype({"anio_objetivo": int,
                                                       "semana_objetivo": int}))


class Ingenuo(BaseEstimator, RegressorMixin):
    """Referencia: las próximas semanas tendrán los mismos casos que la última."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class TendenciaAmortiguada(BaseEstimator, RegressorMixin):
    """Extrapola la tendencia de las últimas semanas, amortiguada (escala logarítmica).

    Pronóstico = amortiguamiento × horizonte × pendiente semanal reciente, donde la
    pendiente es el cambio de las últimas 'rezago' semanas dividido por 'rezago'.
    El amortiguamiento (0 = ingenuo, 1 = tendencia completa) es un hiperparámetro: se
    compara en la validación como cualquier otro (no se aprende de los datos de
    entrenamiento, que en 2020-2022 tienen muy pocas semanas de crecimiento sostenido).
    """

    def __init__(self, horizonte: int = 1, amortiguamiento: float = 0.5,
                 rezago: int = TENDENCIA_REZAGO):
        self.horizonte = horizonte
        self.amortiguamiento = amortiguamiento
        self.rezago = rezago

    def _tendencia(self, X):
        return self.horizonte * np.asarray(X[f"cambio_{self.rezago}"], dtype=float) / self.rezago

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.amortiguamiento * self._tendencia(X)


class TendenciaClima(TendenciaAmortiguada):
    """Tendencia amortiguada más un efecto de la temperatura reciente.

    Pronóstico = amortiguamiento × tendencia + coeficiente × horizonte × (temp_4s − media),
    donde el coeficiente se ajusta por mínimos cuadrados con las semanas de entrenamiento
    que tienen temperatura. Cuando falta la temperatura, el pronóstico es la tendencia.
    """

    minimo_semanas = 10

    def fit(self, X, y):
        temperatura = np.asarray(X["temp_4s"], dtype=float)
        con_dato = ~np.isnan(temperatura)
        self.n_semanas_clima_ = int(con_dato.sum())
        self.temp_media_ = float(np.nanmean(temperatura)) if con_dato.any() else 0.0
        if self.n_semanas_clima_ < self.minimo_semanas:
            self.coeficiente_ = 0.0
            return self
        residuo = (np.asarray(y, dtype=float) - super().predict(X))[con_dato]
        z = self.horizonte * (temperatura[con_dato] - self.temp_media_)
        self.coeficiente_ = float((z @ residuo) / (z @ z)) if (z @ z) > 0 else 0.0
        return self

    def predict(self, X):
        anomalia = np.nan_to_num(np.asarray(X["temp_4s"], dtype=float) - self.temp_media_)
        return super().predict(X) + self.coeficiente_ * self.horizonte * anomalia


class CombinadoConIngenuo(BaseEstimator, RegressorMixin):
    """Ensamble con la referencia: peso × modelo + (1 - peso) × ingenuo (que vale 0)."""

    def __init__(self, estimador=None, peso: float = 0.5):
        self.estimador = estimador
        self.peso = peso

    def fit(self, X, y):
        self.estimador_ = clone(self.estimador).fit(X, y)
        return self

    def predict(self, X):
        return self.peso * self.estimador_.predict(X)


def get_candidatos_pronostico(horizonte: int, random_state: int | None = None) -> dict:
    """Modelos candidatos para un horizonte: {nombre: estimador sin entrenar}."""
    random_state = settings.random_state if random_state is None else random_state
    bosque = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                   random_state=random_state, n_jobs=-1)

    def solo_casos(modelo):
        """Los modelos de aprendizaje automático usan las variables de casos (sin nulos)."""
        seleccion = ColumnTransformer([("casos", "passthrough",
                                        make_column_selector(pattern=VARIABLES_CASOS))])
        return Pipeline([("variables", seleccion), ("model", modelo)])

    return {
        "ingenuo": Ingenuo(),
        **{f"tendencia_{int(a * 100)}": TendenciaAmortiguada(horizonte, a)
           for a in AMORTIGUAMIENTOS},
        **{f"tendencia_clima_{int(a * 100)}": TendenciaClima(horizonte, a)
           for a in AMORTIGUAMIENTOS},
        "ridge": solo_casos(Pipeline([("scaler", StandardScaler()),
                                      ("model", Ridge(alpha=10.0))])),
        "random_forest": solo_casos(bosque),
        "ensamble_rf_ingenuo": CombinadoConIngenuo(solo_casos(bosque), peso=0.5),
        "gradient_boosting": solo_casos(HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=random_state)),
    }


def metricas_pronostico(observado, pronostico) -> dict:
    """MAE, RMSE y MAPE (en casos) de un conjunto de semanas."""
    observado, pronostico = np.asarray(observado, float), np.asarray(pronostico, float)
    error = pronostico - observado
    con_casos = observado > 0
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "mape": float(np.abs(error[con_casos] / observado[con_casos]).mean() * 100),
        "n_semanas": int(len(observado)),
    }


def _predecir_periodo(serie, horizonte, nombre, anio_entrena_fin, anios_evalua,
                      random_state=None) -> pd.DataFrame:
    """Entrena con objetivos hasta 'anio_entrena_fin' y pronostica las semanas de
    'anios_evalua'. Solo usa semanas observadas (no imputadas) para entrenar y evaluar."""
    X, y, meta = dataset_pronostico(serie, horizonte)
    entrena = meta["usable"] & (meta["anio_objetivo"] <= anio_entrena_fin)
    evalua = meta["usable"] & meta["anio_objetivo"].isin(list(anios_evalua))
    modelo = get_candidatos_pronostico(horizonte, random_state)[nombre]
    modelo.fit(X[entrena], y[entrena])
    delta = modelo.predict(X[evalua])
    salida = meta[evalua].copy()
    salida["horizonte"] = horizonte
    salida["modelo"] = nombre
    salida["delta_pronostico"] = delta
    salida["delta_observado"] = y[evalua].to_numpy()
    salida["pronostico"] = np.expm1(salida["log_base"] + delta)
    return salida.reset_index(drop=True)


def comparar_pronostico(serie: pd.DataFrame, anio_validacion: int | None = None,
                        horizontes=None, random_state=None) -> pd.DataFrame:
    """Compara todos los candidatos en el año de validación (entrenan con los anteriores).

    Devuelve una fila por horizonte y modelo con MAE, RMSE, MAPE y la mejora de MAE
    frente al ingenuo. Esta tabla es la que decide el modelo; la prueba no se usa aquí.
    """
    anio_validacion = settings.anio_validacion if anio_validacion is None else anio_validacion
    horizontes = range(1, settings.horizonte_semanas + 1) if horizontes is None else horizontes
    filas = []
    for h in horizontes:
        for nombre in get_candidatos_pronostico(h, random_state):
            pred = _predecir_periodo(serie, h, nombre, anio_validacion - 1, [anio_validacion],
                                     random_state)
            filas.append({"horizonte": h, "modelo": nombre,
                          **metricas_pronostico(pred["casos_objetivo"], pred["pronostico"])})
    tabla = pd.DataFrame(filas)
    ingenuo = tabla[tabla["modelo"] == "ingenuo"].set_index("horizonte")["mae"]
    tabla["mejora_vs_ingenuo_pct"] = (1 - tabla["mae"] / tabla["horizonte"].map(ingenuo)) * 100
    return tabla.sort_values(["horizonte", "mae"]).reset_index(drop=True)


def complejidad(nombre: str) -> int:
    """Orden de simplicidad: ingenuo < tendencia < tendencia con clima < aprendizaje automático."""
    if nombre == "ingenuo":
        return 0
    if nombre.startswith("tendencia_clima"):
        return 2
    if nombre.startswith("tendencia"):
        return 1
    return 3


def elegir_por_horizonte(comparacion: pd.DataFrame,
                         tolerancia: float = TOLERANCIA_EMPATE) -> dict[int, str]:
    """En cada horizonte, el modelo más simple entre los que empatan con el mejor.

    Empatan los modelos cuyo MAE de validación está a menos de 'tolerancia' (1 %) del
    menor; entre ellos se prefiere el de menor complejidad y, a igual complejidad, el de
    menor MAE. Una diferencia más pequeña que eso no justifica un modelo más complejo.
    """
    elegidos = {}
    for h, grupo in comparacion.groupby("horizonte"):
        empatados = grupo[grupo["mae"] <= grupo["mae"].min() * (1 + tolerancia)]
        empatados = empatados.assign(nivel=empatados["modelo"].map(complejidad))
        elegidos[int(h)] = str(empatados.sort_values(["nivel", "mae"])["modelo"].iloc[0])
    return elegidos


def residuos_validacion(serie: pd.DataFrame, modelos: dict[int, str],
                        anio_validacion: int | None = None,
                        nivel: float = NIVEL_INTERVALO) -> dict[int, tuple[float, float]]:
    """Cuantiles de los errores (escala logarítmica) en la validación, por horizonte.

    Dan el intervalo de pronóstico: si el modelo erró entre -20 % y +35 % en el 80 %
    central de las semanas de validación, el intervalo de cada pronóstico usa ese rango.
    """
    anio_validacion = settings.anio_validacion if anio_validacion is None else anio_validacion
    cola = (1 - nivel) / 2
    residuos = {}
    for h, nombre in modelos.items():
        pred = _predecir_periodo(serie, h, nombre, anio_validacion - 1, [anio_validacion])
        error = pred["delta_observado"] - pred["delta_pronostico"]
        residuos[h] = (float(error.quantile(cola)), float(error.quantile(1 - cola)))
    return residuos


def canal_de_referencia(serie: pd.DataFrame, antes_de: int | None = None) -> pd.DataFrame:
    """Canal endémico con los años base anteriores a 'antes_de' (sin mirar el futuro).

    Para evaluar las alertas de 2024-2025 el canal solo puede usar años previos a 2024;
    el canal operativo (antes_de=None) usa todos los años base configurados.
    """
    anios = [a for a in settings.canal_anios_base if antes_de is None or a < antes_de]
    completas = serie[serie["cobertura"] == "completa"]
    canal = canal_endemico(completas, anios, settings.canal_ventana, valor="casos")
    canal["anios_base"] = ", ".join(str(a) for a in anios)
    return canal


def evaluar_pronostico(serie: pd.DataFrame, modelos: dict[int, str] | None = None,
                       anio_validacion: int | None = None,
                       anio_prueba_inicio: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Evaluación final (una sola vez) en el periodo de prueba.

    Cada modelo elegido se entrena con todas las semanas hasta el año de validación y
    pronostica 2024-2025. Devuelve las predicciones semana a semana (con el intervalo, el
    canal endémico de referencia y la alerta) y un resumen de métricas por horizonte,
    comparado con el pronóstico ingenuo en las mismas semanas.
    """
    modelos = PRONOSTICO_MODELO_POR_HORIZONTE if modelos is None else modelos
    anio_validacion = settings.anio_validacion if anio_validacion is None else anio_validacion
    inicio = settings.anio_prueba_inicio if anio_prueba_inicio is None else anio_prueba_inicio
    anios_prueba = sorted(a for a in serie["anio_epi"].unique() if a >= inicio)
    residuos = residuos_validacion(serie, modelos, anio_validacion)
    canal = canal_de_referencia(serie, antes_de=inicio)

    partes, resumen = [], {}
    for h, nombre in modelos.items():
        pred = _predecir_periodo(serie, h, nombre, anio_validacion, anios_prueba)
        ref = _predecir_periodo(serie, h, "ingenuo", anio_validacion, anios_prueba)
        bajo, alto = residuos[h]
        pred["limite_inferior"] = np.expm1(pred["log_base"] + pred["delta_pronostico"] + bajo)
        pred["limite_superior"] = np.expm1(pred["log_base"] + pred["delta_pronostico"] + alto)
        pred = alertas_de_brote(pred, canal)
        partes.append(pred)
        cubierto = pred["casos_objetivo"].between(pred["limite_inferior"],
                                                  pred["limite_superior"])
        metricas = metricas_pronostico(pred["casos_objetivo"], pred["pronostico"])
        metricas_ingenuo = metricas_pronostico(ref["casos_objetivo"], ref["pronostico"])
        resumen[h] = {
            "modelo": nombre, **metricas,
            "mae_ingenuo": metricas_ingenuo["mae"], "mape_ingenuo": metricas_ingenuo["mape"],
            "mejora_vs_ingenuo_pct": (1 - metricas["mae"] / metricas_ingenuo["mae"]) * 100,
            "cobertura_intervalo_pct": float(cubierto.mean() * 100),
            **metricas_alerta(pred),
        }
    resumen = {"anios_prueba": anios_prueba, "canal_anios_base": canal["anios_base"].iloc[0],
               "nivel_intervalo": NIVEL_INTERVALO, "por_horizonte": resumen}
    return pd.concat(partes, ignore_index=True), resumen


def alertas_de_brote(pred: pd.DataFrame, canal: pd.DataFrame) -> pd.DataFrame:
    """Agrega el canal de la semana objetivo, la alerta pronosticada y el brote observado.

    Hay alerta cuando el pronóstico supera el cuartil 3 del canal endémico (zona de
    brote); 'brote_observado' es lo mismo con los casos que realmente ocurrieron.
    """
    pred = pred.merge(canal[["semana_epi", "q1", "mediana", "q3"]], how="left",
                      left_on="semana_objetivo", right_on="semana_epi").drop(columns="semana_epi")
    pred["zona_pronostico"] = zona_canal(pred["pronostico"], pred["q1"], pred["mediana"],
                                         pred["q3"])
    pred["alerta"] = pred["pronostico"] > pred["q3"]
    if "casos_objetivo" in pred:
        pred["brote_observado"] = pred["casos_objetivo"] > pred["q3"]
    return pred


def metricas_alerta(pred: pd.DataFrame) -> dict:
    """Qué tan bien anticipa la alerta las semanas en brote (zona sobre el cuartil 3)."""
    tn, fp, fn, tp = confusion_matrix(pred["brote_observado"], pred["alerta"],
                                      labels=[False, True]).ravel()
    def razon(a, b):
        return float(a / b * 100) if b else None
    return {"semanas_brote": int(tp + fn), "semanas_sin_brote": int(tn + fp),
            "alerta_sensibilidad_pct": razon(tp, tp + fn),
            "alerta_precision_pct": razon(tp, tp + fp),
            "alerta_especificidad_pct": razon(tn, tn + fp)}


def entrenar_pronostico_final(serie: pd.DataFrame,
                              modelos: dict[int, str] | None = None) -> dict:
    """Modelo definitivo: cada horizonte se reentrena con todas las semanas observadas.

    El paquete guarda los modelos, los intervalos (errores de validación) y el canal
    endémico operativo: todo lo que necesita pronosticar() para dar el semáforo.
    """
    modelos = PRONOSTICO_MODELO_POR_HORIZONTE if modelos is None else modelos
    entrenados = {}
    for h, nombre in modelos.items():
        X, y, meta = dataset_pronostico(serie, h)
        modelo = get_candidatos_pronostico(h)[nombre]
        entrenados[h] = modelo.fit(X[meta["usable"]], y[meta["usable"]])
    return {"modelos": entrenados, "nombres": dict(modelos),
            "residuos": residuos_validacion(serie, modelos), "n_rezagos": settings.n_rezagos,
            "canal": canal_de_referencia(serie), "nivel_intervalo": NIVEL_INTERVALO}


def agregar_semanas(serie: pd.DataFrame, casos_nuevos) -> pd.DataFrame:
    """Agrega a la serie los casos de las semanas que siguen a la última (para actualizar
    el pronóstico con los datos más recientes)."""
    casos_nuevos = [float(c) for c in casos_nuevos]
    if not casos_nuevos:
        return serie
    serie = serie.sort_values(["anio_epi", "semana_epi"]).reset_index(drop=True)
    ultima = serie.iloc[-1]
    nuevas = siguientes_semanas(int(ultima["anio_epi"]), int(ultima["semana_epi"]),
                                len(casos_nuevos))
    nuevas = nuevas.assign(casos=casos_nuevos, casos_modelo=casos_nuevos, imputada=False,
                           cobertura="completa")
    return pd.concat([serie, nuevas], ignore_index=True)


def pronosticar(paquete: dict, serie: pd.DataFrame) -> pd.DataFrame:
    """Pronóstico de las próximas semanas a partir de la última semana de la serie.

    Devuelve una fila por horizonte con la semana, el pronóstico, su intervalo, el canal
    endémico de esa semana y el semáforo (zona del canal).
    """
    serie = serie.sort_values(["anio_epi", "semana_epi"]).reset_index(drop=True)
    n_rezagos = paquete["n_rezagos"]
    if len(serie) < n_rezagos:
        raise ValueError(f"Se necesitan al menos {n_rezagos} semanas para pronosticar")
    ultima = serie.iloc[-1]
    horizontes = sorted(paquete["modelos"])
    futuras = siguientes_semanas(int(ultima["anio_epi"]), int(ultima["semana_epi"]),
                                 max(horizontes))
    log_casos = np.log1p(serie["casos_modelo"].astype("float64").to_numpy())
    filas = []
    for h in horizontes:
        objetivo = futuras.iloc[h - 1]
        semanas = np.full(len(log_casos), objetivo["semana_epi"], dtype=float)
        X = _features_origen(log_casos, semanas, n_rezagos, _temperatura(serie)).iloc[[-1]]
        delta = float(paquete["modelos"][h].predict(X)[0])
        bajo, alto = paquete["residuos"][h]
        filas.append({"horizonte": h, "anio_epi": int(objetivo["anio_epi"]),
                      "semana_epi": int(objetivo["semana_epi"]),
                      "fecha_inicio": objetivo["fecha_inicio"],
                      "modelo": paquete["nombres"][h],
                      "pronostico": float(np.expm1(log_casos[-1] + delta)),
                      "limite_inferior": float(np.expm1(log_casos[-1] + delta + bajo)),
                      "limite_superior": float(np.expm1(log_casos[-1] + delta + alto))})
    salida = pd.DataFrame(filas).rename(columns={"semana_epi": "semana_objetivo"})
    salida = alertas_de_brote(salida, paquete["canal"])
    return salida.rename(columns={"semana_objetivo": "semana_epi"})


# ===========================================================================
# 2. Riesgo de signos de alarma o dengue grave
# ===========================================================================
TARGET_RIESGO = "severo"                 # con signos de alarma o dengue grave
TARGET_HOSPITALIZACION = "hospitalizado"  # conducta hospitalaria (segundo clasificador)
RIESGO_NUMERICAS = ["edad", "dias_sintomas_consulta"]
RIESGO_BINARIAS = ["migrante", "desplazamiento_15d", "fiebre", "cefalea", "dolor_retroocular",
                   "mialgias", "artralgia", "erupcion"]
RIESGO_CATEGORICAS = ["sexo", "estrato", "afiliacion", "comuna"]
RIESGO_VARIABLES = RIESGO_NUMERICAS + RIESGO_BINARIAS + RIESGO_CATEGORICAS
# Hospitalización: solo las variables que traen tanto el INS como el Distrito
HOSPITALIZACION_BINARIAS = ["migrante"]
HOSPITALIZACION_CATEGORICAS = ["sexo", "estrato", "afiliacion"]
# Excluidas: definen la clasificación (fuga de datos) u ocurren después de la consulta
RIESGO_EXCLUIDAS = ["dolor_abdominal", "vomito", "diarrea", "somnolencia", "hipotension",
                    "hepatomegalia", "hemorragia_mucosas", "hipotermia",
                    "aumento_hematocrito", "caida_plaquetas", "acumulacion_liquidos",
                    "extravasacion", "hemorragia", "choque", "dano_organo", "hospitalizado",
                    "conducta", "tipo_caso", "fallecido", "clasificacion"]
MAX_DIAS_CONSULTA = 30
SIN_DATO = "Sin dato"
NIVELES_RIESGO = ["Bajo", "Medio", "Alto"]
SINTOMAS_GENERALES = ["fiebre", "cefalea", "dolor_retroocular", "mialgias", "artralgia",
                      "erupcion"]


@dataclass(frozen=True)
class TareaClasificacion:
    """Qué predice un clasificador, con qué variables y con qué casos se entrena y prueba."""
    objetivo: str
    modelo: str
    descripcion: str
    numericas: tuple[str, ...]
    binarias: tuple[str, ...]
    categoricas: tuple[str, ...]
    solo_casos_distrito: bool            # solo casos con clasificación clínica (2022-2023)
    anio_prueba: int | None = None       # desde qué año es prueba; None: el 20 % más reciente
    datos: str = ""

    @property
    def variables(self) -> list[str]:
        return [*self.numericas, *self.binarias, *self.categoricas]


TAREAS_CLASIFICACION = {
    TARGET_RIESGO: TareaClasificacion(
        TARGET_RIESGO, MODELO_RIESGO,
        "riesgo de signos de alarma o dengue grave con datos de la primera consulta",
        tuple(RIESGO_NUMERICAS), tuple(RIESGO_BINARIAS), tuple(RIESGO_CATEGORICAS),
        solo_casos_distrito=True,
        datos="casos del Distrito 2022-2023, los únicos con clasificación clínica"),
    TARGET_HOSPITALIZACION: TareaClasificacion(
        TARGET_HOSPITALIZACION, MODELO_HOSPITALIZACION,
        "probabilidad de hospitalización con datos de la primera consulta",
        tuple(RIESGO_NUMERICAS), tuple(HOSPITALIZACION_BINARIAS),
        tuple(HOSPITALIZACION_CATEGORICAS), solo_casos_distrito=False, anio_prueba=2024,
        datos="casos de 2020 a 2025 del INS y del Distrito"),
}


def params_riesgo(objetivo: str = TARGET_RIESGO) -> dict:
    """Hiperparámetros congelados de cada clasificador (se leen al usarlos)."""
    return HOSPITALIZACION_PARAMS if objetivo == TARGET_HOSPITALIZACION else RIESGO_PARAMS


def preparar_variables_riesgo(casos: pd.DataFrame) -> pd.DataFrame:
    """Variables del modelo de riesgo con tipos uniformes (sirve para Silver y para la API).

    Numéricas como decimales (días fuera de 0-30 quedan nulos), binarias como 0/1 y
    categóricas como texto con 'Sin dato' cuando falta el valor.
    """
    X = pd.DataFrame(index=casos.index)
    X["edad"] = pd.to_numeric(casos["edad"], errors="coerce").astype("float64")
    dias = pd.to_numeric(casos["dias_sintomas_consulta"], errors="coerce").astype("float64")
    X["dias_sintomas_consulta"] = dias.where(dias.between(0, MAX_DIAS_CONSULTA))
    for columna in RIESGO_BINARIAS:
        X[columna] = pd.to_numeric(casos[columna].astype("object"), errors="coerce").astype(
            "float64")
    for columna in RIESGO_CATEGORICAS:
        valores = casos[columna].astype("object")
        if columna in ("estrato", "comuna"):
            valores = pd.to_numeric(valores, errors="coerce").astype("Int64").astype("object")
        X[columna] = valores.where(pd.notna(valores), SIN_DATO).astype(str)
    return X


def load_dataset_riesgo(path: Path | None = None, objetivo: str = TARGET_RIESGO
                        ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Casos con el objetivo conocido: (X, y, fecha de notificación).

    Signos de alarma: solo los casos del Distrito traen síntomas y clasificación final.
    Hospitalización: todos los casos de 2020 a 2025 (el dato viene en todas las fuentes).
    """
    tarea = TAREAS_CLASIFICACION[objetivo]
    casos = pd.read_parquet(Path(path) if path else SILVER / SILVER_CASOS)
    etiqueta = TARGET_RIESGO if tarea.solo_casos_distrito else objetivo
    casos = casos[casos[etiqueta].notna() & casos[objetivo].notna()].reset_index(drop=True)
    return (preparar_variables_riesgo(casos), casos[objetivo].astype(int),
            pd.to_datetime(casos["fecha_notificacion"]))


def split_temporal(X: pd.DataFrame, y: pd.Series, fechas: pd.Series,
                   test_size: float | None = None, anio_prueba: int | None = None):
    """La prueba son los casos más recientes, como ocurriría en uso real.

    Con 'anio_prueba', los notificados desde el 1 de enero de ese año (si hay casos
    antes y después de esa fecha); si no, el 'test_size' más reciente de los casos.
    """
    prueba = None
    if anio_prueba is not None:
        prueba = (fechas >= pd.Timestamp(year=anio_prueba, month=1, day=1)).to_numpy()
        if prueba.all() or not prueba.any():
            log.warning("No hay casos antes y después de %s: se usa el %.0f %% más reciente",
                        anio_prueba, 100 * (test_size or settings.test_size))
            prueba = None
    if prueba is None:
        test_size = settings.test_size if test_size is None else test_size
        prueba = (fechas >= fechas.quantile(1 - test_size)).to_numpy()
    return X[~prueba], X[prueba], y[~prueba], y[prueba]


def build_pipeline_riesgo(model, objetivo: str = TARGET_RIESGO) -> Pipeline:
    """Imputación + escalado de numéricas, one-hot de categóricas y el modelo.

    Recibe todas las columnas de 'preparar_variables_riesgo' y usa solo las de la tarea.
    """
    tarea = TAREAS_CLASIFICACION[objetivo]
    numericas = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())])
    binarias = SimpleImputer(strategy="most_frequent")
    categoricas = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
    prep = ColumnTransformer([("num", numericas, list(tarea.numericas)),
                              ("bin", binarias, list(tarea.binarias)),
                              ("cat", categoricas, list(tarea.categoricas))],
                             sparse_threshold=0)
    return Pipeline([("prep", prep), ("model", model)])


def get_candidatos_riesgo(random_state: int | None = None) -> dict:
    random_state = settings.random_state if random_state is None else random_state
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=2000, C=0.1),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=20,
                                                max_features=0.3, random_state=random_state,
                                                n_jobs=-1),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=random_state),
    }


def get_param_grid_riesgo() -> dict[str, dict]:
    return {
        "logreg": {"model__C": [0.01, 0.1, 1.0]},
        "random_forest": {"model__n_estimators": [400], "model__min_samples_leaf": [10, 20, 40],
                          "model__max_features": [0.2, 0.3, 0.5],
                          "model__max_depth": [None, 10]},
    }


def _cv(random_state=None) -> StratifiedKFold:
    random_state = settings.random_state if random_state is None else random_state
    return StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=random_state)


def comparar_riesgo(X: pd.DataFrame, y: pd.Series, random_state=None,
                    objetivo: str = TARGET_RIESGO) -> pd.DataFrame:
    """Validación cruzada de todos los candidatos: ROC-AUC, brecha y Brier por modelo."""
    filas = []
    for nombre, modelo in get_candidatos_riesgo(random_state).items():
        res = cross_validate(build_pipeline_riesgo(modelo, objetivo), X, y,
                             cv=_cv(random_state),
                             scoring=["roc_auc", "neg_brier_score"], return_train_score=True)
        filas.append({"modelo": nombre, "roc_auc": res["test_roc_auc"].mean(),
                      "roc_auc_std": res["test_roc_auc"].std(),
                      "roc_auc_entrenamiento": res["train_roc_auc"].mean(),
                      "brier": -res["test_neg_brier_score"].mean()})
    tabla = pd.DataFrame(filas)
    tabla["brecha"] = tabla["roc_auc_entrenamiento"] - tabla["roc_auc"]
    log.info("Clasificación - validación cruzada:\n%s", tabla.round(3).to_string(index=False))
    return tabla.sort_values("roc_auc", ascending=False).reset_index(drop=True)


def buscar_hiperparametros_riesgo(X, y, nombre: str = RIESGO_ESTIMADOR,
                                  random_state=None,
                                  objetivo: str = TARGET_RIESGO) -> GridSearchCV:
    """Búsqueda en malla con validación cruzada (solo datos de entrenamiento)."""
    modelo = get_candidatos_riesgo(random_state)[nombre]
    busqueda = GridSearchCV(build_pipeline_riesgo(modelo, objetivo),
                            get_param_grid_riesgo()[nombre], scoring="roc_auc",
                            cv=_cv(random_state), n_jobs=-1)
    return busqueda.fit(X, y)


def umbrales_riesgo(y, proba) -> dict:
    """Punto de corte (Youden: máxima sensibilidad + especificidad) y niveles de riesgo.

    Se calculan con probabilidades fuera de muestra del entrenamiento. Los niveles Bajo,
    Medio y Alto dividen los casos en tercios según su probabilidad.
    """
    fpr, tpr, cortes = roc_curve(y, proba)
    youden = float(cortes[np.argmax(tpr - fpr)])
    bajo, alto = np.quantile(proba, [1 / 3, 2 / 3])
    return {"umbral": youden, "corte_medio": float(bajo), "corte_alto": float(alto)}


def entrenar_riesgo(X, y, params: dict | None = None, nombre: str = RIESGO_ESTIMADOR,
                    random_state=None, objetivo: str = TARGET_RIESGO) -> tuple[Pipeline, dict]:
    """Ajusta el pipeline y calcula sus umbrales con predicciones fuera de muestra."""
    params = params_riesgo(objetivo) if params is None else params
    modelo = build_pipeline_riesgo(get_candidatos_riesgo(random_state)[nombre],
                                   objetivo).set_params(**params)
    oof = cross_val_predict(modelo, X, y, cv=_cv(random_state), method="predict_proba")[:, 1]
    return modelo.fit(X, y), umbrales_riesgo(y, oof)


def nivel_riesgo(proba, umbrales: dict) -> np.ndarray:
    return np.select([proba >= umbrales["corte_alto"], proba >= umbrales["corte_medio"]],
                     NIVELES_RIESGO[2:0:-1], default=NIVELES_RIESGO[0])


def predecir_riesgo(pipeline: Pipeline, X: pd.DataFrame, umbrales: dict) -> pd.DataFrame:
    """Probabilidad, clasificación con el umbral y nivel de riesgo de cada caso."""
    proba = pipeline.predict_proba(X[RIESGO_VARIABLES])[:, 1]
    return pd.DataFrame({"probabilidad": proba,
                         "supera_umbral": proba >= umbrales["umbral"],
                         "nivel_riesgo": nivel_riesgo(proba, umbrales)}, index=X.index)


# Grupos de variables para explicar un resultado (etiqueta -> columnas)
GRUPOS_EXPLICACION = {
    "Edad": ["edad"],
    "Días desde el inicio de los síntomas": ["dias_sintomas_consulta"],
    "Sexo": ["sexo"],
    "Estrato": ["estrato"],
    "Comuna": ["comuna"],
    "Régimen de afiliación": ["afiliacion"],
    "Migrante o viaje reciente": ["migrante", "desplazamiento_15d"],
    "Síntomas generales": SINTOMAS_GENERALES,
}


def muestra_de_fondo(X: pd.DataFrame, n: int = 200, random_state=None) -> pd.DataFrame:
    """Casos reales al azar con los que se compara cada característica de un caso nuevo."""
    random_state = settings.random_state if random_state is None else random_state
    return X[RIESGO_VARIABLES].sample(min(n, len(X)), random_state=random_state
                                      ).reset_index(drop=True)


def contribuciones(pipeline: Pipeline, X_caso: pd.DataFrame, fondo: pd.DataFrame,
                   variables: list[str]) -> pd.DataFrame:
    """Cuánto pesa cada característica del caso en su probabilidad.

    Para cada grupo de variables se reemplaza el valor del caso por el de cada caso de
    la muestra de fondo (casos reales) y se promedia la probabilidad resultante. La
    diferencia con la probabilidad del caso, en puntos porcentuales, indica cuánto sube
    (positivo) o baja (negativo) esa característica el resultado frente a los valores
    habituales. Solo se informan los grupos que usa el modelo.
    """
    X_caso = X_caso[RIESGO_VARIABLES].iloc[[0]].reset_index(drop=True)
    grupos = {etiqueta: [c for c in columnas if c in variables]
              for etiqueta, columnas in GRUPOS_EXPLICACION.items()}
    grupos = {etiqueta: usadas for etiqueta, usadas in grupos.items() if usadas}
    repetido = X_caso.loc[np.zeros(len(fondo), dtype=int)].reset_index(drop=True)
    alterados = []
    for usadas in grupos.values():
        alterado = repetido.copy()
        alterado[usadas] = fondo[usadas].to_numpy()
        alterados.append(alterado)
    proba = pipeline.predict_proba(pd.concat([X_caso, *alterados], ignore_index=True))[:, 1]
    promedios = proba[1:].reshape(len(grupos), len(fondo)).mean(axis=1)
    filas = [{"factor": etiqueta, "variables": usadas, "puntos": float((proba[0] - media) * 100)}
             for (etiqueta, usadas), media in zip(grupos.items(), promedios, strict=True)]
    tabla = pd.DataFrame(filas, columns=["factor", "variables", "puntos"])
    orden = tabla["puntos"].abs().sort_values(ascending=False).index
    return tabla.loc[orden].reset_index(drop=True)


def bootstrap_ic(y, proba, metric=roc_auc_score, n_boot: int = 1000,
                 random_state=None) -> tuple[float, float]:
    """Intervalo de confianza del 95 % por remuestreo (percentiles 2,5 y 97,5)."""
    rng = np.random.default_rng(settings.random_state if random_state is None else random_state)
    y, proba = np.asarray(y), np.asarray(proba)
    valores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            valores.append(metric(y[idx], proba[idx]))
    bajo, alto = np.percentile(valores, [2.5, 97.5])
    return float(bajo), float(alto)


def metricas_por_subgrupo(X: pd.DataFrame, y, proba, umbral: float,
                          min_casos: int = 50) -> pd.DataFrame:
    """AUC, sensibilidad y especificidad por sexo, grupo de edad, afiliación y estrato.

    Sirve para revisar si el modelo funciona peor en algún grupo de población. Solo se
    informan los grupos con al menos 'min_casos' casos y con casos de ambas clases.
    """
    tabla = pd.DataFrame({"y": np.asarray(y), "proba": np.asarray(proba)}, index=X.index)
    tabla["clase"] = tabla["proba"] >= umbral
    grupos = {"sexo": X["sexo"], "grupo_edad": grupo_edad(X["edad"]),
              "afiliacion": X["afiliacion"], "estrato": X["estrato"]}
    filas = []
    for variable, valores in grupos.items():
        for categoria, sub in tabla.groupby(valores.to_numpy()):
            if len(sub) < min_casos or sub["y"].nunique() < 2:
                continue
            positivos, negativos = sub[sub["y"] == 1], sub[sub["y"] == 0]
            filas.append({"variable": variable, "categoria": str(categoria), "n": len(sub),
                          "prevalencia_pct": sub["y"].mean() * 100,
                          "roc_auc": roc_auc_score(sub["y"], sub["proba"]),
                          "sensibilidad": positivos["clase"].mean(),
                          "especificidad": 1 - negativos["clase"].mean()})
    return pd.DataFrame(filas)


def evaluar_riesgo(pipeline, X_test, y_test, umbrales: dict) -> dict:
    """Evaluación final en la prueba: AUC con intervalo, sensibilidad, especificidad..."""
    pred = predecir_riesgo(pipeline, X_test, umbrales)
    proba, clase = pred["probabilidad"].to_numpy(), pred["supera_umbral"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_test, clase, labels=[0, 1]).ravel()
    bajo, alto = bootstrap_ic(y_test, proba)
    por_nivel = pd.Series(np.asarray(y_test)).groupby(pred["nivel_riesgo"].to_numpy()).agg(
        ["mean", "size"])
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)), "roc_auc_ic95": [bajo, alto],
        "sensibilidad": float(tp / (tp + fn)), "especificidad": float(tn / (tn + fp)),
        "valor_predictivo_positivo": float(tp / (tp + fp)) if tp + fp else None,
        "valor_predictivo_negativo": float(tn / (tn + fn)) if tn + fn else None,
        "brier": float(brier_score_loss(y_test, proba)),
        "prevalencia": float(np.mean(y_test)), "n_test": int(len(y_test)),
        "pct_positivos_por_nivel": {k: float(v * 100) for k, v in por_nivel["mean"].items()},
        "casos_por_nivel": {k: int(v) for k, v in por_nivel["size"].items()},
        "por_subgrupo": metricas_por_subgrupo(X_test, y_test, proba, umbrales["umbral"]
                                              ).round(3).to_dict(orient="records"),
    }


# ===========================================================================
# 3. Perfiles de comunas (agrupamiento)
# ===========================================================================
VARIABLES_COMUNAS = ["tasa_100k", "pct_severos", "pct_hospitalizados", "edad_mediana",
                     "estrato_medio", "pct_subsidiado", "hosp_arbovirosis_100k"]
RASGOS_COMUNAS = {                       # cómo se lee cada variable cuando está alta o baja
    "tasa_100k": ("incidencia alta", "incidencia baja"),
    "pct_severos": ("severidad alta", "severidad baja"),
    "pct_hospitalizados": ("más hospitalización", "menos hospitalización"),
    "edad_mediana": ("pacientes mayores", "pacientes más jóvenes"),
    "estrato_medio": ("estrato alto", "estrato bajo"),
    "pct_subsidiado": ("más régimen subsidiado", "menos régimen subsidiado"),
    "hosp_arbovirosis_100k": ("carga hospitalaria histórica alta",
                              "carga hospitalaria histórica baja"),
}


def load_dataset_comunas(path: Path | None = None) -> pd.DataFrame:
    """Indicadores de las 22 comunas urbanas (los corregimientos tienen muy pocos casos)."""
    tabla = pd.read_parquet(Path(path) if path else GOLD / GOLD_INDICADORES_COMUNAS)
    urbanas = tabla[tabla["zona"] == "Urbana"].dropna(subset=VARIABLES_COMUNAS)
    return urbanas.set_index("comuna").sort_index()


def build_pipeline_comunas(k: int, random_state=None) -> Pipeline:
    random_state = settings.random_state if random_state is None else random_state
    return Pipeline([("scaler", StandardScaler()),
                     ("kmeans", KMeans(n_clusters=k, n_init=50, random_state=random_state))])


def comparar_k(comunas: pd.DataFrame, ks=range(2, 7), random_state=None) -> pd.DataFrame:
    """Coeficiente de silueta e inercia para cada número de grupos."""
    filas = []
    for k in ks:
        pipe = build_pipeline_comunas(k, random_state).fit(comunas[VARIABLES_COMUNAS])
        Z = pipe["scaler"].transform(comunas[VARIABLES_COMUNAS])
        etiquetas = pipe["kmeans"].labels_
        filas.append({"k": k, "silueta": float(silhouette_score(Z, etiquetas)),
                      "inercia": float(pipe["kmeans"].inertia_),
                      "tamano_minimo": int(np.bincount(etiquetas).min())})
    return pd.DataFrame(filas)


def describir_grupo(centro_estandarizado: pd.Series, n: int = 3) -> str:
    """Descripción automática con los 'n' rasgos más marcados del grupo.

    El centro está estandarizado (0 = promedio de las comunas), así que los rasgos con
    mayor valor absoluto son los que más distinguen al grupo de las demás comunas.
    """
    rasgos = centro_estandarizado.reindex(
        centro_estandarizado.abs().sort_values(ascending=False).index)[:n]
    texto = ", ".join(RASGOS_COMUNAS[v][0 if z > 0 else 1] for v, z in rasgos.items())
    return texto[:1].upper() + texto[1:]


def ajustar_perfiles(comunas: pd.DataFrame, k: int | None = None,
                     random_state=None) -> tuple[Pipeline, pd.DataFrame]:
    """Agrupa las comunas y nombra los grupos A, B, C... de mayor a menor incidencia."""
    k = settings.n_grupos_comunas if k is None else k
    pipe = build_pipeline_comunas(k, random_state).fit(comunas[VARIABLES_COMUNAS])
    etiquetas = pd.Series(pipe["kmeans"].labels_, index=comunas.index)
    orden = comunas.groupby(etiquetas)["tasa_100k"].mean().sort_values(ascending=False).index
    letra = {g: chr(ord("A") + i) for i, g in enumerate(orden)}
    centros = pd.DataFrame(pipe["kmeans"].cluster_centers_, columns=VARIABLES_COMUNAS)
    descripcion = {g: describir_grupo(centros.loc[g]) for g in letra}
    perfiles = comunas[["territorio", *VARIABLES_COMUNAS]].copy()
    perfiles["grupo"] = etiquetas.map(letra)
    perfiles["perfil"] = etiquetas.map(descripcion)
    Z = pipe["scaler"].transform(comunas[VARIABLES_COMUNAS])
    perfiles.attrs["silueta"] = float(silhouette_score(Z, etiquetas))
    pipe.letras_ = letra
    pipe.descripciones_ = {letra[g]: d for g, d in descripcion.items()}
    return pipe, perfiles.reset_index()


# ===========================================================================
# Guardar y cargar modelos (con metadatos para auditarlos)
# ===========================================================================

def save_model(objeto, metadata: dict, name: str, models_dir: Path = MODELS) -> Path:
    """Guarda el modelo (.joblib) y su archivo de metadatos (.metadata.json)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(objeto, model_path)
    meta = {
        **metadata,
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "random_state": settings.random_state,
        "model_sha256": sha256_of(model_path),
    }
    (models_dir / f"{name}.metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("Modelo guardado en %s", model_path)
    return model_path


def load_model(path: Path) -> tuple[object, dict]:
    """Carga un modelo y sus metadatos; avisa si cambió la versión de scikit-learn.

    Los .joblib pueden ejecutar código al cargarse: solo se cargan modelos propios.
    """
    path = Path(path)
    objeto = joblib.load(path)
    meta = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning("Modelo entrenado con scikit-learn %s; versión actual %s",
                    meta.get("sklearn_version"), sklearn.__version__)
    return objeto, meta


# ===========================================================================
# Orquestación
# ===========================================================================

def _run_pronostico(serie_path: Path, models_dir: Path, gold_dir: Path) -> dict:
    serie = pd.read_parquet(serie_path)
    comparacion = comparar_pronostico(serie)
    log.info("Pronóstico - validación (%d):\n%s", settings.anio_validacion,
             comparacion.round(1).to_string(index=False))
    elegidos = elegir_por_horizonte(comparacion)
    if elegidos != PRONOSTICO_MODELO_POR_HORIZONTE:
        log.warning("La validación elige %s y el congelado es %s: revisar el cuaderno 04",
                    elegidos, PRONOSTICO_MODELO_POR_HORIZONTE)
    predicciones, resumen = evaluar_pronostico(serie)
    for h, m in resumen["por_horizonte"].items():
        log.info("Pronóstico h=%d (%s): MAE %.1f vs ingenuo %.1f; alerta sens. %.1f %%", h,
                 m["modelo"], m["mae"], m["mae_ingenuo"], m["alerta_sensibilidad_pct"])
    paquete = entrenar_pronostico_final(serie)
    proximas = pronosticar(paquete, serie)
    gold_dir.mkdir(parents=True, exist_ok=True)
    predicciones.to_parquet(gold_dir / GOLD_EVALUACION_PRONOSTICO, index=False)
    proximas.to_parquet(gold_dir / GOLD_PRONOSTICO, index=False)
    ultima = serie.sort_values(["anio_epi", "semana_epi"]).iloc[-1]
    save_model(paquete, {
        "tarea": "pronóstico de casos semanales de dengue a 1-4 semanas y alerta de brote",
        "modelos": paquete["nombres"], "n_rezagos": paquete["n_rezagos"],
        "variables": list(_features_origen(np.zeros(1), np.ones(1), paquete["n_rezagos"])),
        "objetivo": "log(1 + casos[t+h]) - log(1 + casos[t])",
        "amortiguamiento": {h: getattr(m, "amortiguamiento", None)
                            for h, m in paquete["modelos"].items()},
        "residuos_intervalo": paquete["residuos"],
        "canal_anios_base": paquete["canal"]["anios_base"].iloc[0],
        "ultima_semana_entrenamiento": f"{int(ultima['anio_epi'])}-S{int(ultima['semana_epi'])}",
        "n_semanas": int(len(serie)), "data_sha256": sha256_of(serie_path),
        "validacion": comparacion.to_dict(orient="records"), "prueba": resumen,
    }, MODELO_PRONOSTICO, models_dir)
    return resumen


def _run_riesgo(casos_path: Path, models_dir: Path, objetivo: str = TARGET_RIESGO) -> dict:
    tarea = TAREAS_CLASIFICACION[objetivo]
    X, y, fechas = load_dataset_riesgo(casos_path, objetivo)
    X_tr, X_te, y_tr, y_te = split_temporal(X, y, fechas, anio_prueba=tarea.anio_prueba)
    comparacion = comparar_riesgo(X_tr, y_tr, objetivo=objetivo)
    pipeline, umbrales = entrenar_riesgo(X_tr, y_tr, objetivo=objetivo)
    prueba = evaluar_riesgo(pipeline, X_te, y_te, umbrales)
    log.info("%s - prueba: AUC %.3f (IC95 %.3f-%.3f), sens. %.2f, espec. %.2f", objetivo,
             prueba["roc_auc"], *prueba["roc_auc_ic95"], prueba["sensibilidad"],
             prueba["especificidad"])
    final, umbrales_finales = entrenar_riesgo(X, y, objetivo=objetivo)   # todos los casos
    paquete = {"pipeline": final, "umbrales": umbrales_finales, "objetivo": objetivo,
               "variables": tarea.variables, "fondo": muestra_de_fondo(X),
               "prevalencia": float(y.mean())}
    save_model(paquete, {
        "tarea": tarea.descripcion, "datos": tarea.datos,
        "estimador": RIESGO_ESTIMADOR, "params": params_riesgo(objetivo),
        "umbral_criterio": RIESGO_UMBRAL,
        "umbrales": umbrales_finales, "objetivo": objetivo,
        "variables": {"numericas": list(tarea.numericas), "binarias": list(tarea.binarias),
                      "categoricas": list(tarea.categoricas)},
        "variables_excluidas": RIESGO_EXCLUIDAS,
        "prevalencia": float(y.mean()),
        "periodo": f"{fechas.min().year}-{fechas.max().year}",
        "n_train": int(len(X)), "n_entrenamiento_evaluacion": int(len(X_tr)),
        "n_prueba": int(len(X_te)),
        "fecha_corte_prueba": str(fechas[fechas.index.isin(X_te.index)].min().date()),
        "data_sha256": sha256_of(casos_path),
        "validacion_cruzada": comparacion.to_dict(orient="records"), "prueba": prueba,
    }, tarea.modelo, models_dir)
    return prueba


def _run_comunas(indicadores_path: Path, models_dir: Path, gold_dir: Path) -> dict:
    comunas = load_dataset_comunas(indicadores_path)
    seleccion = comparar_k(comunas)
    pipeline, perfiles = ajustar_perfiles(comunas)
    perfiles.to_parquet(gold_dir / GOLD_PERFILES_COMUNAS, index=False)
    grupos = perfiles.groupby("grupo")["comuna"].apply(lambda s: [int(c) for c in s]).to_dict()
    log.info("Perfiles de comunas (silueta %.3f): %s", perfiles.attrs["silueta"], grupos)
    save_model(pipeline, {
        "tarea": "perfiles epidemiológicos y socioeconómicos de las 22 comunas (K-means)",
        "k": settings.n_grupos_comunas, "variables": VARIABLES_COMUNAS,
        "silueta": perfiles.attrs["silueta"], "grupos": grupos,
        "descripciones": pipeline.descripciones_,
        "comparacion_k": seleccion.to_dict(orient="records"),
        "n_comunas": int(len(comunas)), "data_sha256": sha256_of(indicadores_path),
    }, MODELO_COMUNAS, models_dir)
    return {"silueta": perfiles.attrs["silueta"], "grupos": grupos}


def run(in_dir: Path | None = None, gold_dir: Path = GOLD, models_dir: Path = MODELS) -> Path:
    """Evalúa honestamente, entrena los modelos definitivos y los guarda."""
    in_dir, gold_dir, models_dir = Path(in_dir or SILVER), Path(gold_dir), Path(models_dir)
    _run_pronostico(in_dir / SILVER_SERIE, models_dir, gold_dir)
    _run_riesgo(in_dir / SILVER_CASOS, models_dir, TARGET_RIESGO)
    _run_riesgo(in_dir / SILVER_CASOS, models_dir, TARGET_HOSPITALIZACION)
    _run_comunas(gold_dir / GOLD_INDICADORES_COMUNAS, models_dir, gold_dir)
    return models_dir
