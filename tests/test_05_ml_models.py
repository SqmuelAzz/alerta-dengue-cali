import importlib
import json

import numpy as np
import pandas as pd
import pytest

from flows.common.epidemiologia import calendario_epidemiologico

ml = importlib.import_module("flows.05_ml_models")


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------

def _serie_sintetica(seed: int = 0) -> pd.DataFrame:
    """Serie 2020-2025 con estacionalidad, una epidemia en 2024 y algunas semanas imputadas."""
    rng = np.random.default_rng(seed)
    serie = calendario_epidemiologico(2020, 2025)
    t = np.arange(len(serie))
    base = 80 + 40 * np.sin(2 * np.pi * serie["semana_epi"].to_numpy() / 52)
    epidemia = 600 * np.exp(-0.5 * ((t - 230) / 10) ** 2)
    casos = np.round(base + epidemia + rng.normal(0, 5, len(t))).clip(1)
    serie["casos"] = casos
    serie["imputada"] = False
    serie.loc[[100, 101, 102], "imputada"] = True                 # un hueco a mitad de 2021
    serie["casos_modelo"] = serie["casos"].astype(float)
    serie.loc[serie["imputada"], "casos"] = np.nan
    serie["cobertura"] = np.where(serie["imputada"], "sin_datos", "completa")
    return serie


def _casos_sinteticos(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Casos con clasificación: los mayores y los que consultan tarde son más severos."""
    rng = np.random.default_rng(seed)
    edad = rng.integers(1, 90, n).astype(float)
    dias = rng.integers(0, 8, n)
    logit = -2 + 0.04 * edad + 0.3 * dias
    severo = rng.random(n) < 1 / (1 + np.exp(-logit + 2))
    booleano = lambda p: pd.array(rng.random(n) < p, dtype="boolean")   # noqa: E731
    casos = pd.DataFrame({
        "id_caso": [f"c{i}" for i in range(n)],
        "fecha_notificacion": pd.Timestamp("2022-01-01") + pd.to_timedelta(
            np.sort(rng.integers(0, 700, n)), unit="D"),
        "edad": edad,
        "dias_sintomas_consulta": pd.array(dias, dtype="Int64"),
        "sexo": rng.choice(["F", "M"], n),
        "estrato": pd.array(rng.integers(1, 7, n), dtype="Int64"),
        "afiliacion": rng.choice(["Contributivo", "Subsidiado", None], n),
        "comuna": pd.array(rng.choice([1, 6, 14, 22], n), dtype="Int64"),
        "severo": pd.array(severo, dtype="boolean"),
    })
    for columna in ml.RIESGO_BINARIAS:
        casos[columna] = booleano(0.5)
    casos["hospitalizado"] = (severo & (rng.random(n) < 0.8)) | (rng.random(n) < 0.1)
    casos.loc[0, "severo"] = pd.NA                                 # sin clasificación
    return casos


def _comunas_sinteticas() -> pd.DataFrame:
    """Tres grupos de comunas bien separados (alta incidencia, estrato alto y estrato bajo)."""
    rng = np.random.default_rng(1)
    centros = {
        "alta": [800, 50, 30, 20, 2.5, 25, 300],
        "estrato_alto": [450, 35, 20, 25, 4.5, 8, 120],
        "estrato_bajo": [380, 38, 22, 18, 1.8, 35, 140],
    }
    filas = []
    for centro in np.repeat(list(centros.values()), 6, axis=0):
        filas.append(np.asarray(centro) * (1 + rng.normal(0, 0.03, 7)))
    tabla = pd.DataFrame(filas, columns=ml.VARIABLES_COMUNAS)
    tabla["comuna"] = np.arange(1, 19)
    tabla["territorio"] = [f"Comuna {c}" for c in tabla["comuna"]]
    tabla["zona"] = "Urbana"
    return tabla


@pytest.fixture(scope="module")
def serie():
    return _serie_sintetica()


# ---------------------------------------------------------------------------
# Pronóstico
# ---------------------------------------------------------------------------

def test_dataset_pronostico_uses_only_the_past(serie):
    X, y, meta = ml.dataset_pronostico(serie, horizonte=2, n_rezagos=4)
    futuro = serie.copy()
    futuro.loc[futuro.index[-1], "casos_modelo"] = 1e6             # cambiar el último dato
    X2, _, _ = ml.dataset_pronostico(futuro, horizonte=2, n_rezagos=4)
    # Las variables de un origen no cambian cuando cambia una semana posterior
    pd.testing.assert_frame_equal(X.iloc[:-1], X2.iloc[: len(X) - 1])
    fila = meta.index[10]
    origen = (serie["anio_epi"] == meta.loc[fila, "anio_origen"]) & (
        serie["semana_epi"] == meta.loc[fila, "semana_origen"])
    i = serie.index[origen][0]
    esperado = np.log1p(serie.loc[i + 2, "casos_modelo"]) - np.log1p(serie.loc[i, "casos_modelo"])
    assert y.loc[fila] == pytest.approx(esperado)
    assert X.filter(regex=ml.VARIABLES_CASOS).notna().all().all()
    assert X["temp_4s"].isna().all()                  # la serie sintética no trae clima
    assert len(X) == len(serie) - 3 - 2


def test_imputed_weeks_are_not_usable_as_origin_or_target(serie):
    _, _, meta = ml.dataset_pronostico(serie, horizonte=1, n_rezagos=4)
    no_usables = meta[~meta["usable"]]
    semanas_imputadas = set(zip(serie.loc[serie["imputada"], "anio_epi"],
                                serie.loc[serie["imputada"], "semana_epi"], strict=True))
    for _, fila in no_usables.iterrows():
        origen = (fila["anio_origen"], fila["semana_origen"])
        objetivo = (fila["anio_objetivo"], fila["semana_objetivo"])
        assert origen in semanas_imputadas or objetivo in semanas_imputadas
    assert len(no_usables) == 4                       # 3 semanas imputadas + la anterior


def test_ingenuo_and_damped_trend():
    X = pd.DataFrame({"cambio_4": [np.log(2.0)], "nivel": [np.log(100.0)]})
    assert ml.Ingenuo().fit(X).predict(X)[0] == 0
    completa = ml.TendenciaAmortiguada(horizonte=4, amortiguamiento=1.0).fit(X)
    assert completa.predict(X)[0] == pytest.approx(np.log(2.0))       # se duplica otra vez
    media = ml.TendenciaAmortiguada(horizonte=4, amortiguamiento=0.5)
    assert media.predict(X)[0] == pytest.approx(0.5 * np.log(2.0))


def test_combinado_con_ingenuo_halves_the_model():
    X, y = pd.DataFrame({"a": np.arange(20.0)}), pd.Series(np.arange(20.0))
    from sklearn.linear_model import LinearRegression
    combinado = ml.CombinadoConIngenuo(LinearRegression(), peso=0.5).fit(X, y)
    assert combinado.predict(X.iloc[[10]])[0] == pytest.approx(5.0)


def test_metricas_pronostico():
    m = ml.metricas_pronostico([10, 20, 0], [12, 18, 1])
    assert m["mae"] == pytest.approx(5 / 3)
    assert m["mape"] == pytest.approx((0.2 + 0.1) / 2 * 100)      # sin las semanas en cero
    assert m["n_semanas"] == 3


def test_comparar_pronostico_and_choose_by_validation(serie):
    tabla = ml.comparar_pronostico(serie, anio_validacion=2023, horizontes=[1, 2])
    assert set(tabla["modelo"]) == set(ml.get_candidatos_pronostico(1))
    assert len(tabla) == 2 * len(ml.get_candidatos_pronostico(1))
    ingenuo = tabla[tabla["modelo"] == "ingenuo"]
    assert (ingenuo["mejora_vs_ingenuo_pct"] == 0).all()
    elegidos = ml.elegir_por_horizonte(tabla)
    for h, nombre in elegidos.items():
        sub = tabla[tabla["horizonte"] == h]
        assert sub.set_index("modelo").loc[nombre, "mae"] <= sub["mae"].min() * 1.01


def test_elegir_prefers_the_simplest_model_when_errors_tie():
    tabla = pd.DataFrame({"horizonte": [1, 1, 1, 2, 2],
                          "modelo": ["random_forest", "tendencia_clima_25", "tendencia_25",
                                     "random_forest", "ingenuo"],
                          "mae": [10.0, 10.05, 10.08, 8.0, 9.0]})
    elegidos = ml.elegir_por_horizonte(tabla)
    assert elegidos[1] == "tendencia_25"           # empate (< 1 %): gana el más simple
    assert elegidos[2] == "random_forest"          # 12 % mejor: la complejidad se justifica


def test_canal_de_referencia_does_not_look_at_the_test_years(serie):
    canal = ml.canal_de_referencia(serie, antes_de=2024)
    assert canal["anios_base"].iloc[0] == "2021, 2022"             # sin 2025
    assert len(canal) == 53


def test_evaluar_pronostico_reports_test_metrics_and_alerts(serie):
    modelos = {1: "ingenuo", 2: "tendencia_50"}
    pred, resumen = ml.evaluar_pronostico(serie, modelos, anio_validacion=2023,
                                          anio_prueba_inicio=2024)
    assert set(pred["anio_objetivo"]) == {2024, 2025}
    assert (pred["limite_inferior"] <= pred["pronostico"] + 1e-9).all()
    assert (pred["pronostico"] <= pred["limite_superior"] + 1e-9).all()
    h2 = resumen["por_horizonte"][2]
    assert 0 <= h2["cobertura_intervalo_pct"] <= 100
    assert h2["semanas_brote"] > 0 and h2["alerta_sensibilidad_pct"] > 50   # la epidemia de 2024
    assert resumen["por_horizonte"][1]["mejora_vs_ingenuo_pct"] == 0


def test_alertas_y_metricas_de_alerta():
    pred = pd.DataFrame({"semana_objetivo": [1, 2, 3, 4], "pronostico": [3, 50, 50, 3],
                         "casos_objetivo": [3, 50, 3, 50]})
    canal = pd.DataFrame({"semana_epi": [1, 2, 3, 4], "q1": 2.0, "mediana": 5.0, "q3": 10.0})
    out = ml.alertas_de_brote(pred, canal)
    assert out["alerta"].tolist() == [False, True, True, False]
    assert out["zona_pronostico"].tolist() == ["Seguridad", "Brote", "Brote", "Seguridad"]
    m = ml.metricas_alerta(out)
    assert (m["alerta_sensibilidad_pct"], m["alerta_precision_pct"]) == (50.0, 50.0)


def test_pronosticar_gives_the_next_weeks_with_semaforo(serie):
    paquete = ml.entrenar_pronostico_final(serie, {1: "ingenuo", 2: "tendencia_25",
                                                   3: "tendencia_25", 4: "tendencia_50"})
    proximas = ml.pronosticar(paquete, serie)
    assert proximas["horizonte"].tolist() == [1, 2, 3, 4]
    semanas = list(zip(proximas["anio_epi"], proximas["semana_epi"], strict=True))
    assert semanas[:2] == [(2026, 1), (2026, 2)]
    assert proximas.loc[0, "pronostico"] == pytest.approx(serie["casos_modelo"].iloc[-1])
    assert (proximas["limite_inferior"] <= proximas["pronostico"]).all()
    assert proximas["zona_pronostico"].isin(["Éxito", "Seguridad", "Alerta", "Brote"]).all()


def test_agregar_semanas_updates_the_forecast_origin(serie):
    paquete = ml.entrenar_pronostico_final(serie, {1: "ingenuo", 2: "tendencia_25"})
    nueva = ml.agregar_semanas(serie, [200, 400])
    assert len(nueva) == len(serie) + 2
    assert nueva.iloc[-1][["anio_epi", "semana_epi"]].tolist() == [2026, 2]
    proximas = ml.pronosticar(paquete, nueva)
    assert proximas.loc[0, "semana_epi"] == 3
    assert proximas.loc[0, "pronostico"] == pytest.approx(400)       # ingenuo: la última
    assert ml.agregar_semanas(serie, []) is serie
    with pytest.raises(ValueError):
        ml.pronosticar(paquete, serie.head(3))


# ---------------------------------------------------------------------------
# Riesgo de signos de alarma
# ---------------------------------------------------------------------------

def test_preparar_variables_riesgo_types_and_missing_values():
    casos = _casos_sinteticos(50)
    casos.loc[1, "dias_sintomas_consulta"] = 45                    # fuera de rango
    casos.loc[2, "comuna"] = pd.NA
    X = ml.preparar_variables_riesgo(casos)
    assert list(X.columns) == ml.RIESGO_VARIABLES
    assert np.isnan(X.loc[1, "dias_sintomas_consulta"])
    assert X.loc[2, "comuna"] == ml.SIN_DATO
    assert set(X["fiebre"].dropna().unique()) <= {0.0, 1.0}
    assert X[ml.RIESGO_CATEGORICAS].map(type).eq(str).all().all()


def test_risk_model_excludes_alarm_signs():
    assert not set(ml.RIESGO_VARIABLES) & set(ml.RIESGO_EXCLUIDAS)
    assert "caida_plaquetas" in ml.RIESGO_EXCLUIDAS and "hospitalizado" in ml.RIESGO_EXCLUIDAS


def test_split_temporal_keeps_the_most_recent_cases_for_test():
    casos = _casos_sinteticos(200)
    X, y = ml.preparar_variables_riesgo(casos), casos["severo"].fillna(False).astype(int)
    fechas = casos["fecha_notificacion"]
    X_tr, X_te, y_tr, y_te = ml.split_temporal(X, y, fechas, test_size=0.2)
    assert fechas[X_tr.index].max() <= fechas[X_te.index].min()
    assert len(X_te) == pytest.approx(40, abs=5)


def test_split_temporal_by_year_with_fallback():
    casos = _casos_sinteticos(200)                       # notificados en 2022 y 2023
    X, y = ml.preparar_variables_riesgo(casos), casos["severo"].fillna(False).astype(int)
    fechas = casos["fecha_notificacion"]
    X_tr, X_te, _, _ = ml.split_temporal(X, y, fechas, anio_prueba=2023)
    assert fechas[X_tr.index].max() < pd.Timestamp("2023-01-01") <= fechas[X_te.index].min()
    # Sin casos de ese año se vuelve al 20 % más reciente
    _, X_te, _, _ = ml.split_temporal(X, y, fechas, test_size=0.2, anio_prueba=2030)
    assert len(X_te) == pytest.approx(40, abs=5)


def test_hospitalizacion_uses_every_case_and_only_shared_variables(tmp_path):
    casos = _casos_sinteticos(400)
    casos.loc[:99, "severo"] = pd.NA                     # sin clasificación, como en el INS
    path = tmp_path / "casos.parquet"
    casos.to_parquet(path, index=False)
    X_s, _, _ = ml.load_dataset_riesgo(path, ml.TARGET_RIESGO)
    X_h, y_h, _ = ml.load_dataset_riesgo(path, ml.TARGET_HOSPITALIZACION)
    assert len(X_s) == 300 and len(X_h) == 400
    tarea = ml.TAREAS_CLASIFICACION[ml.TARGET_HOSPITALIZACION]
    assert not {"comuna", "fiebre", "desplazamiento_15d"} & set(tarea.variables)
    params = {**ml.HOSPITALIZACION_PARAMS, "model__n_estimators": 30}
    pipeline, umbrales = ml.entrenar_riesgo(X_h, y_h, params=params,
                                            objetivo=ml.TARGET_HOSPITALIZACION)
    # Los síntomas y la comuna faltan en el INS: no cambian la probabilidad
    otro = X_h.head(5).assign(fiebre=np.nan, cefalea=1.0, comuna=ml.SIN_DATO)
    np.testing.assert_allclose(
        ml.predecir_riesgo(pipeline, X_h.head(5), umbrales)["probabilidad"].to_numpy(),
        ml.predecir_riesgo(pipeline, otro, umbrales)["probabilidad"].to_numpy())


def test_contribuciones_point_to_the_features_that_raise_the_risk():
    casos = _casos_sinteticos()
    casos = casos[casos["severo"].notna()]
    X, y = ml.preparar_variables_riesgo(casos), casos["severo"].astype(int)
    pipeline, _ = ml.entrenar_riesgo(X, y, params={**ml.RIESGO_PARAMS,
                                                   "model__n_estimators": 50})
    fondo = ml.muestra_de_fondo(X, n=100)
    assert len(fondo) == 100 and list(fondo.columns) == ml.RIESGO_VARIABLES
    mayor = X.head(1).assign(edad=88.0, dias_sintomas_consulta=7.0)
    tabla = ml.contribuciones(pipeline, mayor, fondo, ml.RIESGO_VARIABLES)
    puntos = tabla.set_index("factor")["puntos"]
    assert puntos["Edad"] > 0 and puntos["Días desde el inicio de los síntomas"] > 0
    assert tabla["puntos"].abs().is_monotonic_decreasing      # de mayor a menor peso
    # Solo se explican las variables que usa el modelo
    variables = ml.TAREAS_CLASIFICACION[ml.TARGET_HOSPITALIZACION].variables
    factores = set(ml.contribuciones(pipeline, mayor, fondo, variables)["factor"])
    assert "Comuna" not in factores and "Síntomas generales" not in factores


def test_entrenar_y_predecir_riesgo(tmp_path):
    casos = _casos_sinteticos()
    casos = casos[casos["severo"].notna()]
    X, y = ml.preparar_variables_riesgo(casos), casos["severo"].astype(int)
    X_tr, X_te, y_tr, y_te = ml.split_temporal(X, y, casos["fecha_notificacion"])
    params = {"model__n_estimators": 100, "model__min_samples_leaf": 20,
              "model__max_features": 0.3, "model__max_depth": None}
    pipeline, umbrales = ml.entrenar_riesgo(X_tr, y_tr, params=params)
    assert umbrales["corte_medio"] <= umbrales["corte_alto"]
    metricas = ml.evaluar_riesgo(pipeline, X_te, y_te, umbrales)
    assert metricas["roc_auc"] > 0.65
    assert metricas["roc_auc_ic95"][0] <= metricas["roc_auc"] <= metricas["roc_auc_ic95"][1]
    # El nivel alto concentra más casos severos que el bajo
    por_nivel = metricas["pct_positivos_por_nivel"]
    assert por_nivel["Alto"] > por_nivel["Bajo"]
    subgrupos = pd.DataFrame(metricas["por_subgrupo"])
    assert set(subgrupos["variable"]) >= {"sexo", "grupo_edad"}
    assert (subgrupos["n"] >= 50).all() and subgrupos["roc_auc"].between(0, 1).all()
    # Un caso nuevo con una comuna nunca vista también recibe predicción
    nuevo = X_te.head(1).assign(comuna="99")
    pred = ml.predecir_riesgo(pipeline, nuevo, umbrales)
    assert 0 <= pred["probabilidad"].iloc[0] <= 1
    assert pred["nivel_riesgo"].iloc[0] in ml.NIVELES_RIESGO


def test_comparar_riesgo_ranks_against_dummy():
    casos = _casos_sinteticos(600)
    casos = casos[casos["severo"].notna()]
    X, y = ml.preparar_variables_riesgo(casos), casos["severo"].astype(int)
    tabla = ml.comparar_riesgo(X, y).set_index("modelo")
    assert tabla.loc["dummy", "roc_auc"] == pytest.approx(0.5)
    assert tabla.loc["logreg", "roc_auc"] > 0.6


def test_bootstrap_ic_is_ordered():
    y = np.array([0, 1] * 50)
    proba = np.where(y == 1, 0.7, 0.3) + np.random.default_rng(0).normal(0, 0.2, 100)
    bajo, alto = ml.bootstrap_ic(y, proba, n_boot=200)
    assert 0 <= bajo < alto <= 1


# ---------------------------------------------------------------------------
# Perfiles de comunas
# ---------------------------------------------------------------------------

def test_ajustar_perfiles_recovers_three_groups():
    comunas = _comunas_sinteticas().set_index("comuna")
    seleccion = ml.comparar_k(comunas, ks=range(2, 6))
    assert seleccion.loc[seleccion["silueta"].idxmax(), "k"] == 3
    pipeline, perfiles = ml.ajustar_perfiles(comunas, k=3)
    grupos = perfiles.groupby("grupo")["comuna"].apply(set)
    assert grupos["A"] == set(range(1, 7))                       # A = la mayor incidencia
    assert perfiles.attrs["silueta"] > 0.5
    assert perfiles["perfil"].str.len().gt(0).all()
    assert "incidencia alta" in pipeline.descripciones_["A"].lower()


def test_load_dataset_comunas_keeps_only_urban(tmp_path):
    tabla = _comunas_sinteticas()
    rural = tabla.iloc[[0]].assign(comuna=51, zona="Rural", territorio="Corregimiento")
    path = tmp_path / "indicadores.parquet"
    pd.concat([tabla, rural]).to_parquet(path, index=False)
    assert 51 not in ml.load_dataset_comunas(path).index


# ---------------------------------------------------------------------------
# Persistencia y orquestación
# ---------------------------------------------------------------------------

def test_save_and_load_model_with_metadata(tmp_path):
    path = ml.save_model({"a": 1}, {"tarea": "prueba"}, "modelo_prueba", tmp_path)
    objeto, meta = ml.load_model(path)
    assert objeto == {"a": 1}
    assert meta["tarea"] == "prueba" and len(meta["model_sha256"]) == 64


def test_run_trains_and_saves_every_model(serie, tmp_path, monkeypatch):
    silver, gold, models = tmp_path / "silver", tmp_path / "gold", tmp_path / "models"
    silver.mkdir(), gold.mkdir()
    serie.to_parquet(silver / "serie_semanal.parquet", index=False)
    _casos_sinteticos(800).to_parquet(silver / "casos_dengue.parquet", index=False)
    _comunas_sinteticas().to_parquet(gold / "indicadores_comunas.parquet", index=False)
    monkeypatch.setattr(ml, "RIESGO_PARAMS", {**ml.RIESGO_PARAMS, "model__n_estimators": 50})
    monkeypatch.setattr(ml, "HOSPITALIZACION_PARAMS",
                        {**ml.HOSPITALIZACION_PARAMS, "model__n_estimators": 50})
    ml.run(in_dir=silver, gold_dir=gold, models_dir=models)
    for nombre in (ml.MODELO_PRONOSTICO, ml.MODELO_RIESGO, ml.MODELO_HOSPITALIZACION,
                   ml.MODELO_COMUNAS):
        assert (models / f"{nombre}.joblib").exists()
        meta = json.loads((models / f"{nombre}.metadata.json").read_text(encoding="utf-8"))
        assert len(meta["data_sha256"]) == 64
    assert len(pd.read_parquet(gold / "pronostico_semanal.parquet")) == 4
    assert (gold / "perfiles_comunas.parquet").exists()
    riesgo, _ = ml.load_model(models / f"{ml.MODELO_RIESGO}.joblib")
    assert {"pipeline", "umbrales", "fondo", "prevalencia", "variables"} <= set(riesgo)
    _, meta_h = ml.load_model(models / f"{ml.MODELO_HOSPITALIZACION}.joblib")
    assert meta_h["variables"]["categoricas"] == ml.HOSPITALIZACION_CATEGORICAS


def test_functions_do_not_mutate_their_input(serie):
    antes = serie.copy(deep=True)
    ml.dataset_pronostico(serie, horizonte=2)
    ml.agregar_semanas(serie, [10, 20])
    pd.testing.assert_frame_equal(serie, antes)
    casos = _casos_sinteticos(30)
    casos_antes = casos.copy(deep=True)
    ml.preparar_variables_riesgo(casos)
    pd.testing.assert_frame_equal(casos, casos_antes)


def test_saved_risk_model_gives_the_same_predictions(tmp_path):
    casos = _casos_sinteticos(300)
    casos = casos[casos["severo"].notna()]
    X, y = ml.preparar_variables_riesgo(casos), casos["severo"].astype(int)
    params = {**ml.RIESGO_PARAMS, "model__n_estimators": 30}
    pipeline, umbrales = ml.entrenar_riesgo(X, y, params=params)
    path = ml.save_model({"pipeline": pipeline, "umbrales": umbrales}, {}, "riesgo", tmp_path)
    cargado, _ = ml.load_model(path)
    antes = ml.predecir_riesgo(pipeline, X, umbrales)
    despues = ml.predecir_riesgo(cargado["pipeline"], X, cargado["umbrales"])
    pd.testing.assert_frame_equal(antes, despues)


def test_split_temporal_is_reproducible():
    casos = _casos_sinteticos(100)
    X, y = ml.preparar_variables_riesgo(casos), casos["severo"].fillna(False).astype(int)
    a = ml.split_temporal(X, y, casos["fecha_notificacion"])
    b = ml.split_temporal(X, y, casos["fecha_notificacion"])
    assert a[1].index.equals(b[1].index)


def test_temperature_features_do_not_drop_weeks_and_use_only_the_past(serie):
    con_clima = serie.assign(temp_anomalia=np.where(np.arange(len(serie)) % 3 == 0, np.nan, 0.5))
    X, _, _ = ml.dataset_pronostico(con_clima, horizonte=2, n_rezagos=4)
    X_sin, _, _ = ml.dataset_pronostico(serie, horizonte=2, n_rezagos=4)
    assert len(X) == len(X_sin)                               # el clima faltante no quita filas
    assert X["temp_4s"].notna().mean() > 0.9
    cambio = con_clima.copy()
    cambio.loc[cambio.index[-1], "temp_anomalia"] = 4.0
    X2, _, _ = ml.dataset_pronostico(cambio, horizonte=2, n_rezagos=4)
    pd.testing.assert_series_equal(X["temp_4s"].iloc[:-1], X2["temp_4s"].iloc[:-1])


def test_tendencia_clima_learns_the_temperature_effect():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"cambio_4": rng.normal(0, 0.2, n), "temp_4s": 24 + rng.normal(0, 1, n)})
    y = 0.25 * 2 * X["cambio_4"] / 4 + 0.03 * 2 * (X["temp_4s"] - 24)
    modelo = ml.TendenciaClima(horizonte=2, amortiguamiento=0.25).fit(X, y)
    assert modelo.coeficiente_ == pytest.approx(0.03, abs=0.005)
    sin_temp = X.assign(temp_4s=np.nan)
    np.testing.assert_allclose(modelo.predict(sin_temp),
                               ml.TendenciaAmortiguada(2, 0.25).predict(sin_temp))
    pocas = ml.TendenciaClima(2, 0.25).fit(X.assign(temp_4s=np.nan), y)
    assert pocas.coeficiente_ == 0 and pocas.n_semanas_clima_ == 0


def test_ml_candidates_ignore_the_climate_columns(serie):
    con_clima = serie.assign(temp_anomalia=np.nan)
    X, y, meta = ml.dataset_pronostico(con_clima, horizonte=1)
    modelo = ml.get_candidatos_pronostico(1)["ridge"].fit(X[meta["usable"]], y[meta["usable"]])
    assert np.isfinite(modelo.predict(X)).all()                # los nulos de clima no estorban
