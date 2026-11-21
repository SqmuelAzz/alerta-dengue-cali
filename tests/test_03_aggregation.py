import importlib
import json

import numpy as np
import pandas as pd
import pytest

agg = importlib.import_module("flows.03_aggregation")
de = importlib.import_module("flows.01_data_engineering")


def _serie(valores: dict[int, list], cobertura: str = "completa") -> pd.DataFrame:
    filas = []
    for anio, casos in valores.items():
        for semana, c in enumerate(casos, start=1):
            filas.append((anio, semana, c))
    serie = pd.DataFrame(filas, columns=["anio_epi", "semana_epi", "casos"])
    serie["casos_modelo"] = serie["casos"].astype(float)
    serie["cobertura"] = cobertura
    serie["fecha_inicio"] = pd.Timestamp("2021-01-03") + pd.to_timedelta(
        np.arange(len(serie)) * 7, unit="D")
    serie["fecha_fin"] = serie["fecha_inicio"] + pd.Timedelta(days=6)
    serie["poblacion"] = 2_500_000
    serie["tasa_100k"] = serie["casos"] / serie["poblacion"] * 100_000
    for col in ("hospitalizados", "confirmados", "menores_15"):
        serie[col] = 0
    return serie


@pytest.fixture
def casos_clasificados():
    """Casos del Distrito con clasificación: los severos son mayores y subsidiados."""
    n = 200
    rng = np.random.default_rng(0)
    severo = rng.random(n) < 0.4
    return pd.DataFrame({
        "id_caso": [f"c{i}" for i in range(n)],
        "fuente": "SSPD Cali",
        "anio_epi": 2023,
        "edad": np.where(severo, 60, 20).astype(float),
        "grupo_edad": np.where(severo, "45-64", "15-29"),
        "sexo": rng.choice(["F", "M"], n),
        "afiliacion": np.where(severo, "Subsidiado", "Contributivo"),
        "estrato": pd.array(rng.integers(1, 4, n), dtype="Int64"),
        "zona": "Urbana",
        "comuna": pd.array(rng.choice([6, 14], n), dtype="Int64"),
        "severo": pd.array(severo, dtype="boolean"),
        "hospitalizado": severo,
        "dias_sintomas_consulta": pd.array(np.where(severo, 5, 2), dtype="Int64"),
        "dias_consulta_notificacion": pd.array([1] * n, dtype="Int64"),
    })


def test_canal_endemico_ciudad_ignores_incomplete_weeks():
    serie = _serie({2021: [10] * 52, 2022: [30] * 52})
    serie.loc[serie["anio_epi"] == 2022, "cobertura"] = "parcial"
    canal = agg.canal_endemico_ciudad(serie, anios_base=[2021, 2022], ventana=0)
    assert (canal["q3"].dropna() == 10).all()          # solo cuenta 2021 (completa)
    assert canal["anios_base"].iloc[0] == "2021, 2022"


def test_situacion_semanal_flags_weeks_above_q3():
    serie = _serie({2021: [10] * 52, 2022: [20] * 52, 2024: [5] * 26 + [100] * 26})
    canal = agg.canal_endemico_ciudad(serie, anios_base=[2021, 2022], ventana=0)
    situacion = agg.situacion_semanal(serie, canal)
    s2024 = situacion[situacion["anio_epi"] == 2024]
    assert s2024["supera_q3"].sum() == 26
    assert set(s2024.loc[s2024["supera_q3"], "zona"]) == {"Brote"}
    assert set(s2024.loc[~s2024["supera_q3"], "zona"]) == {"Éxito"}


def test_incidencia_anual_flags_cut_years():
    serie = _serie({2022: [10] * 52, 2023: [20] * 52})
    serie.loc[serie.index[-3:], ["casos", "cobertura"]] = [np.nan, "sin_datos"]
    casos = pd.DataFrame({"anio_epi": [2022, 2023], "edad": [10, 30], "fuente": ["a", "a"]})
    anual = agg.incidencia_anual(serie, casos).set_index("anio_epi")
    assert anual.loc[2022, "casos"] == 520 and bool(anual.loc[2022, "anio_completo"])
    assert anual.loc[2023, "casos"] == 20 * 49 and not bool(anual.loc[2023, "anio_completo"])
    assert anual.loc[2022, "tasa_100k"] == pytest.approx(520 / 2_500_000 * 100_000)


def test_estacionalidad_index_averages_one_per_year():
    serie = _serie({2021: [10] * 26 + [30] * 26, 2022: [100] * 26 + [300] * 26})
    est = agg.estacionalidad(serie)
    assert est["indice_medio"].iloc[0] == pytest.approx(0.5)     # mitad del promedio anual
    assert est["indice_medio"].iloc[-1] == pytest.approx(1.5)
    assert est["indice_medio"].mean() == pytest.approx(1.0)


def test_autocorrelacion_of_a_smooth_series_is_high():
    serie = _serie({2021: list(np.linspace(10, 500, 52))})
    ac = agg.autocorrelacion(serie, max_rezago=3)
    assert ac["rezago_semanas"].tolist() == [1, 2, 3]
    assert (ac["autocorrelacion"] > 0.99).all()


def test_wilson_interval_contains_the_proportion():
    bajo, alto = agg.wilson_interval(pd.Series([40.0]), pd.Series([100]))
    assert bajo.iloc[0] < 0.4 < alto.iloc[0]
    assert bajo.iloc[0] == pytest.approx(0.309, abs=0.001)


def test_severidad_por_grupo_reports_n_and_confidence_interval(casos_clasificados):
    tabla = agg.severidad_por_grupo(casos_clasificados)
    edad = tabla[tabla["variable"] == "grupo_edad"].set_index("categoria")
    assert edad.loc["45-64", "pct_severos"] == 100 and edad.loc["15-29", "pct_severos"] == 0
    assert (tabla["ic95_inferior"] <= tabla["pct_severos"] + 1e-9).all()
    assert (tabla["pct_severos"] <= tabla["ic95_superior"] + 1e-9).all()
    assert tabla["ic95_inferior"].min() >= 0 and tabla["ic95_superior"].max() <= 100
    assert tabla["n"].sum() > 0


def test_chi2_severidad_detects_a_strong_association(casos_clasificados):
    res = agg.chi2_severidad(casos_clasificados, "afiliacion")
    assert set(res) == {"chi2", "p_value", "dof", "n", "cramers_v"}
    # Asociación perfecta; en tablas 2x2 scipy aplica la corrección de Yates (V ≈ 0,99)
    assert res["p_value"] < 0.001 and res["cramers_v"] > 0.95
    sin_efecto = agg.chi2_severidad(casos_clasificados, "sexo")
    assert sin_efecto["cramers_v"] < 0.3


def test_statistical_tests_do_not_crash_with_too_few_cases(casos_clasificados):
    pocos = casos_clasificados.head(10)            # todos los grupos tienen menos de 30
    res = agg.chi2_severidad(pocos, "afiliacion")
    assert res["p_value"] is None and res["n"] == 0
    solo_leves = casos_clasificados.assign(severo=pd.array([False] * 200, dtype="boolean"))
    assert agg.consulta_vs_severidad(solo_leves)["p_value"] is None


def test_consulta_vs_severidad(casos_clasificados):
    res = agg.consulta_vs_severidad(casos_clasificados)
    assert res["mediana_dias_severos"] == 5 and res["mediana_dias_no_severos"] == 2


def test_indicadores_comunas_rates_use_mean_population():
    casos = pd.DataFrame({
        "id_caso": ["a", "b", "c"], "anio_epi": [2022, 2023, 2023],
        "comuna": pd.array([6, 6, 14], dtype="Int64"), "zona": "Urbana",
        "hospitalizado": [True, False, False], "edad": [10.0, 30.0, 50.0],
        "estrato": pd.array([1, 3, 2], dtype="Int64"),
        "severo": pd.array([True, False, pd.NA], dtype="boolean"),
        "afiliacion": ["Subsidiado", "Contributivo", "Subsidiado"],
    })
    poblacion = pd.DataFrame({
        "territorio": ["Comuna 6", "Comuna 6", "Comuna 14", "Comuna 14", "Comuna 6",
                       "Comuna 14"],
        "anio": [2022, 2023, 2022, 2023, 2020, 2020],
        "poblacion": [100_000, 200_000, 50_000, 50_000, 90_000, 45_000],
        "comuna": pd.array([6, 6, 14, 14, 6, 14], dtype="Int64"),
    })
    centinela = pd.DataFrame({"anio": [2020], "comuna": [6], "hosp_arbovirosis": [9]})
    tabla = agg.indicadores_comunas(casos, poblacion, centinela).set_index("comuna")
    assert tabla.loc[6, "tasa_100k"] == pytest.approx(2 / 150_000 * 100_000)
    assert tabla.loc[6, "pct_severos"] == 50 and tabla.loc[6, "con_clasificacion"] == 2
    assert tabla.loc[6, "hosp_arbovirosis_100k"] == pytest.approx(9 / 90_000 * 100_000)
    assert tabla.loc[14, "hosp_arbovirosis_100k"] == 0
    assert all(isinstance(c, str) for c in tabla.reset_index().columns)   # Parquet


def test_run_writes_gold_tables(fake_bronze, tmp_path):
    silver = tmp_path / "silver"
    de.run(in_dir=fake_bronze, out_dir=silver)
    gold = agg.run(in_dir=silver, out_dir=tmp_path / "gold")
    for nombre in ("incidencia_anual", "canal_endemico", "situacion_semanal",
                   "indicadores_comunas", "severidad_por_grupo", "calidad_datos"):
        assert (gold / f"{nombre}.parquet").exists()
    actual = json.loads((gold / "situacion_actual.json").read_text(encoding="utf-8"))
    assert {"anio_epi", "semana_epi", "zona", "casos"} <= set(actual)


def test_clima_rezagos_detects_a_lagged_relationship():
    rng = np.random.default_rng(3)
    n = 120
    temperatura = 24 + rng.normal(0, 1, n)
    # El crecimiento de los casos depende de la temperatura de 3 semanas antes
    crecimiento = np.r_[np.zeros(3), 0.05 * (temperatura[:-3] - 24)]
    casos = 100 * np.exp(np.cumsum(crecimiento))
    serie = pd.DataFrame({"anio_epi": 2024, "semana_epi": range(n), "casos": casos.round(),
                          "lluvia_mm": rng.gamma(2, 5, n), "temp_media": temperatura})
    tabla = agg.clima_rezagos(serie, rezagos=range(0, 7), horizonte=1)
    assert set(tabla["variable"]) == {"lluvia_mm", "temp_media"}
    temp = tabla[tabla["variable"] == "temp_media"].set_index("rezago_semanas")
    assert temp["rho"].idxmax() == 2          # la temperatura de t-2 explica el cambio t->t+1
    assert temp.loc[2, "p_value"] < 0.001 and temp.loc[2, "n"] == n - 3
    fuerte = agg._mas_fuerte(tabla, "temp_media")
    assert fuerte["rezago_semanas"] == 2
