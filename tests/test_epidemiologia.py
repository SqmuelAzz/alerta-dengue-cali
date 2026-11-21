import numpy as np
import pandas as pd
import pytest

from flows.common import epidemiologia as epi


@pytest.mark.parametrize("fecha, esperado", [
    ("2019-12-29", (2020, 1)),     # el domingo 29-dic-2019 ya es semana 1 de 2020
    ("2020-03-03", (2020, 10)),
    ("2021-01-02", (2020, 53)),    # 2020 tuvo 53 semanas
    ("2021-01-03", (2021, 1)),
    ("2023-01-01", (2023, 1)),
    ("2026-01-01", (2025, 53)),    # 2025 también tiene 53 semanas
    ("2026-01-04", (2026, 1)),
])
def test_semana_epidemiologica_matches_the_sivigila_calendar(fecha, esperado):
    out = epi.semana_epidemiologica(pd.Series(pd.to_datetime([fecha])))
    assert (int(out["anio_epi"].iloc[0]), int(out["semana_epi"].iloc[0])) == esperado


def test_semana_epidemiologica_keeps_missing_dates_missing():
    out = epi.semana_epidemiologica(pd.Series(pd.to_datetime(["2023-01-01", None])))
    assert out["anio_epi"].isna().tolist() == [False, True]


def test_semanas_del_anio():
    assert [epi.semanas_del_anio(a) for a in range(2020, 2026)] == [53, 52, 52, 52, 52, 53]


def test_calendario_is_consecutive_and_complete():
    cal = epi.calendario_epidemiologico(2020, 2025)
    assert len(cal) == 53 + 52 * 4 + 53
    assert (cal["fecha_inicio"].diff().dropna() == pd.Timedelta(weeks=1)).all()
    assert (cal["fecha_fin"] - cal["fecha_inicio"] == pd.Timedelta(days=6)).all()
    assert cal["fecha_inicio"].dt.day_of_week.eq(6).all()          # todas empiezan en domingo


def test_siguientes_semanas_crosses_53_week_years():
    out = epi.siguientes_semanas(2025, 52, 3)
    semanas = list(zip(out["anio_epi"], out["semana_epi"], strict=True))
    assert semanas == [(2025, 53), (2026, 1), (2026, 2)]
    assert out["fecha_inicio"].iloc[0] == pd.Timestamp("2025-12-28")
    corto = epi.siguientes_semanas(2024, 52, 1)
    assert (corto["anio_epi"].item(), corto["semana_epi"].item()) == (2025, 1)


def _serie(valores_por_anio: dict) -> pd.DataFrame:
    filas = [(anio, semana, valor) for anio, valores in valores_por_anio.items()
             for semana, valor in enumerate(valores, start=1)]
    return pd.DataFrame(filas, columns=["anio_epi", "semana_epi", "casos"])


def test_canal_endemico_uses_only_base_years_and_quartiles():
    serie = _serie({2021: [10] * 52, 2022: [20] * 52, 2023: [1000] * 52, 2025: [30] * 52})
    canal = epi.canal_endemico(serie, anios_base=[2021, 2022, 2025], ventana=0)
    fila = canal[canal["semana_epi"] == 10].iloc[0]
    assert (fila["q1"], fila["mediana"], fila["q3"]) == (15.0, 20.0, 25.0)   # 2023 no cuenta
    assert fila["n_valores"] == 3
    assert len(canal) == 53                       # la semana 53 usa los valores de la 52


def test_canal_endemico_window_pools_neighbour_weeks_across_new_year():
    serie = _serie({2021: [5] + [100] * 50 + [7], 2022: [6] + [100] * 50 + [8]})
    canal = epi.canal_endemico(serie, anios_base=[2021, 2022], ventana=1)
    semana_1 = canal[canal["semana_epi"] == 1].iloc[0]
    assert semana_1["n_valores"] == 6            # semanas 52, 1 y 2 de cada año
    sin_nulos = epi.canal_endemico(serie.assign(casos=np.nan), [2021], ventana=0)
    assert sin_nulos["q3"].isna().all()          # sin datos no hay canal


def test_zona_canal_classifies_each_band():
    valor = pd.Series([1, 15, 25, 40, None])
    q1, mediana, q3 = (pd.Series([10.0] * 5), pd.Series([20.0] * 5), pd.Series([30.0] * 5))
    zonas = epi.zona_canal(valor, q1, mediana, q3)
    assert zonas.iloc[:4].tolist() == ["Éxito", "Seguridad", "Alerta", "Brote"]
    assert pd.isna(zonas.iloc[4])                # sin valor no hay zona


def test_tasa_por_100k():
    assert epi.tasa_por_100k(35_604, 2_597_663) == pytest.approx(1370.6, abs=0.1)
