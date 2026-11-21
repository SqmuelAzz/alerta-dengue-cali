"""Pruebas de la API (cliente de pruebas de Flask) y del tablero (Streamlit simulado)."""
import importlib
import runpy
import sys
import types
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from tests.test_05_ml_models import _casos_sinteticos, _comunas_sinteticas, _serie_sintetica

matplotlib.use("Agg")
ml = importlib.import_module("flows.05_ml_models")
agg = importlib.import_module("flows.03_aggregation")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recursos(tmp_path_factory):
    """Pipeline de modelos sobre datos sintéticos y recursos cargados como en producción."""
    from app import servicio

    base = tmp_path_factory.mktemp("app")
    silver, gold, models = base / "silver", base / "gold", base / "models"
    silver.mkdir(), gold.mkdir()
    serie = _serie_sintetica()
    serie["fecha_fin"] = serie["fecha_inicio"] + pd.Timedelta(days=6)
    serie["poblacion"], serie["tasa_100k"] = 2_500_000, serie["casos"] / 25
    # Clima de la ciudad con huecos, como el que publica el IDEAM
    rng = np.random.default_rng(1)
    con_clima = (serie["anio_epi"] != 2021).to_numpy()
    serie["lluvia_mm"] = np.where(con_clima, rng.gamma(2, 6, len(serie)).round(1), np.nan)
    serie["temp_media"] = np.where(con_clima, 24 + rng.normal(0, 1, len(serie)), np.nan)
    serie["temp_anomalia"] = serie["temp_media"] - 24
    serie["estaciones_lluvia"] = np.where(con_clima, 3, 0)
    serie["estaciones_temp"] = serie["estaciones_lluvia"]
    serie.to_parquet(silver / "serie_semanal.parquet", index=False)
    pd.DataFrame({"codigo_estacion": ["0026085160", "0026085160", "0026055120", "0026055120"],
                  "estacion": ["SILOE - AUT", "SILOE", "UNIVERSIDAD DEL VALLE - AUT",
                               "UNIVERSIDAD DEL VALLE"],
                  "sistema": ["anterior", "actual", "anterior", "actual"],
                  "anio_epi": [2024, 2024, 2024, 2024], "semana_epi": [22, 23, 35, 36],
                  "n_obs_temp": [168, 168, 168, 168]}
                 ).to_parquet(silver / "clima_semanal.parquet", index=False)
    agg.clima_rezagos(serie).to_parquet(gold / "clima_rezagos.parquet", index=False)
    _casos_sinteticos(600).to_parquet(silver / "casos_dengue.parquet", index=False)
    comunas = _comunas_sinteticas().assign(periodo="2022-2023", casos=100, pct_menores_15=30.0)
    comunas.to_parquet(gold / "indicadores_comunas.parquet", index=False)
    canal = agg.canal_endemico_ciudad(serie, anios_base=[2021, 2022, 2025], ventana=2)
    canal.to_parquet(gold / "canal_endemico.parquet", index=False)
    agg.situacion_semanal(serie, canal).to_parquet(gold / "situacion_semanal.parquet",
                                                   index=False)
    originales = dict(ml.RIESGO_PARAMS), dict(ml.HOSPITALIZACION_PARAMS)
    ml.RIESGO_PARAMS["model__n_estimators"] = 50
    ml.HOSPITALIZACION_PARAMS["model__n_estimators"] = 50
    try:
        ml.run(in_dir=silver, gold_dir=gold, models_dir=models)
    finally:
        ml.RIESGO_PARAMS.update(originales[0])
        ml.HOSPITALIZACION_PARAMS.update(originales[1])
    return servicio.cargar_recursos(models, silver, gold)


@pytest.fixture(scope="module")
def client(recursos):
    from app.api import create_app
    return create_app(recursos).test_client()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_salud(client):
    res = client.get("/salud")
    assert res.status_code == 200
    cuerpo = res.get_json()
    assert cuerpo["estado"] == "ok"
    assert set(cuerpo["modelos"]) == {"pronostico", "riesgo", "hospitalizacion", "comunas"}
    assert cuerpo["ultima_semana_datos"] == "2025-S53"


def test_situacion(client):
    cuerpo = client.get("/situacion").get_json()
    assert (cuerpo["anio_epi"], cuerpo["semana_epi"]) == (2025, 53)
    assert cuerpo["zona"] in ("Éxito", "Seguridad", "Alerta", "Brote")


def test_pronostico_get_and_post(client):
    semanas = client.get("/pronostico").get_json()["semanas"]
    assert [s["horizonte"] for s in semanas] == [1, 2, 3, 4]
    assert semanas[0]["anio_epi"] == 2026 and semanas[0]["semana_epi"] == 1
    assert semanas[0]["fecha_inicio"] == "2026-01-04"
    actualizado = client.post("/pronostico", json={"casos_recientes": [500, 900]}).get_json()
    primera = actualizado["semanas"][0]
    assert primera["semana_epi"] == 3 and primera["pronostico"] == 900     # ingenuo a 1 semana
    assert all(isinstance(s["alerta"], bool) for s in actualizado["semanas"])


@pytest.mark.parametrize("cuerpo", [
    {"casos_recientes": "100"}, {"casos_recientes": [-1]}, {"casos_recientes": [1.5]},
    {"casos_recientes": [True]}, {"casos_recientes": list(range(20))}, [1, 2],
])
def test_pronostico_rejects_invalid_input(client, cuerpo):
    res = client.post("/pronostico", json=cuerpo)
    assert res.status_code == 400 and "error" in res.get_json()


def test_riesgo_returns_probability_and_level(client):
    caso = {"edad": 70, "sexo": "F", "dias_sintomas_consulta": 6, "estrato": 1, "comuna": 6,
            "afiliacion": "Subsidiado", "fiebre": True, "cefalea": True}
    cuerpo = client.post("/riesgo", json=caso).get_json()
    assert 0 <= cuerpo["probabilidad"] <= 1
    assert cuerpo["nivel_riesgo"] in ("Bajo", "Medio", "Alto")
    assert 0 <= cuerpo["hospitalizacion"]["probabilidad"] <= 1      # segundo clasificador
    assert 0 < cuerpo["promedio"] < 1
    assert cuerpo["veces_promedio"] == pytest.approx(cuerpo["probabilidad"]
                                                     / cuerpo["promedio"], abs=0.01)
    factores = {f["factor"]: f for f in cuerpo["factores"]}
    assert factores["Edad"]["valor"] == "70 años"
    assert factores["Síntomas generales"]["valor"] == "2 de 6"
    assert "Comuna" not in {f["factor"] for f in cuerpo["hospitalizacion"]["factores"]}
    minimo = client.post("/riesgo", json={"edad": 30, "sexo": "M",
                                          "dias_sintomas_consulta": 1})
    assert minimo.status_code == 200                       # los opcionales quedan "Sin dato"


@pytest.mark.parametrize("caso, mensaje", [
    ({"sexo": "F", "dias_sintomas_consulta": 2}, "edad"),
    ({"edad": 200, "sexo": "F", "dias_sintomas_consulta": 2}, "edad"),
    ({"edad": 20, "sexo": "X", "dias_sintomas_consulta": 2}, "sexo"),
    ({"edad": 20, "sexo": "F", "dias_sintomas_consulta": 40}, "dias"),
    ({"edad": 20, "sexo": "F", "dias_sintomas_consulta": 2, "estrato": 9}, "estrato"),
    ({"edad": 20, "sexo": "F", "dias_sintomas_consulta": 2, "comuna": 30}, "comuna"),
    ({"edad": 20, "sexo": "F", "dias_sintomas_consulta": 2, "afiliacion": "Otro"},
     "afiliacion"),
    ({"edad": 20, "sexo": "F", "dias_sintomas_consulta": 2, "fiebre": "si"}, "fiebre"),
])
def test_riesgo_validates_input(client, caso, mensaje):
    res = client.post("/riesgo", json=caso)
    assert res.status_code == 400 and mensaje in res.get_json()["error"]


def test_riesgo_without_json_body(client):
    res = client.post("/riesgo", data="texto", content_type="text/plain")
    assert res.status_code == 400


def test_comunas(client):
    tabla = client.get("/comunas").get_json()
    assert len(tabla) == 18 and {"grupo", "perfil", "tasa_100k"} <= set(tabla[0])
    una = client.get("/comunas/1").get_json()
    assert una["comuna"] == 1 and una["grupo"] == "A"
    assert client.get("/comunas/99").status_code == 404
    assert client.get("/no-existe").status_code == 404


def test_cargar_recursos_explains_missing_files(tmp_path):
    from app import servicio
    with pytest.raises(FileNotFoundError, match="make pipeline"):
        servicio.cargar_recursos(tmp_path, tmp_path, tmp_path)


# ---------------------------------------------------------------------------
# Tablero (Streamlit simulado: se ejecuta cada sección sin servidor)
# ---------------------------------------------------------------------------

class _Elemento:
    """Imita los objetos de Streamlit (columnas, pestañas, formularios)."""

    def __init__(self, registro):
        self._registro = registro

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, nombre):
        return getattr(_StreamlitFalso(self._registro), nombre)


class _StreamlitFalso(types.ModuleType):
    def __init__(self, registro):
        super().__init__("streamlit")
        self._registro = registro

    def __getattr__(self, nombre):
        def llamada(*args, **kwargs):
            self._registro.append((nombre, args, kwargs))
        return llamada

    def cache_resource(self, funcion):
        return funcion

    def columns(self, n, **kwargs):
        return [_Elemento(self._registro) for _ in range(n if isinstance(n, int) else len(n))]

    @property
    def sidebar(self):
        return _Elemento(self._registro)

    def expander(self, *args, **kwargs):
        return _Elemento(self._registro)

    def radio(self, label, options, index=0, **kwargs):
        return list(options)[index]

    def download_button(self, label, data, **kwargs):
        self._registro.append(("download_button", (label,), kwargs))
        assert len(data) > 0
        return False

    def tabs(self, nombres):
        return [_Elemento(self._registro) for _ in nombres]

    def form(self, *args, **kwargs):
        return _Elemento(self._registro)

    def selectbox(self, label, options, index=0, format_func=str):
        opciones = list(options)
        for opcion in opciones:
            format_func(opcion)
        return opciones[index]

    def number_input(self, label, min_value, max_value, value):
        return value

    def text_input(self, *args, **kwargs):
        return "40, 60"

    def checkbox(self, label, value=False):
        return value

    def form_submit_button(self, *args, **kwargs):
        return True

    def stop(self):
        raise RuntimeError("st.stop")


def test_dashboard_renders_every_section(recursos, monkeypatch):
    from app import servicio
    registro = []
    monkeypatch.setitem(sys.modules, "streamlit", _StreamlitFalso(registro))
    monkeypatch.setattr(servicio, "cargar_recursos", lambda: recursos)
    runpy.run_path(str(ROOT / "app" / "dashboard.py"), run_name="__main__")
    nombres = [n for n, _, _ in registro]
    assert "error" not in nombres                          # ninguna sección falló
    assert nombres.count("pyplot") == 5          # canal, clima, rezagos, pronóstico, comunas
    assert nombres.count("download_button") == 4  # situación, clima, pronóstico y comunas
    html = " ".join(str(a[0]) for n, a, _ in registro if n == "markdown" and a)
    assert "Signos de alarma o dengue grave" in html and "Hospitalización" in html
    assert "Grupo A" in html and "Umbral de brote" in html
    assert "Qué pesa en el riesgo de signos de alarma" in html
    assert "Promedio de los casos" in html
    assert "Clima y casos en" in html and "No con estos datos" in html
    matplotlib.pyplot.close("all")


def test_dashboard_shows_a_clear_message_without_models(monkeypatch):
    from app import servicio
    registro = []
    monkeypatch.setitem(sys.modules, "streamlit", _StreamlitFalso(registro))

    def sin_modelos():
        raise FileNotFoundError("Faltan archivos del pipeline; ejecute 'make pipeline'")

    monkeypatch.setattr(servicio, "cargar_recursos", sin_modelos)
    with pytest.raises(RuntimeError, match="st.stop"):
        runpy.run_path(str(ROOT / "app" / "dashboard.py"), run_name="__main__")
    assert any(n == "error" and "make pipeline" in a[0] for n, a, _ in registro)


# ---------------------------------------------------------------------------
# Servicios usados por el tablero
# ---------------------------------------------------------------------------

def test_historial_y_pronostico_marks_added_weeks(recursos):
    from app import servicio
    historial, tabla = servicio.historial_y_pronostico(recursos, [500, 900], semanas=10)
    assert len(historial) == 10 and historial["agregada"].sum() == 2
    assert historial["q3"].notna().all()
    assert historial.iloc[-1][["anio_epi", "semana_epi"]].tolist() == [2026, 2]
    assert tabla["semana_epi"].iloc[0] == 3
    sin_nuevas, _ = servicio.historial_y_pronostico(recursos)
    assert not sin_nuevas["agregada"].any()


def test_clima_anio_marks_the_sensor_change(recursos):
    from app import servicio
    assert [c["semana_epi"] for c in recursos.cambio_sistema] == [23, 36]
    assert recursos.cambio_sistema[0]["estacion"] == "Siloé"       # nombre de la estación
    tabla, resumen = servicio.clima_anio(recursos, 2024)
    assert len(tabla) == resumen["semanas"] == 52
    assert resumen["semanas_con_lluvia"] == 52 and resumen["semanas_sin_clima"] == 0
    assert len(resumen["cambio_sistema"]) == 2
    _, sin_clima = servicio.clima_anio(recursos, 2021)
    assert sin_clima["semanas_con_temperatura"] == 0 and sin_clima["temperatura_media_c"] is None
    assert sin_clima["cambio_sistema"] == []
    relacion = servicio.relacion_clima(recursos)
    assert set(relacion) == {"lluvia_mm", "temp_media", "temp_anomalia"}
    assert all(0 <= r["rezago_semanas"] <= 12 for r in relacion.values())


def test_clima_api(client):
    respuesta = client.get("/clima?anio=2024")
    assert respuesta.status_code == 200
    cuerpo = respuesta.get_json()
    assert cuerpo["anio_epi"] == 2024 and len(cuerpo["detalle_semanal"]) == 52
    assert "lluvia_mm" in cuerpo["relacion_con_casos"]
    assert client.get("/clima").get_json()["anio_epi"] == 2025       # último año por defecto
    assert client.get("/clima?anio=1999").status_code == 404


def test_resumen_anio(recursos):
    from app import servicio
    resumen = servicio.resumen_anio(recursos, 2024)
    datos = recursos.situacion[recursos.situacion["anio_epi"] == 2024]
    assert resumen["casos"] == int(datos["casos"].sum())
    assert resumen["casos_pico"] == int(datos["casos"].max())
    assert 0 <= resumen["semanas_brote"] <= resumen["semanas_con_datos"]


def test_formatos_del_tablero(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", _StreamlitFalso([]))
    tablero = runpy.run_path(str(ROOT / "app" / "dashboard.py"), run_name="tablero")
    assert tablero["entero"](35604) == "35.604"
    assert tablero["decimal"](1370.62) == "1.370,6"
    assert tablero["fecha_corta"]("2026-01-04") == "4 ene 2026"
    html = tablero["tarjeta"]("Casos", "30", "Semana 53", "#000")
    assert "Casos" in html and "border-top" in html
