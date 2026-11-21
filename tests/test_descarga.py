"""Pruebas de la descarga sin conexión: la API se reemplaza por respuestas inventadas."""
import csv
import io
import urllib.parse

import pytest

from flows import descarga


def _csv(filas):
    salida = io.StringIO()
    escritor = csv.DictWriter(salida, fieldnames=descarga.IDEAM_COLUMNAS)
    escritor.writeheader()
    escritor.writerows(filas)
    return salida.getvalue().encode("utf-8")


def _parametros(url):
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


def test_url_socrata_filters_station_and_period():
    p = _parametros(descarga.url_socrata("s54a-sgyg", "0026055120", "2024-01-01", "2024-02-01",
                                         offset=50_000))
    assert "codigoestacion='0026055120'" in p["$where"]
    assert "fechaobservacion >= '2024-01-01T00:00:00'" in p["$where"]
    assert "fechaobservacion < '2024-02-01T00:00:00'" in p["$where"]
    assert p["$offset"] == "50000" and p["$order"] == ":id"


def test_meses_covers_the_whole_period():
    periodos = descarga.meses(2020, 2021)
    assert len(periodos) == 24
    assert periodos[0] == ("2020-01-01", "2020-02-01") and periodos[-1][1] == "2022-01-01"


def test_descargar_estacion_follows_the_pages(monkeypatch):
    monkeypatch.setattr(descarga, "PAGINA", 2)
    paginas = {"0": [{"valorobservado": "1"}, {"valorobservado": "2"}],
               "2": [{"valorobservado": "3"}]}

    def falso(url):
        return _csv(paginas[_parametros(url)["$offset"]])

    filas = descarga.descargar_estacion("s54a-sgyg", "1", "2024-01-01", "2024-02-01", falso)
    assert [f["valorobservado"] for f in filas] == ["1", "2", "3"]


def test_descargar_ideam_writes_one_file_per_variable(tmp_path):
    def falso(url):
        p = _parametros(url)
        estacion = p["$where"].split("'")[1]
        if ">= '2024-03-01" not in p["$where"] or p["$offset"] != "0":
            return _csv([])
        return _csv([{"codigoestacion": estacion, "fechaobservacion": "2024-03-02T10:00:00.000",
                      "valorobservado": "0.5"}])

    resumen = descarga.descargar_ideam(tmp_path, 2024, 2024, {"A": "a", "B": "b"}, falso)
    assert resumen["precipitacion"]["lecturas"] == 2
    assert resumen["temperatura"]["por_estacion"] == {"A": 1, "B": 1}
    with (tmp_path / "ideam_precipitacion_cali_2024-2024.csv").open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert list(filas[0]) == descarga.IDEAM_COLUMNAS and len(filas) == 2


def test_descargar_dane_checks_that_it_is_an_excel(tmp_path):
    resumen = descarga.descargar_dane(tmp_path, lambda url: b"PK\x03\x04contenido")
    assert (tmp_path / descarga.NOMBRE_DANE).read_bytes().startswith(b"PK")
    assert resumen["bytes"] == 13
    with pytest.raises(ValueError):
        descarga.descargar_dane(tmp_path, lambda url: b"<html>error</html>")
