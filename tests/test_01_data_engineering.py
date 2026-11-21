import importlib

import numpy as np
import pandas as pd
import pytest

de = importlib.import_module("flows.01_data_engineering")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def test_parse_fecha_reads_the_three_source_formats():
    texto = pd.Series(["19-abr-2022", "01/01/2023", "2/08/2020 12:00:00 a.\xa0m.",
                       "  -   -", "", None])
    fechas = de.parse_fecha(texto)
    assert fechas.iloc[:3].tolist() == [pd.Timestamp("2022-04-19"), pd.Timestamp("2023-01-01"),
                                        pd.Timestamp("2020-08-02")]
    assert fechas.iloc[3:].isna().all()


def test_recode_and_to_bool_normalize_spelling():
    afiliacion = de.recode(pd.Series(["Excepcion", "Excepción", "S", " no  asegurado", "xx"]),
                           de.AFILIACION)
    assert afiliacion.tolist()[:4] == ["Excepción o especial", "Excepción o especial",
                                       "Subsidiado", "No asegurado"]
    assert pd.isna(afiliacion.iloc[4])
    assert de.to_bool(pd.Series(["SI", "No", "1", "2", "-"])).tolist() == [
        True, False, True, False, pd.NA]


def test_to_int_in_rejects_codes_outside_the_valid_list():
    comuna = de.to_int_in(pd.Series(["6", "SD", "100", "51", " 22 "]), de.COMUNAS_VALIDAS)
    assert comuna.tolist() == [6, pd.NA, pd.NA, 51, 22]


def test_drop_empty_rows():
    df = pd.DataFrame({"a": ["1", "", " "], "b": ["", "", None]})
    assert len(de.drop_empty_rows(df)) == 1


def test_validate_bronze_rejects_missing_columns():
    with pytest.raises(ValueError, match="Faltan columnas"):
        de.validate_bronze(pd.DataFrame({"a": [1]}), ["a", "b"], "prueba")


# ---------------------------------------------------------------------------
# Casos
# ---------------------------------------------------------------------------

def _ins(**cambios) -> pd.DataFrame:
    fila = dict.fromkeys(de.EXPECTED_INS, "")
    fila.update({"CONSECUTIVE": "1", "FEC_NOT": "10/03/2020 12:00:00 a. m.", "SEMANA": "10",
                 "ANO": "2020", "EDAD": "6", "UNI_MED": "2", "SEXO": "f", "TIP_SS": "S",
                 "PER_ETN": "6", "estrato": "2", "GP_MIGRANT": "1", "COD_DPTO_R": "76",
                 "COD_MUN_R": "001", "FEC_CON": "3/03/2020", "INI_SIN": "1/03/2020",
                 "TIP_CAS": "3", "PAC_HOS": "1", "CON_FIN": "2", "AJUSTE": "0"})
    fila.update(cambios)
    return pd.DataFrame([fila])


def test_harmonize_ins_converts_units_and_codes():
    out = de.harmonize_ins(_ins())
    fila = out.iloc[0]
    assert fila["edad"] == pytest.approx(0.5)              # 6 meses
    assert fila["sexo"] == "F" and fila["afiliacion"] == "Subsidiado"
    assert fila["tipo_caso"] == "Confirmado por laboratorio"
    assert bool(fila["hospitalizado"]) and bool(fila["migrante"]) and bool(fila["fallecido"])
    assert bool(fila["residencia_cali"])
    assert pd.isna(fila["comuna"]) and pd.isna(fila["clasificacion"])   # el INS no los publica


def test_drop_discarded_ins_removes_cases_the_sivigila_annulled():
    df = pd.concat([_ins(AJUSTE="0"), _ins(AJUSTE="6"), _ins(AJUSTE="d"), _ins(AJUSTE="7")])
    assert de.drop_discarded_ins(df)["AJUSTE"].tolist() == ["0", "7"]


def _casos(**columnas) -> pd.DataFrame:
    base = {"fuente": de.FUENTE_CALI, "evento": de.EVENTO_DENGUE}
    return pd.DataFrame({**base, **columnas})


def test_drop_duplicate_cases_only_touches_the_district_source():
    fila = dict.fromkeys(de.DEDUP_KEY, "x")
    df = pd.DataFrame([fila, fila, fila, fila]).assign(
        fuente=[de.FUENTE_CALI, de.FUENTE_CALI, de.FUENTE_INS, de.FUENTE_INS])
    out = de.drop_duplicate_cases(df)
    assert out["fuente"].tolist() == [de.FUENTE_CALI, de.FUENTE_INS, de.FUENTE_INS]


def test_mark_deaths_links_the_death_to_its_case_without_counting_it_twice():
    llave = {"edad": 35.0, "sexo": "M", "comuna": 17,
             "fecha_inicio_sintomas": pd.Timestamp("2023-07-17"),
             "fecha_consulta": pd.Timestamp("2023-07-20")}
    casos = pd.DataFrame([llave | {"id_caso": "a", "fallecido": False},
                          llave | {"edad": 40.0, "id_caso": "b", "fallecido": False}])
    muertes = pd.DataFrame([llave | {"id_caso": "m"}])
    out = de.mark_deaths(casos, muertes)
    assert len(out) == 2 and out["fallecido"].tolist() == [True, False]
    huerfana = pd.DataFrame([llave | {"edad": 90.0, "id_caso": "m2"}])
    assert len(de.mark_deaths(casos, huerfana)) == 3            # sin caso: se agrega


def test_add_epi_week_respects_the_official_week_and_flags_differences():
    casos = pd.DataFrame({
        "fecha_inicio_sintomas": pd.to_datetime(["2020-03-01", "2020-03-01", "2023-01-01"]),
        "anio_reportado": pd.array([2020, 2020, pd.NA], dtype="Int64"),
        "semana_reportada": pd.array([10, 11, pd.NA], dtype="Int64"),
    })
    out = de.add_epi_week(casos)
    assert out["semana_epi"].tolist() == [10, 11, 1]
    assert out["semana_inconsistente"].tolist() == [False, True, False]


def test_quality_flags_mark_instead_of_delete():
    casos = _casos(
        dias_sintomas_consulta=pd.array([2, -1, 40], dtype="Int64"),
        dias_consulta_notificacion=pd.array([1, 0, 0], dtype="Int64"),
        comuna=pd.array([6, pd.NA, pd.NA], dtype="Int64"),
        comuna_original=["6", "SD", "100"],
    )
    out = de.add_quality_flags(casos)
    assert len(out) == 3
    assert out["fecha_incoherente"].tolist() == [False, True, False]
    assert out["intervalo_atipico"].tolist() == [False, False, True]
    assert out["estado_comuna"].tolist() == ["Válida", "Sin dato", "Código no válido"]


def test_severity_is_missing_when_there_is_no_classification():
    casos = _casos(clasificacion=["Sin signos de alarma", "Con signos de alarma",
                                  "Dengue grave", None])
    assert de.add_severity(casos)["severo"].tolist() == [False, True, True, pd.NA]


def test_functions_do_not_mutate_their_input():
    casos = _casos(edad=[3.0, 70.0], fecha_inicio_sintomas=pd.to_datetime(["2023-01-01"] * 2),
                   fecha_consulta=pd.to_datetime(["2023-01-02"] * 2),
                   fecha_notificacion=pd.to_datetime(["2023-01-03"] * 2),
                   comuna=pd.array([6, 51], dtype="Int64"),
                   clasificacion=["Sin signos de alarma", "Dengue grave"])
    antes = casos.copy(deep=True)
    for fn in (de.add_age_group, de.add_delays, de.add_zone, de.add_severity):
        fn(casos)
    pd.testing.assert_frame_equal(casos, antes)
    assert de.add_age_group(casos)["grupo_edad"].tolist() == ["0-4", "65+"]
    assert de.add_zone(casos)["zona"].tolist() == ["Urbana", "Rural"]


# ---------------------------------------------------------------------------
# Población y serie semanal
# ---------------------------------------------------------------------------

def test_project_poblacion_extends_the_linear_trend_and_marks_it():
    pob = pd.DataFrame({"territorio": "TOTAL CALI", "anio": range(2016, 2021),
                        "poblacion": [100, 110, 120, 130, 140], "tipo": "Ciudad",
                        "comuna": pd.array([pd.NA] * 5, dtype="Int64")})
    out = de.project_poblacion(pob, anio_fin=2022)
    nuevas = out[out["proyectada"]]
    assert nuevas["anio"].tolist() == [2021, 2022]
    assert nuevas["poblacion"].tolist() == [150, 160]
    assert not out.loc[out["anio"] <= 2020, "proyectada"].any()


def _poblacion_dap():
    """Planeación: ciudad, cabecera, dos comunas, 'Otros', zona rural y un corregimiento."""
    filas = []
    for anio in range(2016, 2021):
        d = anio - 2016
        filas += [("TOTAL CALI", anio, 1_100 + 10 * d), ("Total Cabecera", anio, 1_000 + 10 * d),
                  ("Comuna 1", anio, 600 + 6 * d), ("Comuna 2", anio, 300 + 3 * d),
                  ("Otros", anio, 100 + d), ("Total Corregimientos", anio, 100),
                  ("Navarro", anio, 100)]
    base = pd.DataFrame(filas, columns=["territorio", "anio", "poblacion"])
    return de.add_territory_type(base)


def _dane(anios=range(2018, 2023)):
    filas = []
    for anio in anios:
        filas += [(str(float(anio)), "Total", "2000.0"), (str(float(anio)), "Cabecera Municipal",
                                                          "1800.0"),
                  (str(float(anio)), "Centros Poblados y Rural Disperso", "200.0")]
    return pd.DataFrame(filas, columns=["AÑO", "ÁREA GEOGRÁFICA", "TOTAL"])


def test_adjust_to_dane_uses_official_totals_and_planning_shares():
    out = de.adjust_to_dane(_poblacion_dap(), de.tidy_poblacion_dane(_dane()), anio_fin=2022)
    por = out.set_index(["territorio", "anio"])["poblacion"]
    assert por[("TOTAL CALI", 2020)] == 2000 and por[("TOTAL CALI", 2017)] == 1110
    # 2020: la comuna 1 tiene 624 de 1.040 en la cabecera de Planeación (60 %)
    assert por[("Comuna 1", 2020)] == round(1800 * 624 / 1040)
    assert por[("Navarro", 2021)] == 200                      # único corregimiento
    # Después de 2020 se conserva la participación de 2020 y se marca como proyectada
    assert por[("Comuna 2", 2022)] == round(1800 * 312 / 1040)
    marcas = out.set_index(["territorio", "anio"])
    assert bool(marcas.loc[("Comuna 2", 2022), "proyectada"])
    assert not bool(marcas.loc[("TOTAL CALI", 2022), "proyectada"])     # total oficial
    assert marcas.loc[("TOTAL CALI", 2022), "fuente_poblacion"] == "DANE"
    assert marcas.loc[("TOTAL CALI", 2016), "fuente_poblacion"] == "Planeación Distrital"
    with pytest.raises(ValueError):
        de.adjust_to_dane(_poblacion_dap(), de.tidy_poblacion_dane(_dane(range(2018, 2021))),
                          anio_fin=2022)


def test_build_poblacion_falls_back_to_the_linear_trend_without_dane():
    ancho = pd.DataFrame({"Descripción": ["TOTAL CALI", "Comuna 1"],
                          **{str(a): [1_000 + 10 * i, 600 + 6 * i]
                             for i, a in enumerate(range(2016, 2021))}})
    sin_dane = de.build_poblacion(ancho, anio_fin=2022)
    assert set(sin_dane["fuente_poblacion"]) == {"Planeación Distrital",
                                                 "Tendencia lineal de Planeación"}
    con_dane = de.build_poblacion(ancho, anio_fin=2022, dane=_dane())
    ciudad = con_dane[con_dane["tipo"] == "Ciudad"].set_index("anio")["poblacion"]
    assert ciudad[2022] == 2000 and ciudad[2017] == 1010


def test_clima_ciudad_averages_urban_stations_with_enough_readings():
    clima = pd.DataFrame({
        "codigo_estacion": ["0026055120", "0026085160", "0026085170", "0026055100"],
        "anio_epi": 2024, "semana_epi": 10,
        "lluvia_mm": [10.0, 20.0, 5.0, 90.0], "cobertura_lluvia": [1.0, 0.5, 0.2, 1.0],
        "temp_media": [24.0, 26.0, 30.0, 15.0], "cobertura_temp": [1.0, 1.0, 0.1, 1.0],
        "sistema": "actual",
    })
    out = de.clima_ciudad(clima).iloc[0]
    # Siloé: 20 mm con la mitad de las lecturas -> 40 mm en la semana completa; la Base Aérea
    # (20 % de lecturas) y Farallones (no urbana) no cuentan
    assert out["lluvia_mm"] == pytest.approx((10 + 40) / 2)
    assert out["temp_media"] == pytest.approx(25.0)
    assert (out["estaciones_lluvia"], out["estaciones_temp"]) == (2, 2)


def test_union_clima_joins_manual_and_api_downloads():
    manual = pd.DataFrame({"codigoestacion": ["1"], "valorobservado": ["0.1"]})
    api = pd.DataFrame({"codigoestacion": ["1", "2"], "valorobservado": ["0.1", "0.3"]})
    lluvia, temperatura = de.union_clima({"precipitacion": manual, "precipitacion_api": api,
                                          "temperatura": manual})
    assert len(lluvia) == 3 and len(temperatura) == 1


def _serie_semanal(casos, fin_semanas):
    return pd.DataFrame({"anio_epi": 2023, "semana_epi": range(1, len(casos) + 1),
                         "fecha_fin": pd.to_datetime(fin_semanas), "casos": casos,
                         "hospitalizados": 0, "confirmados": 0, "menores_15": 0,
                         "casos_graves": np.nan})


def test_add_coverage_marks_partial_and_missing_weeks_after_the_cut():
    serie = _serie_semanal([10, 12, 5, np.nan],
                           ["2023-11-18", "2023-11-25", "2023-12-02", "2023-12-09"])
    casos = pd.DataFrame({"anio_epi": [2023], "fecha_notificacion": [pd.Timestamp("2023-12-05")]})
    out = de.add_coverage(serie, casos, semanas_rezago=1)
    assert out["cobertura"].tolist() == ["completa", "completa", "parcial", "sin_datos"]
    assert out["casos"].isna().tolist() == [False, False, False, True]   # sin datos ≠ cero


def test_impute_gaps_keeps_observed_values_and_fills_the_rest_in_log_scale():
    serie = pd.DataFrame({"casos": [100, 50, np.nan, 400],
                          "cobertura": ["completa", "parcial", "sin_datos", "completa"]})
    out = de.impute_gaps(serie)
    assert out["imputada"].tolist() == [False, True, True, False]
    assert out["casos"].iloc[1] == 50                      # el dato observado no se toca
    assert out["casos_modelo"].iloc[0] == 100 and out["casos_modelo"].iloc[3] == 400
    assert 100 < out["casos_modelo"].iloc[1] < out["casos_modelo"].iloc[2] < 400


# ---------------------------------------------------------------------------
# De punta a punta (fuentes inventadas)
# ---------------------------------------------------------------------------

def test_build_silver_end_to_end(fake_bronze):
    silver = de.build_silver(de.read_bronze(fake_bronze))
    casos = silver[de.SILVER_CASOS]

    # 2022: 2 casos + 1 repetido; 2023: 3 casos + 1 grave; INS: 3 + 2 + 2 + 1 de Cali,
    # de los cuales 1 fue anulado (AJUSTE = 6)
    assert (casos["fuente"] == de.FUENTE_CALI).sum() == 2 + 3 + 1
    assert (casos["fuente"] == de.FUENTE_INS).sum() == 8 - 1
    assert casos["id_caso"].is_unique
    assert casos["fallecido"].sum() == 1
    assert casos.loc[casos["fallecido"], "evento"].item() == de.EVENTO_GRAVE
    assert set(casos.loc[casos["comuna"].isna() & (casos["fuente"] == de.FUENTE_CALI),
                         "estado_comuna"]) == {"Sin dato", "Código no válido"}

    serie = silver[de.SILVER_SERIE]
    assert len(serie) == 314                                    # semanas 2020-2025
    assert serie["casos_modelo"].notna().all()
    assert (serie["fecha_inicio"].diff().dropna() == pd.Timedelta(weeks=1)).all()
    assert serie.loc[serie["imputada"], "cobertura"].ne("completa").all()

    poblacion = silver[de.SILVER_POBLACION]
    ciudad = poblacion[poblacion["tipo"] == "Ciudad"]
    assert ciudad["anio"].max() == 2025 and ciudad["proyectada"].sum() == 5

    clima = silver[de.SILVER_CLIMA]
    assert clima["temp_max"].max() < 45                         # la lectura de 99 °C se marcó

    resumen = de.cleaning_summary(de.read_bronze(fake_bronze), silver)
    assert resumen["duplicados_eliminados"] == 1
    assert resumen["registros_ins_anulados"] == 1
    assert resumen["filas_vacias_descartadas"]["total"] == 3 + 2


def test_build_silver_with_complementary_sources(fake_sources, tmp_path):
    from tests.conftest import write_fake_complementos

    ing = importlib.import_module("flows.00_ingestion")
    write_fake_complementos(fake_sources)
    ing.run(source_dir=fake_sources, out_dir=tmp_path / "bronze")
    bronze = de.read_bronze(tmp_path / "bronze")
    silver = de.build_silver(bronze)
    poblacion = silver[de.SILVER_POBLACION]
    ciudad = poblacion[poblacion["tipo"] == "Ciudad"].set_index("anio")
    assert ciudad.loc[2024, "poblacion"] == 2_056_000 and not ciudad["proyectada"].any()
    serie = silver[de.SILVER_SERIE]
    semana = serie[(serie["anio_epi"] == 2024) & (serie["semana_epi"] == 23)].iloc[0]
    assert semana["lluvia_mm"] == pytest.approx(0.1 * 7 * 24 * 6)        # 2 estaciones
    assert semana["temp_media"] == 25 and semana["estaciones_lluvia"] == 2
    resumen = de.cleaning_summary(bronze, silver)
    assert "DANE" in resumen["poblacion"] and resumen["clima"]["semanas_con_lluvia"] >= 1


def test_run_writes_the_five_silver_tables(fake_bronze, tmp_path):
    out = tmp_path / "silver"
    de.run(in_dir=fake_bronze, out_dir=out)
    for nombre in (de.SILVER_CASOS, de.SILVER_SERIE, de.SILVER_POBLACION, de.SILVER_CLIMA,
                   de.SILVER_CENTINELA, de.SILVER_CLEANING_NAME):
        assert (out / nombre).exists()


def test_temperature_anomaly_removes_the_measurement_system_jump():
    semanas = range(1, 7)
    clima = pd.DataFrame({
        "codigo_estacion": "0026055120", "anio_epi": 2024, "semana_epi": list(semanas),
        "lluvia_mm": 10.0, "cobertura_lluvia": 1.0, "cobertura_temp": 1.0,
        "sistema": ["anterior"] * 3 + ["actual"] * 3,
        "temp_media": [28.0, 29.0, 27.0, 24.0, 25.0, 23.0],      # salto de 4 °C del sistema
    })
    out = de.clima_ciudad(clima)
    assert out["temp_anomalia"].tolist() == [0.0, 1.0, -1.0, 0.0, 1.0, -1.0]
    assert out["temp_media"].tolist() == [28.0, 29.0, 27.0, 24.0, 25.0, 23.0]


def test_sistema_medicion_reads_the_sensor_description():
    descripcion = pd.Series(["Temp Aire 2 m", "TEMPERATURA DEL AIRE A 2 m", "Precipitacion",
                             "PRECIPITACIÓN"])
    assert de.sistema_medicion(descripcion).tolist() == ["anterior", "actual", "anterior",
                                                         "actual"]
