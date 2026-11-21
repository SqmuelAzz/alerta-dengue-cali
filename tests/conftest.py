"""Datos de prueba: una versión diminuta e inventada de las 13 fuentes.

Reproduce los formatos reales (separadores, codificación latin-1, fechas en tres
formatos, filas vacías, códigos de comuna sin dato, microdatos del INS en Excel) con
unas pocas filas, para probar el pipeline sin depender de los archivos originales.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

COLUMNAS_SIVIGILA = [
    "codigo_sspd", "fecha_nototificacion", "edad", "nacionalidad", "sexo", "comuna",
    "tipo_seguridad_social", "etnia", "estrato", "ciudad_residencia", "dpto_residencia",
    "fecha_consulta", "fecha_inicio_sintomas", "tipo_caso", "Hospitalizado",
    "fecha_de_hospitalizacion", "deplazamiento_ultimos_15_dias", "municipio_desplazamiento",
    "fiebre", "cefalea", "dolor_retro_ocular", "malgias", "artralgia", "erupcion",
    "dolor_abdominal", "vomito", "diarrea", "somnolencia", "hipotension", "hepatomegalia",
    "hem_mucosa", "hipotermia", "aum_hemato", "caida_plaquetas", "acumulacion_liquidos",
    "extravasacion", "hemorragia", "choque", "daño_organo", "clasificacion_final", "conducta",
    "nombre_municipio_procedencia",
]
COLUMNAS_INS = [
    "CONSECUTIVE", "COD_EVE", "FEC_NOT", "SEMANA", "ANO", "EDAD", "UNI_MED", "SEXO",
    "COD_DPTO_O", "COD_MUN_O", "TIP_SS", "PER_ETN", "estrato", "GP_MIGRANT", "COD_DPTO_R",
    "COD_MUN_R", "FEC_CON", "INI_SIN", "TIP_CAS", "PAC_HOS", "FEC_HOS", "CON_FIN", "AJUSTE",
]


def _sivigila_row(fecha_not, edad, sexo, comuna, inicio, consulta, clasificacion, si="Si",
                  no="No", **extra) -> dict:
    fila = dict.fromkeys(COLUMNAS_SIVIGILA, no)
    fila.update({
        "codigo_sspd": "2,10202E+11", "fecha_nototificacion": fecha_not, "edad": edad,
        "nacionalidad": "Colombia", "sexo": sexo, "comuna": comuna,
        "tipo_seguridad_social": "Contributivo", "etnia": "Otro", "estrato": "2",
        "ciudad_residencia": "Cali", "dpto_residencia": "Valle", "fecha_consulta": consulta,
        "fecha_inicio_sintomas": inicio, "tipo_caso": "Probable", "Hospitalizado": no,
        "fecha_de_hospitalizacion": "-   -", "municipio_desplazamiento": "",
        "fiebre": si, "clasificacion_final": clasificacion, "conducta": "Ambulatorio",
        "nombre_municipio_procedencia": "Cali",
    })
    fila.update(extra)
    return fila


def _write_csv(path: Path, columnas: list[str], filas: list[dict], sep=";",
               encoding="latin-1", vacias: int = 0) -> None:
    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columnas, delimiter=sep)
        writer.writeheader()
        writer.writerows(filas)
        for _ in range(vacias):                       # filas vacías como en el portal
            f.write(sep * (len(columnas) - 1) + "\r\n")


def _ins_row(consecutivo, anio, semana, inicio, mun_o="001", dpto_o="76", ajuste="0",
             edad="20", uni="1") -> list[str]:
    fecha = f"{inicio} 12:00:00 a.\xa0m."
    return [consecutivo, "210", fecha, semana, str(anio), edad, uni, "F", dpto_o, mun_o, "S",
            "6", "2", "2", "76", "001", fecha, fecha, "3", "2", "", "1", ajuste]


def write_fake_sources(folder: Path) -> Path:
    """Crea en 'folder' los 13 archivos con el nombre con que se descargan."""
    folder.mkdir(parents=True, exist_ok=True)

    # SIVIGILA del Distrito 2022: fechas 'dd-mmm-aaaa', SI/NO, desplazamiento 1/2, filas vacías
    filas_2022 = [
        _sivigila_row("05-ene-2022", "8", "F", "6", "02-ene-2022", "04-ene-2022",
                      "Sin signos de alarma", si="SI", no="NO",
                      deplazamiento_ultimos_15_dias="2", nacionalidad="COLOMBIA"),
        _sivigila_row("12-ene-2022", "70", "M", "SD", "08-ene-2022", "11-ene-2022",
                      "Con signos de alarma", si="SI", no="NO",
                      deplazamiento_ultimos_15_dias="1", nacionalidad="VENEZUELA"),
    ]
    filas_2022.append(dict(filas_2022[0]))            # notificación repetida
    _write_csv(folder / "bd_dengue.csv", COLUMNAS_SIVIGILA + ["fecha_corte"],
               [f | {"fecha_corte": "13-dic-2022"} for f in filas_2022], vacias=3)

    # SIVIGILA del Distrito 2023: fechas 'dd/mm/aaaa', Si/No, comuna no válida
    filas_2023 = [
        _sivigila_row("03/01/2023", "30", "F", "14", "01/01/2023", "02/01/2023",
                      "Con signos de alarma", estrato="0"),
        _sivigila_row("20/02/2023", "45", "M", "100", "15/02/2023", "19/02/2023",
                      "Sin signos de alarma"),
        _sivigila_row("10/03/2023", "3", "F", "51", "05/03/2023", "09/03/2023",
                      "Sin signos de alarma", estrato="1"),
    ]
    _write_csv(folder / "53.-bd_casos-notificados-en-sivigila-dengue-corte-6-dic.csv",
               COLUMNAS_SIVIGILA, filas_2023)

    # Dengue grave 2023 (columnas con otros nombres) y su registro de mortalidad
    renombrar = {"dpto_residencia": "Departamento_residencia", "malgias": "mialgias",
                 "hem_mucosa": "hemorragia_mucosa", "aum_hemato": "aumento_hematocrito",
                 "deplazamiento_ultimos_15_dias": "deplazamiento_ultimos_SI5_dias"}
    grave = _sivigila_row("16/01/2023", "64", "M", "8", "09/01/2023", "14/01/2023",
                          "Dengue grave", conducta="Unidad de cuidados intensivos",
                          Hospitalizado="Si")
    columnas_grave = [renombrar.get(c, c) for c in COLUMNAS_SIVIGILA]
    _write_csv(folder / "54.-bd_casos-notificados-en-sivigila-para-dengue_grave-corte-6-dic.csv",
               columnas_grave, [{renombrar.get(k, k): v for k, v in grave.items()}])
    columnas_muerte = [c for c in columnas_grave if c not in ("clasificacion_final", "conducta")]
    muerte = {renombrar.get(k, k): v for k, v in grave.items()
              if k not in ("clasificacion_final", "conducta")}
    muerte["tip_caso"] = muerte.pop("tipo_caso")
    columnas_muerte = ["tip_caso" if c == "tipo_caso" else c for c in columnas_muerte]
    _write_csv(folder / "55.-bd_casos-notificados-en-sivigila-mortalidad_dengue-corte-6-dic.csv",
               columnas_muerte + ["fecha_de_defuncion"],
               [muerte | {"fecha_de_defuncion": "20/01/2023"}], vacias=2)

    # Microdatos del INS: un Excel por año con filas de Cali y de otros municipios
    ins = {
        2020: [_ins_row("1", 2020, "1", "30/12/2019"), _ins_row("2", 2020, "10", "3/03/2020"),
               _ins_row("3", 2020, "10", "3/03/2020", mun_o="520"),
               _ins_row("4", 2020, "53", "29/12/2020", ajuste="6")],
        2021: [_ins_row("5", 2021, "5", "3/02/2021", edad="6", uni="2"),
               _ins_row("6", 2021, "52", "28/12/2021")],
        2024: [_ins_row("7", 2024, "21", "20/05/2024"), _ins_row("8", 2024, "21", "21/05/2024"),
               _ins_row("9", 2024, "21", "21/05/2024", dpto_o="05")],
        2025: [_ins_row("10", 2025, "53", "1/01/2026")],
    }
    for anio, filas in ins.items():
        libro = Workbook()
        hoja = libro.active
        hoja.append(COLUMNAS_INS)
        for fila in filas:
            hoja.append(fila)
        libro.save(folder / f"Datos_{anio}_210.xlsx")

    # Población (formato ancho, encabezado con tilde, UTF-8)
    anios = list(range(2005, 2021))
    filas_pob = [["TOTAL CALI", *[2_000_000 + 25_000 * i for i in range(len(anios))]],
                 ["Comuna 6", *[150_000 + 3_000 * i for i in range(len(anios))]],
                 ["Comuna 14", *[140_000 + 2_500 * i for i in range(len(anios))]],
                 ["Comuna 8", *[100_000 + 500 * i for i in range(len(anios))]],
                 ["Navarro", *[1_000 - 10 * i for i in range(len(anios))]]]
    with (folder / "proyecciones_de_poblaci_n_de_cali_por_comuna_y_corregimiento__2006-2020.csv"
          ).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Descripción", *anios])
        writer.writerows(filas_pob)

    # Centinela de hospitalizaciones en menores de 5 años
    _write_csv(folder / "14.-unificado-centinela-gsa-sspm-2010-2020.csv",
               ["ano", "comuna", "barrio", "tot_hosp", "num_hosp_ERA", "num_hosp_asma",
                "num_hosp_EDA", "num_hosp_deng_chik_zika"],
               [{"ano": 2019, "comuna": "6", "barrio": "Floralia ", "tot_hosp": 10,
                 "num_hosp_ERA": 4, "num_hosp_asma": 1, "num_hosp_EDA": 2,
                 "num_hosp_deng_chik_zika": 3},
                {"ano": 2020, "comuna": "6", "barrio": "Floralia ", "tot_hosp": 8,
                 "num_hosp_ERA": 3, "num_hosp_asma": 1, "num_hosp_EDA": 1,
                 "num_hosp_deng_chik_zika": 2},
                {"ano": 2020, "comuna": "Rural", "barrio": "x", "tot_hosp": 1,
                 "num_hosp_ERA": 1, "num_hosp_asma": 0, "num_hosp_EDA": 0,
                 "num_hosp_deng_chik_zika": 0}])

    # IDEAM: precipitación, temperatura (una lectura fuera de rango) y catálogo
    clima = ["codigoestacion", "codigosensor", "fechaobservacion", "valorobservado",
             "nombreestacion", "departamento", "municipio", "zonahidrografica", "latitud",
             "longitud", "descripcionsensor", "unidadmedida"]
    base = {"codigoestacion": "0026055120", "nombreestacion": "UNIVERSIDAD DEL VALLE  - AUT",
            "departamento": "VALLE DEL CAUCA", "municipio": "CALI", "zonahidrografica": "CAUCA",
            "latitud": "3.378", "longitud": "-76.53"}
    _write_csv(folder / "s54a-sgyg.csv", clima, [
        base | {"codigosensor": "0240", "fechaobservacion": "2023-01-02T07:10:00.000",
                "valorobservado": "0.2", "descripcionsensor": "Precipitacion",
                "unidadmedida": "mm"},
        base | {"codigosensor": "0240", "fechaobservacion": "2023-01-09T07:10:00.000",
                "valorobservado": "1.0", "descripcionsensor": "Precipitacion",
                "unidadmedida": "mm"}], sep=",", encoding="utf-8")
    _write_csv(folder / "sbwg-7ju4.csv", clima, [
        base | {"codigosensor": "0068", "fechaobservacion": "2023-01-02T12:00:00.000",
                "valorobservado": "31.6", "descripcionsensor": "Temp Aire 2 m",
                "unidadmedida": "°C"},
        base | {"codigosensor": "0068", "fechaobservacion": "2023-01-03T12:00:00.000",
                "valorobservado": "99", "descripcionsensor": "Temp Aire 2 m",
                "unidadmedida": "°C"}], sep=",", encoding="utf-8")
    _write_csv(folder / "Cat_logo_Nacional_de_Estaciones_del_IDEAM.csv",
               ["Codigo", "Nombre", "Categoria", "Tecnologia", "Estado", "Altitud", "LONGITUD",
                "LATITUD"],
               [{"Codigo": "0026055120", "Nombre": "UNIVERSIDAD DEL VALLE  - AUT [26055120]",
                 "Categoria": "Climatológica Principal", "Tecnologia": "Automática",
                 "Estado": "Activa", "Altitud": "996", "LONGITUD": "-76.53", "LATITUD": "3.378"}],
               sep=",", encoding="utf-8")
    return folder


def write_fake_complementos(folder: Path) -> Path:
    """Fuentes de 'make descargar': series del IDEAM por la API y proyecciones del DANE."""
    clima = ["codigoestacion", "codigosensor", "fechaobservacion", "valorobservado",
             "nombreestacion", "departamento", "municipio", "zonahidrografica", "latitud",
             "longitud", "descripcionsensor", "unidadmedida"]
    base = {"departamento": "VALLE DEL CAUCA", "municipio": "CALI", "zonahidrografica": "CAUCA",
            "latitud": "3.4", "longitud": "-76.5"}
    lluvia, temperatura = [], []
    for codigo, nombre in (("0026055120", "UNIVERSIDAD DEL VALLE  - AUT"),
                           ("0026085160", "SILOE - AUT")):
        for i in range(7 * 24 * 6):                  # una semana completa cada 10 minutos
            hora = pd.Timestamp("2024-06-02") + pd.Timedelta(minutes=10 * i)
            lluvia.append(base | {"codigoestacion": codigo, "nombreestacion": nombre,
                                  "codigosensor": "0240", "valorobservado": "0.1",
                                  "fechaobservacion": hora.strftime("%Y-%m-%dT%H:%M:%S.000"),
                                  "descripcionsensor": "Precipitacion", "unidadmedida": "mm"})
        for i in range(7 * 24):
            hora = pd.Timestamp("2024-06-02") + pd.Timedelta(hours=i)
            temperatura.append(base | {"codigoestacion": codigo, "nombreestacion": nombre,
                                       "codigosensor": "0068", "valorobservado": "25",
                                       "fechaobservacion": hora.strftime("%Y-%m-%dT%H:%M:%S.000"),
                                       "descripcionsensor": "Temp Aire 2 m",
                                       "unidadmedida": "°C"})
    _write_csv(folder / "ideam_precipitacion_cali_2020-2025.csv", clima, lluvia, sep=",",
               encoding="utf-8")
    _write_csv(folder / "ideam_temperatura_cali_2020-2025.csv", clima, temperatura, sep=",",
               encoding="utf-8")

    libro = Workbook()
    libro.active.title = "Índice"
    libro.create_sheet("PPED").append(["Proyecciones"])
    hoja = libro.create_sheet("PobMunicipalxÁrea")
    hoja.append(["PROYECCIONES DE POBLACIÓN POR ÁREA. AÑOS 2018 A 2042"])
    hoja.append([])
    hoja.append(["DP", "DPNOM", "MPIO", "DPMP", "AÑO", "ÁREA GEOGRÁFICA", "TOTAL"])
    hoja.append([])
    for anio in range(2018, 2026):
        for area, total in (("Cabecera Municipal", 2_000_000), ("Centros Poblados y Rural "
                                                                "Disperso", 50_000),
                            ("Total", 2_050_000)):
            hoja.append(["76", "Valle del Cauca", "76001", "Cali", float(anio), area,
                         float(total + 1_000 * (anio - 2018))])
        hoja.append(["05", "Antioquia", "05001", "Medellín", float(anio), "Total", 2_500_000.0])
    libro.save(folder / "PPED-AreaMun-2018-2042_VP.xlsx")
    return folder


@pytest.fixture
def fake_sources(tmp_path) -> Path:
    """Carpeta con las 13 fuentes inventadas."""
    return write_fake_sources(tmp_path / "fuentes")


@pytest.fixture
def fake_bronze(fake_sources, tmp_path) -> Path:
    """Bronze construido por la ingesta real a partir de las fuentes inventadas."""
    import importlib

    ing = importlib.import_module("flows.00_ingestion")
    out = tmp_path / "bronze"
    ing.run(source_dir=fake_sources, out_dir=out)
    return out
