import importlib
import json

import pandas as pd
import pytest

from flows.common.hashing import sha256_of

ing = importlib.import_module("flows.00_ingestion")


def test_split_procedencia_keeps_only_the_municipality():
    rows = [
        ["CONSECUTIVE", "COD_DPTO_O", "COD_MUN_O"],
        ["1", "76", "001"],
        ["2", "76", "520"],          # otro municipio del Valle
        ["3", "05", "001"],          # mismo código de municipio, otro departamento
        ["4", " 76 ", "001 "],       # espacios alrededor del código
    ]
    header, kept, total = ing.split_procedencia(rows, "76", "001")
    assert header == ["CONSECUTIVE", "COD_DPTO_O", "COD_MUN_O"]
    assert total == 4
    assert [r[0] for r in kept] == ["1", "4"]
    assert kept[1] == ["4", " 76 ", "001 "]          # los valores no se alteran


def test_quality_report_counts_blank_text_as_missing():
    df = pd.DataFrame({"a": ["1", "", "3"], "b": ["x", "  ", None]})
    rep = ing.quality_report(df)
    assert rep["n_rows"] == 3 and rep["n_cols"] == 2
    assert rep["nulls"] == {"a": 1, "b": 2}
    assert rep["empty_rows"] == 1                    # la fila 2 solo tiene blancos
    assert rep["dtypes"] == {"a": "número", "b": "texto"}
    assert round(rep["null_pct"]["b"], 1) == 66.7


def test_find_source_explains_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="DENGUE_SOURCE_DIR"):
        ing.find_source(tmp_path, "no_existe_*.csv")


def test_run_writes_bronze_metadata_and_quality(fake_sources, tmp_path):
    out = tmp_path / "bronze"
    meta = ing.run(source_dir=fake_sources, out_dir=out)

    obligatorias = [s for s in ing.SOURCES if s.required]
    assert meta["n_files"] == len(obligatorias) == 13       # las complementarias no están
    assert (out / "_ingestion_metadata.json").exists()
    quality = json.loads((out / "_quality_report.json").read_text(encoding="utf-8"))
    assert set(quality) == set(meta["files"])

    # Los CSV son copias exactas: misma huella que el archivo descargado
    csv_2023 = meta["files"]["sivigila_2023"]
    assert csv_2023["sha256_source"] == csv_2023["sha256_bronze"]
    assert len(csv_2023["sha256_bronze"]) == 64

    # Del Excel del INS solo quedan las filas con procedencia en Cali (76 / 001)
    ins_2020 = meta["files"]["ins_2020"]
    assert ins_2020["rows_source"] == 4 and ins_2020["rows_bronze"] == 3
    assert "COD_MUN_O" in ins_2020["filter"]


def test_bronze_csv_is_an_unmodified_copy(fake_sources, tmp_path):
    out = tmp_path / "bronze"
    ing.run(source_dir=fake_sources, out_dir=out)
    origen = fake_sources / "bd_dengue.csv"
    assert sha256_of(out / "sivigila_cali_dengue_2022.csv") == sha256_of(origen)
    assert (out / "sivigila_cali_dengue_2022.csv").read_bytes() == origen.read_bytes()


def test_unchanged_sources_are_not_processed_again(fake_sources, tmp_path, monkeypatch):
    out = tmp_path / "bronze"
    primera = ing.run(source_dir=fake_sources, out_dir=out)

    def falla(*args, **kwargs):
        raise AssertionError("no debía volver a procesar una fuente sin cambios")

    monkeypatch.setattr(ing, "_ingest_csv", falla)
    monkeypatch.setattr(ing, "_ingest_ins_excel", falla)
    segunda = ing.run(source_dir=fake_sources, out_dir=out)
    assert segunda["rows_bronze_total"] == primera["rows_bronze_total"]


def test_force_reprocesses_sources(fake_sources, tmp_path, monkeypatch):
    out = tmp_path / "bronze"
    ing.run(source_dir=fake_sources, out_dir=out)
    llamadas = []
    original = ing._ingest_csv
    monkeypatch.setattr(ing, "_ingest_csv",
                        lambda *a, **k: llamadas.append(1) or original(*a, **k))
    ing.run(source_dir=fake_sources, out_dir=out, force=True)
    assert len(llamadas) == 9          # los 9 CSV vuelven a copiarse


def test_complementary_sources_are_ingested_when_present(fake_sources, tmp_path):
    from tests.conftest import write_fake_complementos

    write_fake_complementos(fake_sources)
    out = tmp_path / "bronze"
    meta = ing.run(source_dir=fake_sources, out_dir=out)
    assert meta["n_files"] == len(ing.SOURCES) == 16
    dane = meta["files"]["poblacion_dane"]
    assert dane["rows_source"] == 32 and dane["rows_bronze"] == 24    # solo Cali (76001)
    assert dane["filter"] == "MPIO == '76001'"
    bronze = pd.read_csv(out / dane["bronze_file"], dtype=str)
    assert set(bronze["MPIO"]) == {"76001"} and "ÁREA GEOGRÁFICA" in bronze.columns
    api = meta["files"]["precipitacion_api"]
    assert api["sha256_source"] == api["sha256_bronze"]                  # copia exacta


def test_missing_complementary_source_removes_stale_bronze(fake_sources, tmp_path):
    from tests.conftest import write_fake_complementos

    out = tmp_path / "bronze"
    write_fake_complementos(fake_sources)
    ing.run(source_dir=fake_sources, out_dir=out)
    (fake_sources / "PPED-AreaMun-2018-2042_VP.xlsx").unlink()
    meta = ing.run(source_dir=fake_sources, out_dir=out)
    assert "poblacion_dane" not in meta["files"]
    assert not (out / "dane_poblacion_cali.csv").exists()


def test_split_municipio_dane_skips_titles_and_blank_rows():
    filas = [["Título"], [None], ["DP", "MPIO", "AÑO", "TOTAL"], [None, None, None, None],
             ["76", "76001", 2020.0, 10.0], ["05", "05001", 2020.0, 20.0]]
    header, kept, total = ing.split_municipio_dane(filas, "76001")
    assert header == ["DP", "MPIO", "AÑO", "TOTAL"]
    assert kept == [["76", "76001", "2020.0", "10.0"]] and total == 2
    with pytest.raises(ValueError):
        ing.split_municipio_dane([["sin encabezado"]], "76001")
