import importlib

import pytest

from flows import job


def test_stages_follow_the_medallion_order():
    assert list(job.STAGES) == ["ingestion", "silver", "gold", "train"]


@pytest.mark.parametrize("modulo", job.STAGES.values())
def test_every_stage_exposes_run(modulo):
    assert callable(importlib.import_module(modulo).run)


def test_stage_all_runs_every_stage_in_order(monkeypatch):
    ejecutadas = []
    monkeypatch.setattr(job, "run_stage", ejecutadas.append)
    job.main(["--stage", "all"])
    assert ejecutadas == list(job.STAGES)


def test_a_single_stage_runs_alone(monkeypatch):
    ejecutadas = []
    monkeypatch.setattr(job, "run_stage", ejecutadas.append)
    job.main(["--stage", "gold"])
    assert ejecutadas == ["gold"]


def test_unknown_stage_is_rejected():
    with pytest.raises(SystemExit):
        job.main(["--stage", "deploy"])


def test_run_stage_calls_the_module_run(monkeypatch):
    modulo = importlib.import_module("flows.03_aggregation")
    monkeypatch.setattr(modulo, "run", lambda: "listo")
    assert job.run_stage("gold") == "listo"
