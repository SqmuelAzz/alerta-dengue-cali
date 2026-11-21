.PHONY: install descargar ingest silver gold train pipeline test lint api app clean

install:
	pip install -e ".[dev,app]"

descargar:
	python -m flows.descarga

ingest:
	python -m flows.job --stage ingestion

silver:
	python -m flows.job --stage silver

gold:
	python -m flows.job --stage gold

train:
	python -m flows.job --stage train

pipeline:
	python -m flows.job --stage all

test:
	pytest -q

lint:
	ruff check .

api:
	python -m app.api

app:
	streamlit run app/dashboard.py

clean:
	rm -rf data models .pytest_cache .ruff_cache job.log
