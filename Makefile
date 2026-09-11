-include Makefile.ch9
-include Makefile.ch10
.PHONY: ch8 ch8-extract ch8-build ch8-validate ch8-figures ch8-publish ch8-test ch8-clean

PY := python3
SOURCE ?= postgres

## Full Chapter 8 pipeline, in order.
ch8: ch8-extract ch8-build ch8-validate ch8-figures

## Stage 1 - raw staging tables (or CSV) -> typed interim parquet.
ch8-extract:
	$(PY) -m src.features.extract --source $(SOURCE)

## Stage 2 - engineer features, partition chronologically, persist encoders.
ch8-build:
	$(PY) -m src.features.build_features

## Stage 3 - leakage and contract checks; fails loudly.
ch8-validate:
	$(PY) -m src.features.validate

## Stage 4 - chapter figures and tables.
ch8-figures:
	$(PY) -m src.features.figures

## Optional - publish the feature sets back into PostgreSQL for Ch. 9 and Ch. 12.
ch8-publish:
	$(PY) -m src.db.load_features

## Unit and contract tests against the synthetic fixture.
ch8-test:
	$(PY) -m pytest tests -q

## Remove generated artifacts, keeping raw data untouched.
ch8-clean:
	rm -f data/interim/*.parquet data/processed/*.parquet
	rm -f artifacts/features_v1/*
