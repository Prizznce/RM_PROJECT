.PHONY: all data test evaluate report clean

PYTHON = .venv/python
PYTEST = .venv/python -m pytest

all:
	$(PYTHON) -m src.run_all

clean:
	rmdir /s /q __pycache__ 2>nul || true
