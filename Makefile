.PHONY: test lint typecheck benchmark clean install

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	python -m py_compile conformal/core/predictor.py
	python -m py_compile conformal/core/quantile.py
	python -m py_compile conformal/core/classification.py
	python -m py_compile conformal/online/tracker.py
	python -m py_compile conformal/risk/control.py

typecheck:
	mypy conformal/ --ignore-missing-imports

benchmark:
	python benchmarks/coverage_efficiency.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	rm -rf *.egg-info build dist .mypy_cache .pytest_cache
