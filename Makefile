.PHONY: test lint lint-fix type check clean

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

lint-fix:
	ruff check --fix .
	ruff format .

type:
	mypy core/ llm/ tools/ skills/ sessions/

check: lint type test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache