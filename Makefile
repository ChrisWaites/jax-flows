.PHONY: quality style test test-examples

quality:
	black --check --line-length 119 --target-version py35 examples templates tests src utils
	isort --check-only --recursive examples templates tests src utils
	flake8 examples templates tests src utils

style:
	black --line-length 119 --target-version py35 examples templates tests src utils
	isort --recursive examples templates tests src utils

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/
