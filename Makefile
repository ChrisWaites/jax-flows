.PHONY: quality style test

quality:
	black --check --line-length 119 --target-version py35 flax examples tests
	isort --check-only --recursive flax examples tests
	flake8 flax examples tests

style:
	black --line-length 119 --target-version py35 flax examples tests
	isort --recursive flax examples tests

test:
	python -m unittest discover -s tests -t . -v
