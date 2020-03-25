.PHONY: quality style test

quality:
	black --check --line-length 119 --target-version py35 flows examples tests
	isort --check-only --recursive flows examples tests
	flake8 flows examples tests

style:
	black --line-length 119 --target-version py35 flows examples tests
	isort --recursive flows examples tests

test:
	python -m unittest discover -s tests -t . -v
