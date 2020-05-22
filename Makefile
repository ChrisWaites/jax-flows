.PHONY: quality style test

quality:
	black --check --line-length 119 --target-version py36 flows examples tests
	flake8 --max-line-length 119 flows examples tests

style:
	black --line-length 119 --target-version py36 flows examples tests

test:
	python -m pytest flows tests --capture=no --verbose --doctest-modules
