PROJECT_NAME:=torch_frft
EXECUTER:=poetry run

all: requirements format lint security test

clean:
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov

test:
	$(EXECUTER) pytest --cov-report term-missing --cov-report html --cov $(PROJECT_NAME)/

requirements:
	poetry export -f requirements.txt -o requirements.txt --with dev --without-hashes

format:
	$(EXECUTER) isort --diff -c .
	$(EXECUTER) black --diff --check --color .

lint:
	$(EXECUTER) mypy .
	$(EXECUTER) pyupgrade --py310-plus **/*.py

security:
	$(EXECUTER) bandit -r $(PROJECT_NAME)/
	$(EXECUTER) pip-audit
