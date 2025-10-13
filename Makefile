PROJECT_NAME:=torch_frft
EXECUTER:=uv run

all: requirements format lint security test

clean:
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov

test:
	$(EXECUTER) pytest --cov-report term-missing --cov-report html --cov $(PROJECT_NAME)/

requirements:
	uv export --no-hashes --format requirements-txt > requirements.txt

format:
	$(EXECUTER) ruff format .

lint:
	$(EXECUTER) ruff check . --fix
	$(EXECUTER) mypy .

security:
	$(EXECUTER) bandit -r $(PROJECT_NAME)/
