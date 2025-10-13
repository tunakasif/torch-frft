PROJECT_NAME:=torch_frft
EXECUTER:=uv run

all: format lint security test

clean:
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov

test:
	$(EXECUTER) pytest --cov-report term-missing --cov-report html --cov $(PROJECT_NAME)/

format:
	$(EXECUTER) ruff format .

lint:
	$(EXECUTER) ruff check . --fix
	$(EXECUTER) mypy .

security:
	$(EXECUTER) bandit -r $(PROJECT_NAME)/
