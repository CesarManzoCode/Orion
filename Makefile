.PHONY: help install dev clean lint format type-check test test-cov test-unit test-integration test-e2e security run debug

help:
	@echo "🚀 Orion - Agente Personal Conversacional"
	@echo "Comandos disponibles:"
	@echo ""
	@echo "  make install       - Instalar dependencias (dev + extras)"
	@echo "  make dev           - Setup completo para desarrollo"
	@echo "  make clean         - Limpiar archivos temporales y cache"
	@echo ""
	@echo "  make lint          - Linting con ruff"
	@echo "  make format        - Formatear código con ruff"
	@echo "  make type-check    - Type checking con mypy"
	@echo ""
	@echo "  make test          - Ejecutar todos los tests"
	@echo "  make test-cov      - Tests con cobertura"
	@echo "  make test-unit     - Tests unitarios solamente"
	@echo "  make test-integration - Tests de integración"
	@echo "  make test-e2e      - Tests end-to-end"
	@echo ""
	@echo "  make security      - Análisis de seguridad (bandit + safety)"
	@echo ""
	@echo "  make run           - Ejecutar el agente (forge-user IPC server)"
	@echo "  make chat          - CLI interactiva de chat ⭐ EMPEZAR AQUÍ"
	@echo "  make debug         - Ejecutar con logs DEBUG (forge-user-debug)"

install:
	pip install -e ".[dev]"

dev: install lint

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.egg-info" -delete 2>/dev/null || true
	@echo "✅ Limpieza completada"

lint:
	ruff check src/ forge_core/ tests/
	@echo "✅ Linting completado"

format:
	ruff format src/ forge_core/ tests/
	@echo "✅ Código formateado"

type-check:
	mypy src/ forge_core/ --strict
	@echo "✅ Type checking completado"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov=forge_core --cov-report=html --cov-report=term-missing
	@echo "✅ Coverage report generado en htmlcov/index.html"

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

security:
	@echo "🔒 Ejecutando análisis de seguridad..."
	bandit -r src/ forge_core/ -ll
	safety check --json
	@echo "✅ Análisis de seguridad completado"

run:
	@echo "🚀 Iniciando Orion (HiperForge User)..."
	forge-user

debug:
	@echo "🐛 Iniciando Orion en modo DEBUG..."
	FORGE_USER__OBSERVABILITY__LOG_LEVEL=DEBUG forge-user

chat:
	@echo "💬 Iniciando CLI de chat interactivo..."
	python3 orion_cli.py

# Alias útiles
check: lint type-check
all: clean format lint type-check test security

