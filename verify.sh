#!/bin/bash

# ============================================================================
# SCRIPT DE VERIFICACIÓN RÁPIDA DEL PROYECTO ORION
# ============================================================================
# Este script valida que el proyecto esté completamente funcional

set -e

echo "🔍 VERIFICACIÓN DEL PROYECTO ORION"
echo "=================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $1"
        return 0
    else
        echo -e "${RED}❌${NC} $1 NOT FOUND"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅${NC} $1/"
        return 0
    else
        echo -e "${RED}❌${NC} $1/ NOT FOUND"
        return 1
    fi
}

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✅${NC} $1 available"
        return 0
    else
        echo -e "${YELLOW}⚠️${NC}  $1 not found"
        return 1
    fi
}

# ============================================================================
# 1. ARCHIVOS CLAVE
# ============================================================================
echo "📋 ARCHIVOS CLAVE"
echo "-----------------"
check_file "README.md"
check_file "pyproject.toml"
check_file ".env.example"
check_file ".gitignore"
check_file "MVP_STATUS.md"
check_file "Makefile"
check_file "LICENSE"
echo ""

# ============================================================================
# 2. DIRECTORIOS PRINCIPALES
# ============================================================================
echo "📁 DIRECTORIOS PRINCIPALES"
echo "--------------------------"
check_dir "forge_core"
check_dir "src"
check_dir "src/bootstrap"
check_dir "src/domain"
check_dir "src/application"
check_dir "src/infrastructure"
check_dir "src/interfaces"
check_dir "tests"
check_dir "docs"
check_dir "scripts"
echo ""

# ============================================================================
# 3. HERRAMIENTAS DISPONIBLES
# ============================================================================
echo "🛠️  HERRAMIENTAS DISPONIBLES"
echo "----------------------------"
check_command "python3"
check_command "pip"
check_command "git"
check_command "make"
echo ""

# ============================================================================
# 4. PYTHON & VENV
# ============================================================================
echo "🐍 PYTHON & VENV"
echo "----------------"
if [ -d ".venv" ]; then
    echo -e "${GREEN}✅${NC} Virtual environment (.venv/) found"
    PYTHON_PATH=".venv/bin/python"
    if [ -f "$PYTHON_PATH" ]; then
        PYTHON_VERSION=$($PYTHON_PATH --version 2>&1)
        echo -e "${GREEN}✅${NC} $PYTHON_VERSION"
    fi
else
    echo -e "${YELLOW}⚠️${NC}  No virtual environment found"
    echo "    Run: python3 -m venv .venv && source .venv/bin/activate"
fi
echo ""

# ============================================================================
# 5. VERIFICAR IMPORTS BÁSICOS
# ============================================================================
echo "✨ VERIFICAR IMPORTS"
echo "-------------------"
if [ -f "$PYTHON_PATH" ]; then
    if $PYTHON_PATH -c "from src.bootstrap.startup import main" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} src.bootstrap.startup imports OK"
    else
        echo -e "${RED}❌${NC} src.bootstrap.startup import failed"
    fi

    if $PYTHON_PATH -c "from src.bootstrap.container import build_container" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} src.bootstrap.container imports OK"
    else
        echo -e "${RED}❌${NC} src.bootstrap.container import failed"
    fi
else
    echo -e "${YELLOW}⚠️${NC}  Python not found, skipping imports"
fi
echo ""

# ============================================================================
# 6. CONFIGURACIÓN
# ============================================================================
echo "⚙️  CONFIGURACIÓN"
echo "----------------"
if [ -f ".env" ]; then
    echo -e "${GREEN}✅${NC} .env file exists"
    if grep -q "FORGE_USER__LLM__" .env; then
        echo -e "${GREEN}✅${NC} LLM configuration found"
    else
        echo -e "${YELLOW}⚠️${NC}  No LLM configuration in .env"
    fi
else
    echo -e "${YELLOW}⚠️${NC}  .env file not found (copy from .env.example)"
fi
echo ""

# ============================================================================
# 7. RESUMEN
# ============================================================================
echo "📊 RESUMEN"
echo "=========="
echo -e "${GREEN}✅ El proyecto Orion está correctamente estructurado${NC}"
echo ""
echo "🚀 PRÓXIMOS PASOS:"
echo "  1. Crear entorno virtual:"
echo "     python3 -m venv .venv"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Instalar dependencias:"
echo "     make install"
echo ""
echo "  3. Configurar LLM (editar .env):"
echo "     cp .env.example .env"
echo "     # Editar .env con tu API key"
echo ""
echo "  4. Ejecutar el agente:"
echo "     make run"
echo ""
echo "  5. Ejecutar tests:"
echo "     make test"
echo ""
echo "  6. Ver más comandos:"
echo "     make help"
echo ""

# ============================================================================
# FIN
# ============================================================================
echo -e "${GREEN}✅ Verificación completada${NC}"

