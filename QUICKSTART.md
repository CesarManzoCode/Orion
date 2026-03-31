# 🚀 GUÍA RÁPIDA - PROYECTO ORION

> **Estado**: ✅ MVP Completamente Funcional  
> **Versión**: 0.1.0  
> **Última actualización**: 31 de Marzo de 2026

---

## 📋 Tabla de Contenidos
1. [¿Qué es Orion?](#qué-es-orion)
2. [Verificación Rápida](#verificación-rápida)
3. [Instalación](#instalación)
4. [Configuración](#configuración)
5. [Ejecución](#ejecución)
6. [Solución de Problemas](#solución-de-problemas)

---

## ¿Qué es Orion?

Orion es un **agente conversacional de IA personal** que:

- 🧠 **Entiende intenciones** - Clasifica automáticamente qué quiere hacer el usuario
- 📋 **Planifica** - Genera planes de tareas complejas con LLM
- ⚙️ **Ejecuta** - Realiza tareas con seguridad de nivel empresarial
- 💾 **Recuerda** - Guarda historial completo en SQLite local
- 🔒 **Es seguro** - Validación de seguridad en cada paso

---

## ✅ Verificación Rápida

```bash
# Ejecutar script de verificación
cd /home/cesarmanzocode/PycharmProjects/Orion
bash verify.sh
```

Debería ver:
```
✅ README.md
✅ pyproject.toml
✅ .env.example
✅ forge_core/
✅ src/
✅ tests/
```

---

## 💾 Instalación

### Paso 1: Preparar Entorno Virtual

```bash
# Crear entorno virtual (si no existe)
python3 -m venv .venv

# Activar
source .venv/bin/activate
# En Windows: .venv\Scripts\activate
```

### Paso 2: Instalar Dependencias

```bash
# Opción rápida
make install

# O manualmente
pip install -e ".[dev]"
```

Esto instala:
- OpenAI & Anthropic SDKs
- SQLite async
- Logging estructurado
- Testing frameworks
- Tools de desarrollo (ruff, mypy, pytest, etc.)

---

## ⚙️ Configuración

### Opción 1: Usar OpenAI (Recomendado para empezar)

```bash
# Copiar template
cp .env.example .env

# Editar .env y agregar:
FORGE_USER__LLM__DEFAULT_PROVIDER=openai
FORGE_USER__LLM__OPENAI_API_KEY=sk-proj-TU_API_KEY_AQUI
FORGE_USER__LLM__OPENAI_MODEL=gpt-4o
```

**Obtener API key**: https://platform.openai.com/api-keys

### Opción 2: Usar Anthropic (Claude)

```bash
# En .env:
FORGE_USER__LLM__DEFAULT_PROVIDER=anthropic
FORGE_USER__LLM__ANTHROPIC_API_KEY=sk-ant-TU_API_KEY_AQUI
FORGE_USER__LLM__ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

**Obtener API key**: https://console.anthropic.com/

### Opción 3: Local con Ollama (Offline)

```bash
# 1. Instalar Ollama: https://ollama.ai
# 2. Descargar modelo
ollama pull llama2

# 3. En .env:
FORGE_USER__LLM__DEFAULT_PROVIDER=local
FORGE_USER__LLM__LOCAL_MODEL_NAME=llama2

# 4. En otra terminal, mantener Ollama corriendo:
ollama serve
```

---

## 🚀 Ejecución

### Opción 1: Usando Makefile (Recomendado)

```bash
# Ver todos los comandos disponibles
make help

# Ejecutar el agente
make run

# Ejecutar en modo DEBUG (logs detallados)
make debug
```

### Opción 2: Directamente

```bash
# Arrancar
forge-user

# O con logs detallados
FORGE_USER__OBSERVABILITY__LOG_LEVEL=DEBUG forge-user
```

### ✅ Verificar que Funciona

Deberías ver algo como:

```
2026-03-31 10:22:34 [info] hiperforge_user_arrancando
2026-03-31 10:22:34 [info] bootstrap_iniciando
2026-03-31 10:22:34 [info] config_cargada llm_model=gpt-4o
2026-03-31 10:22:34 [info] storage_inicializando
2026-03-31 10:22:37 [info] health_check_completado all_ok=True
2026-03-31 10:22:37 [info] ipc_server_iniciado
2026-03-31 10:22:37 [info] sesion_nueva_creada
{"id": "system", "event": "ready", "session_id": "sess_..."}
```

---

## 🔍 Solución de Problemas

### Error: "No module named 'src'"

```bash
# Asegúrate de estar en el directorio del proyecto
cd /home/cesarmanzocode/PycharmProjects/Orion

# Activar venv
source .venv/bin/activate

# Reinstalar
pip install -e "."
```

### Error: "API key not found"

```bash
# Verificar que .env existe y tiene API key
cat .env | grep OPENAI_API_KEY

# O usar variables de entorno directamente
export FORGE_USER__LLM__OPENAI_API_KEY=sk-proj-...
forge-user
```

### Error: "Port already in use"

El IPC server usa `/tmp/forge_user_ipc.sock`:

```bash
# Limpiar socket viejo
rm /tmp/forge_user_ipc.sock 2>/dev/null || true

# Reintentar
forge-user
```

### Logs muy grandes / Muchos archivos en `.venv`

```bash
# Limpiar cache de Python
make clean

# Limpiar logs viejos
rm -rf ~/.hiperforge_user/logs/*.log.* 2>/dev/null || true
```

---

## 📊 Estructura del Proyecto

```
Orion/
├── 📄 README.md              ← Documentación completa
├── 📄 MVP_STATUS.md          ← Estado detallado del MVP
├── 📄 ASSESSMENT.md          ← Análisis de completitud
├── 📄 .gitignore             ← Filtros de git profesionales
├── 📄 Makefile               ← Comandos útiles
│
├── 🔧 forge_core/            ← Framework reutilizable
│   ├── config/              ← Configuración centralizada
│   ├── llm/                 ← Protocolos LLM
│   ├── policy/              ← Motor de políticas
│   ├── tools/               ← Registry de herramientas
│   └── observability/       ← Logging y tracing
│
├── 🎯 src/                   ← Aplicación Orion
│   ├── bootstrap/           ← Startup y DI container
│   ├── domain/              ← Lógica de negocio
│   ├── application/         ← Coordinadores
│   ├── infrastructure/      ← Adapters concretos
│   └── interfaces/          ← CLI, Desktop, API
│
├── 🧪 tests/                 ← Tests (framework ready)
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── contract/
│   └── security/
│
└── 📚 docs/                  ← Documentación
    ├── architecture.md
    ├── security-model.md
    ├── deployment-guide.md
    └── adr/
```

---

## 🛠️ Comandos Frecuentes

```bash
# INSTALACIÓN Y SETUP
make install               # Instalar todas las dependencias
make dev                   # Setup para desarrollo
make clean                 # Limpiar cache y temporales

# DESARROLLO
make format                # Formatear código con ruff
make lint                  # Linting
make type-check            # Type checking con mypy
make check                 # lint + type-check

# TESTING
make test                  # Todos los tests
make test-cov              # Con reporte de cobertura
make test-unit             # Solo tests unitarios
make test-integration      # Solo tests de integración
make test-e2e              # Solo tests end-to-end

# SEGURIDAD
make security              # Análisis de seguridad (bandit + safety)

# EJECUCIÓN
make run                   # Ejecutar agente
make debug                 # Ejecutar en modo DEBUG

# UTILIDADES
make help                  # Ver todos los comandos
make all                   # Limpieza + format + lint + type-check + test + security
```

---

## 📈 Próximos Pasos

### Corto Plazo (Esta semana)
- [ ] Explorar logs en modo DEBUG
- [ ] Entender estructura del código
- [ ] Ejecutar tests
- [ ] Leer [MVP_STATUS.md](MVP_STATUS.md)

### Mediano Plazo (Esta/próxima semana)
- [ ] Implementar CLI completa
- [ ] Agregar tests unitarios
- [ ] Explorar PolicyEngine
- [ ] Entender LLM adapters

### Largo Plazo (Próximas semanas)
- [ ] Desktop app (Tauri)
- [ ] Full streaming
- [ ] Vision/multimodal
- [ ] Advanced planning

---

## 📚 Documentación Adicional

| Archivo | Contenido |
|---------|----------|
| [README.md](README.md) | Documentación completa del proyecto |
| [MVP_STATUS.md](MVP_STATUS.md) | Estado detallado del MVP |
| [ASSESSMENT.md](ASSESSMENT.md) | Análisis de completitud |
| [docs/architecture.md](docs/architecture.md) | Diseño de alto nivel |
| [docs/security-model.md](docs/security-model.md) | Modelo de seguridad |
| [docs/adr/](docs/adr/) | Architecture Decision Records |

---

## 🆘 Soporte

### Comandos útiles para debugging

```bash
# Ver logs en tiempo real
FORGE_USER__OBSERVABILITY__LOG_LEVEL=DEBUG forge-user

# Verificar configuración
python3 -c "from src.bootstrap.container import build_container; print('OK')"

# Revisar dependencias
pip freeze | head -20

# Verificar imports
python3 -c "from src.bootstrap.startup import main; print('OK')"
```

### Lugares para buscar ayuda

1. Logs en `~/.hiperforge_user/logs/`
2. Documentación en `docs/`
3. Status en `MVP_STATUS.md`
4. Assessment en `ASSESSMENT.md`

---

## ✅ Checklist Inicial

- [ ] Proyecto clonado/ubicado en `/home/cesarmanzocode/PycharmProjects/Orion`
- [ ] Virtual environment creado y activado (`.venv`)
- [ ] Dependencias instaladas (`make install`)
- [ ] `.env` configurado con API key
- [ ] Proyecto arranca sin errores (`make run`)
- [ ] Logs muestran "health_check_completado all_ok=True"
- [ ] IPC server inicializado ("ipc_server_iniciado")

---

**¿Listo? ¡Adelante con `make run`! 🚀**

