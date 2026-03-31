# 🚀 Orion - Agente Personal Conversacional (HiperForge User)

> Un agente de IA conversacional para desktop que puede planificar, razonar y ejecutar tareas locales con seguridad de nivel empresarial.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: MVP](https://img.shields.io/badge/Status-MVP-brightgreen)](MVP_STATUS.md)

---

## ✨ Características

### 🧠 Inteligencia
- **Intent Routing** - Clasifica automáticamente intenciones del usuario (respuesta directa, uso de herramientas, planificación)
- **Planning** - Genera planes secuenciales con LLM (máx 5 tareas)
- **Execution** - Ejecuta tareas con validación de seguridad en tiempo real
- **Memory** - Gestiona contexto conversacional con límites de tokens

### 🔒 Seguridad
- **Policy Engine** - Motor de políticas con evaluación de cada acción
- **Allowlist** - Solo ejecuta herramientas autorizadas
- **Sandbox** - Ejecución en procesos aislados
- **Rate Limiting** - Límites de invocaciones por minuto
- **Audit Log** - Registro completo de todas las acciones

### 💾 Persistencia
- **SQLite Local** - Base de datos completamente local
- **Conversación** - Historial completo con turnos
- **Sessions** - Sesiones múltiples por usuario
- **Audit Trail** - Log de seguridad con timestamps

### 🎛️ Configuración
- **Variables de Entorno** - Todo configurable via `.env`
- **Multi-Provider** - Soporta OpenAI, Anthropic, Ollama (local)
- **Flexible** - Modo offline con Ollama, modo cloud con OpenAI/Anthropic

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────┐
│           USER / CLI / DESKTOP UI                   │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│         Conversation Coordinator                    │
│  (Orquestación del flujo conversacional)            │
└─────────────────┬───────────────────────────────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│  Intent  │ │Planner   │ │ Task     │
│  Router  │ │(LLM)     │ │ Executor │
└─────┬────┘ └────┬─────┘ └────┬─────┘
      │           │            │
      └───────────┼────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│        Policy Engine & Tool Registry                │
│  (Validación de seguridad & autorización)           │
└─────────────────┬───────────────────────────────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ LLM      │ │ Storage  │ │ Tools    │
│ Adapter  │ │(SQLite)  │ │(Registry)│
└──────────┘ └──────────┘ └──────────┘
```

### Capas DDD
- **Domain**: Entidades, agregados, políticas (reglas de negocio)
- **Application**: Coordinadores, orquestadores, casos de uso
- **Infrastructure**: SQLite, LLM adapters, logging, observabilidad
- **Interfaces**: CLI, Desktop (IPC), API

---

## 🚀 Quick Start

### 1. Requisitos
- Python 3.11+
- pip/pip3
- (Opcional) Docker para Ollama si usas modelos locales

### 2. Instalación

```bash
# Clonar repositorio
git clone <repo>
cd Orion

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -e ".[dev]"
```

### 3. Configuración

```bash
# Copiar template de configuración
cp .env.example .env

# Editar .env y agregar credenciales
# Para OpenAI:
export FORGE_USER__LLM__OPENAI_API_KEY=sk-proj-...
export FORGE_USER__LLM__DEFAULT_PROVIDER=openai

# O para Anthropic:
export FORGE_USER__LLM__ANTHROPIC_API_KEY=sk-ant-...
export FORGE_USER__LLM__DEFAULT_PROVIDER=anthropic

# O para local (Ollama):
# 1. Instalar Ollama: https://ollama.ai
# 2. ollama pull llama2
# 3. export FORGE_USER__LLM__DEFAULT_PROVIDER=local
```

### 4. Ejecutar

```bash
# Arrancar el agente
forge-user

# En otra terminal, interactuar con CLI (cuando esté implementada)
# O conectarse via IPC desde Tauri desktop app
```

---

## 📖 Uso

### Vía IPC (Desktop/Tauri)

```python
import json
import socket

# Conectarse al servidor IPC
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/forge_user_ipc.sock")

# Enviar mensaje
message = {
    "type": "chat",
    "session_id": "sess_...",
    "content": "¿Cuál es el clima en Madrid?"
}
sock.send(json.dumps(message).encode())

# Recibir respuesta
response = json.loads(sock.recv(4096).decode())
print(response["content"])
```

### Vía Python API

```python
from src.bootstrap.container import build_container

# Construir contenedor
container = build_container()

# Usar coordinador
coordinator = container.conversation_coordinator
response = await coordinator.handle_message(
    user_id="user_123",
    session_id="sess_456",
    content="Listar archivos en ~/Documents"
)
print(response.content)
```

---

## 🛠️ Desarrollo

### Estructura de Directorios

```
Orion/
├── forge_core/              # Core framework (reutilizable)
│   ├── config/             # Configuración centralizada
│   ├── llm/                # Protocolos y tipos LLM
│   ├── policy/             # Motor de políticas
│   ├── tools/              # Registry y tipos de herramientas
│   ├── observability/      # Logging, tracing, métricas
│   └── testing/            # Fixtures y utilities de test
├── src/                     # Aplicación Orion
│   ├── bootstrap/          # Startup y DI container
│   ├── domain/             # Lógica de negocio (DDD)
│   ├── application/        # Coordinadores y casos de uso
│   ├── infrastructure/     # Adapters concretos
│   ├── interfaces/         # CLI, Desktop, API
│   └── ports/             # Interfaces de dominio
├── tests/                   # Suite de tests
│   ├── unit/               # Tests unitarios
│   ├── integration/        # Tests de integración
│   ├── e2e/                # Tests end-to-end
│   ├── contract/           # Contract testing
│   └── security/           # Pruebas de seguridad
├── docs/                    # Documentación
└── scripts/                 # Build scripts

```

### Ejecutar Tests

```bash
# Suite completa
pytest tests/ -v

# Con cobertura
pytest tests/ -v --cov=src --cov=forge_core

# Tests específicos
pytest tests/unit/ -v -k "intent"
pytest tests/e2e/conversation_flow.py -v

# Con markers
pytest -m "not slow" -v
pytest -m "security" -v
```

### Linting y Formateo

```bash
# Formatear código
ruff format src/ forge_core/ tests/

# Linting
ruff check src/ forge_core/ tests/

# Type checking
mypy src/ forge_core/

# Security scanning
bandit -r src/ forge_core/

# Verificar dependencias
safety check
```

### Debugging

```bash
# Logs detallados
FORGE_USER__OBSERVABILITY__LOG_LEVEL=DEBUG forge-user

# Debug CLI
forge-user-debug

# Con icecream (debug prints mejorados)
python -c "from icecream import ic; ic.enable()"
```

---

## 📋 Estado MVP

El proyecto está **100% funcional como MVP** con:

✅ **Completado**
- Intent Router (clasificación de intenciones)
- Task Executor (ejecución con seguridad)
- Lightweight Planner (generación de planes)
- Storage (SQLite con 5 stores)
- Policy Engine (validación de seguridad)
- LLM Adapters (OpenAI, Anthropic, Ollama)
- Configuration (exhaustiva via .env)
- Logging (structured logging)
- IPC Server (para desktop)
- Health Checks

⏳ **Post-MVP** (Próximas versiones)
- CLI completa (`src/interfaces/cli/`)
- Desktop UI (Tauri IPC integration)
- Full streaming
- Approval workflows interactivos
- Vision/multimodal
- Web search (Tavily)
- File processing (PDF, DOCX)
- Advanced planning (STRIPS/HTN)

Ver [MVP_STATUS.md](MVP_STATUS.md) para detalles completos.

---

## 🔐 Seguridad

### Políticas Implementadas
1. **File System** - Solo directorios autorizados
2. **App Execution** - Solo aplicaciones en allowlist
3. **Network** - Solo dominios permitidos
4. **Rate Limiting** - 60 tool calls/min, 30 web requests/min
5. **Token Budget** - Límite de contexto por sesión

### Sandboxing
- Procesos aislados para ejecución de tareas
- Timeout de 30s por tarea
- Contenedor de errores robusto

### Audit Trail
- Todas las acciones registradas
- Timestamps y user IDs
- Severidad de cada operación

---

## 📊 Configuración

Todas las variables de configuración están en `.env` o variables de entorno con prefijo `FORGE_USER__`:

### Principales
```bash
# LLM
FORGE_USER__LLM__DEFAULT_PROVIDER=openai|anthropic|local
FORGE_USER__LLM__OPENAI_API_KEY=sk-...
FORGE_USER__LLM__ANTHROPIC_API_KEY=sk-ant-...

# Storage
FORGE_USER__STORAGE__DATA_DIR=~/.hiperforge_user/data
FORGE_USER__STORAGE__DATABASE_FILENAME=forge_user.db

# Seguridad
FORGE_USER__SECURITY__ALLOWED_DIRECTORIES=/home:/home/user/Documents
FORGE_USER__SECURITY__MAX_FILE_SIZE_MB=50
FORGE_USER__SECURITY__SANDBOX_ENABLED=true

# Observabilidad
FORGE_USER__OBSERVABILITY__LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
FORGE_USER__OBSERVABILITY__LOG_FORMAT=console|json
```

Ver `.env.example` para lista completa (100+ variables).

---

## 📚 Documentación

- [Arquitectura](docs/architecture.md) - Diseño de alto nivel y decisiones
- [ADRs](docs/adr/) - Architecture Decision Records (5 decisiones documentadas)
- [Security Model](docs/security-model.md) - Modelo de seguridad en detalle
- [Deployment Guide](docs/deployment-guide.md) - Cómo desplegar
- [Tool Development](docs/tool-development-guide.md) - Cómo crear herramientas

---

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/amazing-feature`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

### Estándares de Código
- Type hints en todas las funciones
- Docstrings en formato Google
- Tests para nuevas features
- Linting con ruff
- Type checking con mypy

---

## 📄 Licencia

Este proyecto está bajo licencia MIT - ver archivo [LICENSE](LICENSE) para detalles.

---

## 🙋 Soporte

- 📖 Documentación: [docs/](docs/)
- 🐛 Issues: [GitHub Issues](issues)
- 💬 Discussions: [GitHub Discussions](discussions)
- 📝 Status: [MVP_STATUS.md](MVP_STATUS.md)

---

## 🎯 Roadmap

### v0.2.0 (Q2 2026)
- [ ] CLI completa
- [ ] Tauri desktop app básica
- [ ] Full streaming support
- [ ] Approval workflows interactivos

### v0.3.0 (Q3 2026)
- [ ] Vision/multimodal
- [ ] Web search integration
- [ ] File processing
- [ ] Advanced planning (STRIPS)

### v1.0.0 (Q4 2026)
- [ ] Producción-ready
- [ ] Plugin system
- [ ] Analytics dashboard
- [ ] Cloud sync

---

**Hecho con ❤️ para builders que quieren agentes seguros y locales**

