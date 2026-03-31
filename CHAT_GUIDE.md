# 💬 CÓMO INTERACTUAR CON ORION

> **Guía rápida**: Hay dos formas de interactuar con Orion

---

## 🚀 OPCIÓN 1: CLI Interactiva (Recomendado para Testing)

### ¿Qué es?
Una interfaz de línea de comandos simple que permite chatear directamente con el agente Orion.

### ¿Cómo usarla?

```bash
# Ejecutar la CLI interactiva
make chat

# O directamente
python3 orion_cli.py
```

### Ejemplo de Sesión

```
🚀 Inicializando Orion...
✅ Nueva sesión creada: sess_01KN2BNWMM02324G93A9GR8JYA
✅ Orion listo!

============================================================
💬 CLI INTERACTIVO DE ORION
============================================================
Escribe 'salir' o 'exit' para terminar
Escribe 'help' para ver comandos
============================================================

🤖 Tú: ¿Cuál es tu nombre?

⏳ Procesando...

🤖 Orion: Soy Orion, un agente conversacional personal...

🤖 Tú: Listar archivos en el directorio actual

⏳ Procesando...

🤖 Orion: Aquí están los archivos en el directorio actual...

🤖 Tú: help

📖 COMANDOS DISPONIBLES:
  help        - Mostrar esta ayuda
  info        - Información de la sesión
  salir/exit  - Terminar el chat

🤖 Tú: salir

👋 Hasta luego!
```

### Comandos Disponibles

| Comando | Descripción |
|---------|-------------|
| `help` | Mostrar ayuda |
| `info` | Información de sesión |
| `salir` / `exit` | Terminar |

---

## 🔌 OPCIÓN 2: IPC Server (Para Desktop/Tauri)

### ¿Qué es?
Un servidor que escucha en un socket UNIX y recibe mensajes JSON desde aplicaciones desktop (como Tauri).

### ¿Cómo usarlo?

```bash
# Ejecutar el servidor IPC
make run

# O con logs detallados
make debug
```

### Estructura de Mensajes JSON

```python
# Enviar mensaje
{
    "type": "chat",
    "session_id": "sess_01KN2BNWMM02324G93A9GR8JYA",
    "content": "¿Cuál es el clima?"
}

# Respuesta
{
    "id": "user_01",
    "event": "message_response",
    "session_id": "sess_01KN2BNWMM02324G93A9GR8JYA",
    "content": "El clima es...",
    "status": "success"
}
```

### Ejemplo de Conexión (Python)

```python
import json
import socket

# Conectar al servidor IPC
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/forge_user_ipc.sock")

# Enviar mensaje
message = {
    "type": "chat",
    "session_id": "sess_123",
    "content": "¿Hola?"
}
sock.send(json.dumps(message).encode())

# Recibir respuesta
response = json.loads(sock.recv(4096).decode())
print(response["content"])
```

---

## 📊 COMPARATIVA

| Característica | CLI Interactiva | IPC Server |
|---|---|---|
| **Setup** | `make chat` | `make run` |
| **Interfaz** | Terminal | Socket UNIX |
| **Cliente** | Built-in | External (Tauri, etc) |
| **Para Testing** | ✅ Ideal | ⏳ Más complejo |
| **Para Desktop** | ⏳ Limitado | ✅ Ideal |
| **Desarrollo** | ✅ Perfecto | ✅ Bueno |

---

## 🎯 RECOMENDACIONES

### Empezar Rápido
```bash
# 1. Instalar
make install

# 2. Configurar (editar .env)
cp .env.example .env

# 3. Chatear
make chat

# Escribe: "¿Hola, quién eres?"
```

### Entender la Arquitectura
```bash
# Ver cómo funciona el servidor IPC
make run

# En otra terminal, ver logs
make debug
```

### Desarrollar Integración
```bash
# Mantener el servidor corriendo
make run

# Conectar cliente externo (Tauri, etc)
# Ver ejemplo de socket arriba
```

---

## ❓ PREGUNTAS FRECUENTES

### "¿Dónde está el chat?"
Hay dos opciones:
- **CLI**: `make chat` (para testing rápido)
- **IPC Server**: `make run` (para apps desktop)

### "¿Por qué JSON?"
Porque el servidor IPC es agnóstico a la plataforma. Cualquier aplicación (web, desktop, mobile) puede conectarse si habla JSON.

### "¿Puedo usar esto desde Tauri?"
Sí, el IPC server está diseñado exactamente para eso. Ver ejemplo de socket arriba.

### "¿Cuántas sesiones puedo tener?"
Ilimitadas. Cada usuario puede tener múltiples sesiones paralelas.

### "¿Se guardan los chats?"
Sí, todo se persiste en SQLite en `~/.hiperforge_user/data/forge_user.db`

---

## 🔧 TROUBLESHOOTING

### Error: "Socket already in use"
```bash
# Limpiar socket viejo
rm /tmp/forge_user_ipc.sock 2>/dev/null || true
make run
```

### Error: "JSON inválido"
Si escribes texto plano en el IPC server, recibirás este error. Usa:
- `make chat` para interfaz de texto simple
- JSON válido si conectas directo al socket

### Error: "ModuleNotFoundError"
```bash
# Reinstalar
make install
```

---

## 📚 Próximos Pasos

1. ✅ Prueba `make chat`
2. Explora los comandos `help` e `info`
3. Lee la documentación en [README.md](README.md)
4. Integra con Tauri si necesitas desktop

---

**¡Listo para chatear! Prueba `make chat` ahora 🚀**

