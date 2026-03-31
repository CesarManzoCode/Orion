#!/usr/bin/env python3
"""
CLI Interactiva para Orion - Agente Conversacional
Permite chatear con el agente usando la API de Python
"""

import asyncio
import json
from pathlib import Path

# Agregar rutas del proyecto
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.bootstrap.container import build_container
from src.domain.value_objects.identifiers import UserId, SessionId
from src.ports.inbound.conversation_port import UserMessageRequest


class OrionCLI:
    """CLI interactiva para Orion"""

    def __init__(self):
        self.container = None
        self.coordinator = None
        self.session = None
        self.session_id_str = None
        # Generar o crear un UserId válido
        self.user_id = UserId.generate()

    async def initialize(self):
        """Inicializar el contenedor y coordinador"""
        print("\n🚀 Inicializando Orion...")
        try:
            self.container = await build_container()
            self.coordinator = self.container.conversation_port
            
            # Obtener o crear sesión
            # NOTA: Siempre creamos una nueva sesión para evitar el bug conocido
            # de incompatibilidad en Session.restore() con turn_count
            session_manager = self.container.session_manager
            
            try:
                session = await session_manager.create_session(user_id=self.user_id)
                session_id = session.session_id.to_str() if hasattr(session.session_id, 'to_str') else str(session.session_id)
                self.session = session
                self.session_id_str = session_id
                print(f"✅ Nueva sesión creada: {session_id}")
            except Exception as e:
                print(f"❌ Error creando sesión: {e}")
                raise
            
            print("✅ Orion listo!")
            print("\n" + "="*60)
            print("💬 CLI INTERACTIVO DE ORION")
            print("="*60)
            print("Escribe 'salir' o 'exit' para terminar")
            print("Escribe 'help' para ver comandos")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Error inicializando: {e}")
            raise

    async def handle_message(self, user_input: str) -> str:
        """Procesar mensaje del usuario usando la API correcta con manejo robusto de errores"""
        if not user_input.strip():
            return ""
        
        try:
            # Crear el request con la API correcta
            request = UserMessageRequest(
                session_id=SessionId.from_str(self.session_id_str),
                message=user_input,
                attachments=[]
            )
            
            # Enviar mensaje usando send_message (la API correcta)
            response = await self.coordinator.send_message(request)
            
            # Retornar el contenido de texto de la respuesta
            if response.text_content:
                return response.text_content
            elif response.response_type.name == 'ERROR':
                return "Lo siento, hubo un error procesando tu mensaje. Por favor intenta de nuevo."
            else:
                return "Sin respuesta"
            
        except Exception as e:
            error_msg = str(e)
            # Errores de storage conocidos
            if "turn_count" in error_msg or "STORAGE_ERROR" in error_msg:
                return "⚠️ Error temporal del almacenamiento. Por favor intenta de nuevo."
            elif "session" in error_msg.lower():
                return "⚠️ Error con la sesión. Por favor intenta de nuevo."
            else:
                # Log del error real pero mostrar mensaje amigable
                import traceback
                traceback.print_exc()
                return f"❌ Error inesperado. Intenta de nuevo."

    async def run(self):
        """Ejecutar CLI interactiva"""
        await self.initialize()
        
        try:
            while True:
                try:
                    user_input = input("🤖 Tú: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Comandos especiales
                    if user_input.lower() in ['salir', 'exit', 'quit']:
                        print("\n👋 Hasta luego!")
                        break
                    
                    if user_input.lower() == 'help':
                        self.show_help()
                        continue
                    
                    if user_input.lower() == 'info':
                        await self.show_info()
                        continue
                    
                    # Procesar mensaje
                    print("\n⏳ Procesando...\n")
                    response = await self.handle_message(user_input)
                    print(f"🤖 Orion: {response}\n")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Hasta luego!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
        
        except Exception as e:
            print(f"❌ Error en CLI: {e}")

    def show_help(self):
        """Mostrar ayuda de comandos"""
        print("\n📖 COMANDOS DISPONIBLES:")
        print("  help        - Mostrar esta ayuda")
        print("  info        - Información de la sesión")
        print("  salir/exit  - Terminar el chat")
        print("\n💡 EJEMPLOS DE PREGUNTAS:")
        print("  - ¿Cuál es el clima en Madrid?")
        print("  - Lista los archivos en ~/Documents")
        print("  - ¿Qué hora es?")
        print("  - Ayúdame a entender DDD")
        print()

    async def show_info(self):
        """Mostrar información de sesión"""
        print("\n📊 INFORMACIÓN DE SESIÓN:")
        print(f"  Session ID: {self.session_id_str}")
        print(f"  User ID: {self.user_id.to_str()}")
        if hasattr(self.session, 'turn_count'):
            print(f"  Turnos: {self.session.turn_count}")
        print()


async def main():
    """Punto de entrada principal"""
    cli = OrionCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())

