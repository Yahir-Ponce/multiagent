import os
import asyncio
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.core_plugins.text_plugin import TextPlugin
from semantic_kernel.core_plugins.math_plugin import MathPlugin

async def main():
    # 1. Cargar variables de entorno
    load_dotenv()

    # 2. Crear el kernel y agregar plugins
    kernel = Kernel()
    
    # Agregar el servicio de chat al kernel
    kernel.add_service(AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21-preview")
    ))
    
    # Agregar plugins al kernel
    math_plugin = kernel.add_plugin(MathPlugin(), plugin_name="math")
    text_plugin = kernel.add_plugin(TextPlugin(), plugin_name="text")

    # 3. Definir los agentes especializados
    billing_agent = ChatCompletionAgent(
        name="BillingAgent",
        instructions="""
        Eres un experto en facturación: facturas, cargos, tarifas, métodos de pago, ciclos de facturación, problemas de pago.
        Usa el plugin de matemáticas cuando el usuario solicite cálculos.
        Responde siempre en español de manera clara y profesional.
        """,
        kernel=kernel,
    )

    refund_agent = ChatCompletionAgent(
        name="RefundAgent",
        instructions="""
        Eres un experto en reembolsos: elegibilidad, políticas, procesamiento, actualizaciones de estado, quejas.
        Usa el plugin de texto cuando el usuario pida un resumen de la conversación.
        Responde siempre en español de manera clara y profesional.
        """,
        kernel=kernel
    )

    tech_agent = ChatCompletionAgent(
        name="TechSupportAgent",
        instructions="""
        Eres un experto en problemas técnicos: errores de inicio de sesión, bugs, instalación, conectividad, rendimiento.
        Responde siempre en español de manera clara y profesional.
        """,
        kernel=kernel
    )

    # 4. Crear un sistema de enrutamiento simple
    async def route_query(user_query: str):
        """
        Función para determinar qué agente debe manejar la consulta
        """
        query_lower = user_query.lower()
        
        # Palabras clave para cada agente
        billing_keywords = ["factur", "cobr", "pag", "tarif", "ciclo", "cargo", "costo", "precio"]
        refund_keywords = ["reembol", "devol", "cancelar", "anular", "política"]
        tech_keywords = ["error", "bug", "login", "conectar", "instalar", "técnico", "problema"]
        
        # Verificar palabras clave
        if any(keyword in query_lower for keyword in refund_keywords):
            return refund_agent
        elif any(keyword in query_lower for keyword in billing_keywords):
            return billing_agent
        elif any(keyword in query_lower for keyword in tech_keywords):
            return tech_agent
        else:
            # Por defecto, usar el agente de soporte técnico
            return tech_agent

    # 5. Procesar la consulta del usuario
    user_query = "Quiero pedir un reembolso de mi suscripción anual"
    
    # Determinar el agente apropiado
    selected_agent = await route_query(user_query)
    print(f"Consulta enrutada al agente: {selected_agent.name}")
    
    # Obtener respuesta del agente seleccionado
    async for response in selected_agent.invoke(user_query):
        print(f"Respuesta de {selected_agent.name}: {response.content}")

    # 6. Ejemplo de conversación interactiva (opcional)
    print("\n--- Modo conversación interactiva ---")
    print("Escribe 'salir' para terminar")
    
    while True:
        user_input = input("\nTu consulta: ")
        if user_input.lower() in ['salir', 'exit', 'quit']:
            break
        
        selected_agent = await route_query(user_input)
        print(f"Enrutado a: {selected_agent.name}")
        
        async for response in selected_agent.invoke(user_input):
            print(f"{selected_agent.name}: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())