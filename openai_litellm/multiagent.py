from dotenv import load_dotenv
import os
from agents import Agent, Runner, set_tracing_disabled, function_tool, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel

load_dotenv()
set_tracing_disabled(True)

API_KEY    = os.getenv("AZURE_API_KEY")
models = {
    "gpt-4o-mini": "Innovation-gpt4o-mini",
    "gpt-4o": "Innovation-gpt4o",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "gpt-o1": "o1",
    "gpt-o3": "o3-mini",
    "gpt-o4-mini": "o4-mini",
    "gpt-5": "gpt-5-chat"
}

@function_tool
def get_weather(city: str, unit: str = "C") -> str:
    """
    Devuelve clima falso (demo).
    """
    fake = {
        "Monterrey": {"C": "32°C, soleado", "F": "89.6°F, soleado"},
        "CDMX": {"C": "22°C, nublado", "F": "71.6°F, nublado"},
        "Madrid": {"C": "25°C, claro", "F": "77°F, claro"},
    }
    data = fake.get(city, {"C": "N/D", "F": "N/D"}).get(unit, "N/D")
    return f"Clima en {city}: {data}"

spanish_agent = Agent(
    name="Spanish agent",
    instructions=(
        "Responde en español, breve y directo. "
        "Si el usuario pide clima, llama a la tool get_weather."
    ),
    model=LitellmModel(
        model=f"azure/{models['gpt-4o']}",
        api_key=API_KEY,
        
    ),
    tools=[get_weather],
    model_settings=ModelSettings(include_usage=True, parallel_tool_calls=True),
)

english_agent = Agent(
    name="English Assistant",
    instructions=(
        "Reply in concise English. "
        "If the user requests weather, call get_weather."
    ),
    model=LitellmModel(
        model=f"azure/{models['gpt-4o']}",
        api_key=API_KEY,
        
    ),
    tools=[get_weather],
    model_settings=ModelSettings(include_usage=True, parallel_tool_calls=True),
)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Decide el handoff: si el mensaje está mayormente en español -> Spanish agent; "
        "si está en inglés -> English Assistant. Si dudas, responde en el idioma detectado. "
        "No expliques la decisión, solo responde o delega."
    ),
    tools=[get_weather],
    model=f"litellm/azure/{models['gpt-4o']}",
    handoffs=[spanish_agent, english_agent],
    model_settings=ModelSettings(include_usage=True),
)

if __name__ == "__main__":
    # --- 1) Handoff automático por idioma ---
    r1 = Runner.run_sync(triage_agent, input="Puedes darme el clima de CDMX en ingles ??")
    print("\n[TRIAGE -> AUTO] (español):")
    print(r1.final_output)

    r2 = Runner.run_sync(triage_agent, input="What's the weather in Madrid in F?")
    print("\n[TRIAGE -> AUTO] (english):")
    print(r2.final_output)

    # --- 2) Llamada directa a un solo agente ---
    r3 = Runner.run_sync(spanish_agent, input="¿Me dices el clima de Monterrey en C?")
    print("\n[DIRECT -> Spanish agent]:")
    print(r3.final_output)

    # --- 3) Tool explícita (demostración) ---
    r4 = Runner.run_sync(english_agent, input="Get me the weather for CDMX in C.")
    print("\n[DIRECT -> English Assistant + tool]:")
    print(r4.final_output)
