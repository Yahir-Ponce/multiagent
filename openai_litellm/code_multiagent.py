from dotenv import load_dotenv
import os, requests
from bs4 import BeautifulSoup
from agents import Agent, Runner, set_tracing_disabled, function_tool, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel

# --- Cargar variables de entorno ---
load_dotenv()
set_tracing_disabled(True)   # Desactiva el tracing (útil para debug/logs internos de agents)

API_KEY = os.getenv("AZURE_API_KEY")


# --- function_tool : fetch_url -> scrappea una URL usando BS4 ---
@function_tool
def fetch_url(url: str) -> str:
    """
    Scrapea el contenido de una URL y devuelve solo el texto plano.
    
    Args:
        url (str): URL proporcionada por el usuario
    
    Returns:
        str: Texto plano de la página (máx. 8000 caracteres)
    """
    try:
        print("[Looking for info]")
        r = requests.get(url, timeout=15)    # Timeout de 15s para evitar colgarse
        r.raise_for_status()                 # Lanza excepción si status != 200
        soup = BeautifulSoup(r.text, "html.parser")

        # Elimina etiquetas irrelevantes para lectura
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extrae texto limpio
        text = soup.get_text(separator=" ", strip=True)
        return text[:8000]  # Limita a 8k caracteres (para no saturar al modelo)
    except Exception as e:
        return f"Error fetching {url}: {e}"


# --- Agent : code_agent -> generador de código ---
code_agent = Agent(
    name="Engineer",
    instructions=(
        "You are a coding assistant. If the user asks for code, generate correct and concise examples. "
        "Use best practices for clarity."
    ),
    model=LitellmModel(
        model="azure/gpt-4.1",   # Modelo usado (deployment en Azure)
        api_key=API_KEY,
    ),
    model_settings=ModelSettings(include_usage=True),  # Incluye métricas de uso en la salida
)


# --- Agent : doc_explainer -> que explica documentación ---
doc_explainer = Agent(
    name="Explorer",
    instructions=(
        "You are a helpful assistant that explains documentation in simple English. "
        "If the user provides a URL, call fetch_doc to get the content and then summarize or explain it clearly."
        "If the user needs "
    ),
    model=LitellmModel(
        model="azure/gpt-4.1",
        api_key=API_KEY,
    ),
    tools=[fetch_url],   # Usa fetch_url para obtener contenido de páginas
    model_settings=ModelSettings(
        include_usage=True,
        parallel_tool_calls=True   # Permite llamadas paralelas a herramientas
    ),
    handoffs=[code_agent]  # Si el user pide código además de explicación, pasa el control al code_agent
)


# --- Agent : doc_explainer decide a quién mandar la petición ---
triage_agent = Agent(
    name="Orchestrator",
    instructions = (
        "If the user provides a URL, handoff to the Documentation Explainer. "
        "If the user explicitly asks for code, and also gave a URL, "
        "handoff first to Documentation Explainer, then to Code Generator. "
        "If the user just wants code without URL, handoff to Code Generator directly."
    ),
    model = LitellmModel(
        model="azure/gpt-5",
        api_key=API_KEY,
    ),
    handoffs = [doc_explainer, code_agent],
    model_settings=ModelSettings(include_usage=True),
)

# --- Ejemplo de ejecución ---
if __name__ == "__main__":
    # Aquí pruebas directamente al doc_explainer
    r = Runner.run_sync(
        triage_agent,
        input = [{
            "role": "user",
            "content": 
                "Como puedo instalar pandas https://pandas.pydata.org/docs/getting_started/install.html"
        }]
    )
    print(r.final_output)