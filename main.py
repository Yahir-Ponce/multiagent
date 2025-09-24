import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=f"{os.getenv('AZURE_API_BASE').rstrip('/')}/openai/v1/",
)

resp = client.responses.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    input="Solamente di: ""OK, esto es una prueba""",
)
print(resp.output_text)
