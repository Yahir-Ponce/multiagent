from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-09-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-codex")      # alias del deployment
resp = client.responses.create(model=DEPLOYMENT, input="Return exactly: OK")
print(resp.output_text)
