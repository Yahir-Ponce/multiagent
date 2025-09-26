import os
import time
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
CLIENT = OpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=f"{os.getenv('AZURE_API_BASE').rstrip('/')}/openai/v1/",
)


# --- 1) Texto simple ---
async def text_demo():
    resp = CLIENT.responses.create(
        model="gpt-4.1",
        input='Solamente di: "OK, esto es una prueba"',
    )
    print("\n[Texto simple]")
    print(resp.output_text)


# --- 2) Visión por URL ---
async def vision_demo():
    resp = CLIENT.responses.create(
        model="gpt-5",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is the name of the animal in this image?"},
                {"type": "input_image", "image_url": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg"},
            ],
        }],
    )
    print("\n[Visión por URL]")
    print(resp.output_text)


# --- 3) Streaming ---
async def stream_demo():
    stream = CLIENT.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Que es RAG ???"}],
        stream=True,
    )
    print("\n[Streaming]")
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
            time.sleep(0.03)


if __name__ == "__main__":
    asyncio.run(text_demo())
    asyncio.run(vision_demo())
    asyncio.run(stream_demo())