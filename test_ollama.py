import ollama

from config import LLM_MODEL

response = ollama.chat(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Ollama connection successful",
        }
    ],
)

print(response["message"]["content"])