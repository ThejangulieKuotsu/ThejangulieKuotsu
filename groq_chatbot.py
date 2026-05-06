from groq import Groq
import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set. Please create a .env file with your API key.")

# Configuration
DEFAULT_MODEL = "qwen/qwen3-32b"
DEFAULT_TEMPERATURE = 0.1
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant. Keep replies short and clear.")
MAX_HISTORY_LENGTH = 20  # Prevent token overflow

client = Groq(api_key=api_key)


def add_to_chat_history(history: List[Dict[str, str]], role: str, content: str) -> None:
    """Append one message to chat history."""
    history.append({"role": role, "content": content})


def trigger_model(history: List[Dict[str, str]], model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> str:
    """Send full chat history to the model and return assistant text."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc


def manage_history_length(history: List[Dict[str, str]], max_length: int = MAX_HISTORY_LENGTH) -> None:
    """Keep only the system message and recent messages to prevent token overflow."""
    if len(history) > max_length:
        # Keep system message (index 0) and most recent messages
        history[:] = [history[0]] + history[-(max_length - 1):]


def chat_interface() -> None:
    """Simple CLI chat loop using in-memory history."""
    history: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    print("Simple Groq Chat")
    print("Type 'exit' or 'quit' to quit, 'clear' to reset conversation.\n")

    while True:
        user_text = input("You: ").strip()
        
        if not user_text:
            continue
        
        if user_text.lower() in {"exit", "quit"}:
            print("Chat ended.")
            break
        
        if user_text.lower() == "clear":
            history = [history[0]]  # Keep system message, clear rest
            print("Conversation cleared.\n")
            continue
        
        add_to_chat_history(history, "user", user_text)

        try:
            assistant_text = trigger_model(history)
            add_to_chat_history(history, "assistant", assistant_text)
            print(f"Assistant: {assistant_text}\n")
            
            # Manage history length to prevent token overflow
            manage_history_length(history)
            
        except RuntimeError as exc:
            print(f"Assistant: {exc}\n")
            continue


if __name__ == "__main__":
    chat_interface()
