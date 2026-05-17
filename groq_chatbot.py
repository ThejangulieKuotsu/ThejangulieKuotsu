import os
from groq import Groq

# It's best practice to pull the key from environment variables 
# instead of hardcoding it. Set it in your terminal: export GROQ_API_KEY="your_key"
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "your_fallback_key_here"))

def add_to_chat_history(history, role, content):
	"""Append one message to chat history."""
	history.append({"role": role, "content": content})
	return history


def trigger_model(history, model="qwen/qwen3-32b", temperature=0.1):
	"""Send full chat history to the model and return assistant text."""
	completion = client.chat.completions.create(
		model=model,
		messages=history,
		temperature=temperature,
	)
	return completion.choices[0].message.content

  
def chat_interface():
	"""Simple CLI chat loop using in-memory history."""
	history = [
		{
			"role": "system",
			"content": "You are a helpful assistant. Keep replies short and clear.",
		}
	]

	print("Simple Groq Chat")
	print("Type 'exit' to quit.\n")

	while True:
		user_text = input("You: ").strip()
		if not user_text:
			continue
		if user_text.lower() in {"exit", "quit"}:
			print("Chat ended.")
			break
		
		# 1. Add the user message
		add_to_chat_history(history, "user", user_text)

		try:
			assistant_text = trigger_model(history)
		except Exception as exc:
			print(f"Assistant: Request failed: {exc}")
			# 2. FIX: Remove the failed user message so history doesn't get corrupted
			history.pop() 
			continue

		# 3. Add assistant response if successful
		add_to_chat_history(history, "assistant", assistant_text)
		print(f"Assistant: {assistant_text}\n")


if __name__ == "__main__":
	chat_interface()
