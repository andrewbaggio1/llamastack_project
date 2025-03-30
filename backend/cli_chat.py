import os
import sys
import dotenv

dotenv.load_dotenv()

def create_http_client():
    from llama_stack_client import LlamaStackClient
    return LlamaStackClient(
        base_url=f"http://llama-stack:{os.getenv('LLAMA_STACK_PORT')}"
    )

client = create_http_client()

# List available models
models = client.models.list()
print("--- Available models: ---")
for m in models:
    print(f"- {m.identifier}")
print()

model_id = os.getenv("INFERENCE_MODEL")
if not model_id:
    print("INFERENCE_MODEL not set in .env")
    sys.exit(1)

# Chat history to maintain conversation context
chat_history = [
    {"role": "system", 
     "content": 
     
     """
        You are a AI that analyzes incidents in bodycam transcriptions.
        You will be given audio transcripts
        If you recognize events that constitute conflict or altercations
        summarize and flag it for manual review. There may be multiple
        incidents in one transcript.
        if the transcription contains nothing of note, say
        that there is nothing of note.
    """
    }
]

print("Chatbot is ready! Type 'exit' to end the conversation.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    # Append user message to history
    chat_history.append({"role": "user", "content": user_input})

    try:
        # Get response from model
        response = client.inference.chat_completion(
            model_id=model_id,
            messages=chat_history
        )

        assistant_message = response.completion_message.content
        stop_reason = response.completion_message.stop_reason or "stop"

        print(f"Bot: {assistant_message}\n")

        # Append bot response to history (include stop_reason)
        chat_history.append({
            "role": "assistant",
            "content": assistant_message,
            "stop_reason": stop_reason
        })

    except Exception as e:
        print(f"An error occurred: {e}\n")



