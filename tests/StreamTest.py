import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AsyncLLM

async def test_streaming():
    llm = AsyncLLM()

    # Load the model (use the same parameters as before)
    await llm._load_model(
        model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        temperature=0.7,
        n_predict=2046,
        streaming=True,      # this flag is for the model loading; streaming will be enabled per request
        n_ctx=4024,
        stop=["<|im_end|>"]
    )

    await llm.start()

    prompt = "Hi, tell me a very short joke."

    print(f"Streaming response for: {prompt}\n")
    print(">> ", end="", flush=True)

    # Use the streaming method
    async for token in llm.stream_llm(prompt, chat_id="stream_test"):
        print(token, end="", flush=True)   # prints each token as it arrives

    print("\n\nStreaming finished.")

    # Clean up
    await llm.stop()
    llm.unload_all_models()

if __name__ == "__main__":
    asyncio.run(test_streaming())