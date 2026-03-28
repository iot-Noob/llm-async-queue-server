# 🚀 Async LLM Engine – Production-Ready LLM Serving with LangChain & FastAPI

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Async](https://img.shields.io/badge/async-uvloop-brightgreen.svg)
![LangChain](https://img.shields.io/badge/LangChain-integrated-orange.svg)

**Async LLM Engine** is a high‑performance, production‑ready asynchronous LLM inference server built with **LangChain**, **FastAPI**, **uvloop**, and a thread‑safe **queue‑worker pattern**. It handles hundreds of concurrent users with a single GPU, protects hardware resources, and integrates seamlessly with LangChain’s pipe (`|`) operator.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **⚡ Asynchronous Core** | Powered by `uvloop` (2–4× faster than standard asyncio) for maximum concurrency. |
| **🔄 Queue‑Worker Pattern** | One background worker processes LLM requests sequentially – protects GPU and prevents overload. |
| **🧩 LangChain Compatibility** | Works with `|` operator – chain prompts, LLMs, and parsers just like native LangChain. |
| **🚀 FastAPI Ready** | Easy integration with FastAPI; includes `/health` and `/chat` endpoints out‑of‑the‑box. |
| **💾 Model Management** | Load / unload models, search `.gguf` files recursively, and free GPU memory. |
| **⏱️ Timeout Protection** | Every request has a configurable timeout – no hanging calls. |
| **📊 Memory Cleanup** | `gc.collect()` + `malloc_trim` to return memory to the OS. |
| **🧹 Graceful Shutdown** | Cancels pending futures and stops the worker cleanly. |

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/async-llm-engine.git
cd async-llm-engine
pip install -r requirements.txt
```

### Requirements

```txt
langchain>=0.1.0
langchain-community>=0.1.0
langchain-core>=0.1.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
uvloop>=0.19.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
psutil>=5.9.0
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ASYNC EVENT LOOP                            │
│  (uvloop – runs all async tasks on a single thread)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User 1 ──┐                                                     │
│  User 2 ──┼──> asyncio.Queue  ──> Worker Task ──> LLM Call    │
│  User N ──┘      (Global Store)      │        (in thread)       │
│                     │                │                         │
│                     ▼                ▼                         │
│             Each user gets a      result via                   │
│             dedicated Future      future.set_result()          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     THREAD POOL                                 │
│  (asyncio.to_thread) – runs blocking LLM inference             │
└─────────────────────────────────────────────────────────────────┘
```

**Key Concepts:**
- **Queue** = Global buffer (like a Redux store) holding all pending requests.
- **Worker** = Single async task that processes the queue one by one.
- **Future** = Promise given to each user – worker fulfills it when the LLM responds.
- **Timeout** = `asyncio.wait_for` prevents hanging requests.

---

## 🚀 Quick Start

### 1. Configure Environment

Create a `.env` file:

```bash
MODEL_PATH=/path/to/your/models/directory
```

### 2. Basic Usage (Async)

```python
import asyncio
from async_llm_engine import AsyncLLM

async def main():
    # Initialize the service
    llm = AsyncLLM()
    
    # Load a model (searches .gguf files recursively)
    await llm._load_model(model_name="mistral-7b-instruct.Q4_K_M.gguf")
    
    # Start the background worker
    await llm.start()
    
    # Simple chat
    response = await llm.chat_llm("What is Python?")
    print(response)
    
    # Multiple concurrent users
    results = await asyncio.gather(
        llm.chat_llm("Tell me a joke"),
        llm.chat_llm("Explain async programming"),
        llm.chat_llm("What is machine learning?")
    )
    
    # Clean shutdown
    await llm.stop()
    llm.unload_model()

asyncio.run(main())
```

### 3. LangChain Pipe Operator (`|`)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = AsyncLLM()
await llm._load_model(model_name="mistral-7b-instruct.Q4_K_M.gguf")
await llm.start()

# Create a LangChain chain – works with | operator!
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm | StrOutputParser()

result = await chain.ainvoke({"topic": "programming"})
print(result)

await llm.stop()
```

### 4. FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

llm = AsyncLLM()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await llm._load_model(model_name="mistral-7b-instruct.Q4_K_M.gguf")
    await llm.start()
    yield
    await llm.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(prompt: str):
    try:
        response = await llm.chat_llm(prompt)
        return {"response": response}
    except TimeoutError:
        raise HTTPException(504, "Request timed out")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": len(llm.current_model),
        "queue_size": llm.queue.qsize()
    }
```

Run with:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## 🔧 API Reference

### `AsyncLLM`

| Method | Description |
|--------|-------------|
| `__init__()` | Initializes the queue, futures, and uvloop. |
| `async start()` | Starts the background worker. |
| `async stop()` | Stops the worker and cancels pending futures. |
| `async chat_llm(prompt: str) -> str` | Sends a prompt and returns the response. |
| `async _load_model(**kwargs)` | Loads a `.gguf` model. |
| `unload_model(model_name: str = None)` | Unloads a specific model or all models. |
| `get_loaded_models() -> List[str]` | Returns names of loaded models. |
| `get_memory_usage() -> dict` | Returns RSS, VMS, and model information. |

---

## 🧠 How It Works – Explained

### Queue as a Global Store
The `asyncio.Queue` acts like a **Redux store** – all pending requests (actions) are placed in it. The worker (reducer) processes them sequentially.

### Worker – One at a Time
Only **one** LLM call runs at any moment, protecting the GPU. Other users’ async tasks are paused at `await future`, leaving the event loop free.

### Future – Promise of a Result
Each user receives a unique `asyncio.Future`. When the worker finishes, it calls `future.set_result()` – only that specific user is woken up.

### Timeout – Safety Net
Every `await future` is wrapped in `asyncio.wait_for(future, timeout=60)`. If the LLM hangs, a `TimeoutError` is raised and the request is cleaned up.

### uvloop – Faster Event Loop
By replacing the default asyncio loop with `uvloop`, task switching and I/O operations become 2–4× faster.

### LangChain Integration
`AsyncLLM` inherits from `Runnable` and implements `ainvoke()`. This allows it to be used in LangChain pipelines with the `|` operator.

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Model** | 7B parameter, 4‑bit quantized |
| **VRAM** | ~5 GB |
| **Per‑request latency** | 3–5 seconds |
| **Concurrent waiting users** | Unlimited (async tasks, ~1KB each) |
| **Concurrent LLM calls** | 1 (GPU protected) |
| **Throughput** | ~12–20 requests/minute |

---

## 🧪 Testing

```python
# Run the built‑in test
python -m async_llm_engine

# Or run the FastAPI server and use curl
curl -X POST "http://localhost:8000/chat?prompt=What%20is%20Python%3F"
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT © Talha

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/yourusername/async-llm-engine?style=social)](https://github.com/yourusername/async-llm-engine)

---

## 🔗 Related Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [uvloop – Fast asyncio Event Loop](https://github.com/MagicStack/uvloop)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
```