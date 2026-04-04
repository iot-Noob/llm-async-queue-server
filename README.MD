# 🚀 Async LLM Engine – Production-Ready LLM Serving with LangChain & FastAPI

![Python Version](https://img.shields.io/badge/python-3.10.19-blue.svg)
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
| **📊 Memory Cleanup** | `gc.collect()` + `malloc_trim` to return GPU memory to the OS. |
| **🧹 Graceful Shutdown** | Cancels pending futures and stops the worker cleanly. |
| **🔄 Streaming Support** | Real-time token streaming with cancellation support – perfect for chat applications. |
| **📝 Prompt Templating** | `PromptTemplate`, `ChatPromptTemplate` for dynamic prompt generation and variables. |
| **📊 JSON/Pydantic Parsing** | `JsonOutputParser`, `PydanticOutputParser` for structured LLM output validation. |
| **📈 Metrics & Monitoring** | Real-time metrics tracking (requests, errors, streams, queue size). |
| **🔐 Thread-Safe Operations** | Lock-based coordination for concurrent access to shared resources. |
| **📋 Document Management** | Optimized `Document` class for RAG pipelines with metadata handling and efficient batching. |
| **🎨 ECharts Integration** | `EcharrParsers` for template-based chart data extraction from LLM responses. |
| **⚙️ Configurable Parameters** | Central `Config` class for queue size, timeouts, memory limits, and auto-restart policies. |

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

### 4. Streaming in FastAPI

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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

@app.post("/chat/stream")
async def chat_stream(prompt: str):
    """Stream tokens in real-time"""
    async def token_generator():
        async for token in llm.stream_llm(prompt):
            yield f"data: {token}\n\n"
    
    return StreamingResponse(token_generator(), media_type="text/event-stream")

@app.get("/health")
async def health():
    metrics = await llm.get_metrics()
    return {
        "status": "healthy" if await llm.is_healthy() else "unhealthy",
        "models_loaded": metrics["models_loaded"],
        "queue_size": metrics["queue_size"],
        "active_streams": metrics["active_streams"]
    }
```

Run with:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## 🔧 API Reference

### `AsyncLLM` – Main Service Class

| Method | Description |
|--------|-------------|
| `__init__(**kwargs)` | Initializes queue, config, and uvloop. |
| `async start()` | Starts the background worker task. |
| `async stop()` | Gracefully stops the worker and cancels pending requests. |
| `async chat_llm(chat: str, chat_id: str = None, timeout: float = None, **gen_kwargs) -> str` | Sends a prompt and returns the response. |
| `async stream_llm(chat: str, chat_id: str = None, timeout: float = None, **gen_kwargs)` | Streams tokens in real-time (async generator). |
| `async _load_model(**kwargs)` | Loads a `.gguf` model with configurable parameters. |
| `async unload_model(model_name: str = None)` | Unloads a specific or all models. |
| `async get_loaded_models() -> List[str]` | Returns names of loaded models. |
| `async get_memory_usage() -> dict` | Returns RSS, VMS, and model information. |
| `async is_healthy() -> bool` | Checks if service is running with models loaded. |
| `async get_metrics() -> dict` | Returns real-time metrics (requests, errors, streams, queue size). |
| `async cancel_request(request_id: str)` | Cancels a pending request or stream by ID. |
| `async ainvoke(input: Union[str, Dict]) -> str` | LangChain-compatible async invoke. |
| `invoke(input: Union[str, Dict]) -> str` | LangChain-compatible sync invoke. |

### `PromptTemplate` – Dynamic Prompt Formatting

```python
from main import PromptTemplate

template = PromptTemplate(
    template="You are a helpful assistant. Answer this: {question}",
    input_variables=["question"]
)

# Format manually
formatted = template.format(question="What is Python?")

# Use partial variables
partial_template = template.partial(question="What is Python?")

# Chain with LLM
from main import AsyncLLM
llm = AsyncLLM()
chain = template | llm
result = await chain.ainvoke({"question": "What is Python?"})
```

### `ChatPromptTemplate` – Multi-turn Conversations

```python
from main import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant."),
    ("human", "Explain {topic}"),
    ("assistant", "Sure! Here's an explanation of {topic}:")
])

# Format prompt
prompt = chat_template.format_prompt(topic="async programming")
```

### `JsonOutputParser` – Extract Structured Data

```python
from main import JsonOutputParser
from pydantic import BaseModel

class CodeReview(BaseModel):
    issues: list[str]
    suggestions: list[str]

parser = JsonOutputParser(pydantic_object=CodeReview)
llm_response = '```json\n{"issues": ["bug1"], "suggestions": ["fix1"]}\n```'
result = parser.parse(llm_response)  # Returns CodeReview instance
```

### `Document` – RAG & Retrieval

```python
from main import Document

# Create a document
doc = Document(page_content="Machine learning is...", metadata={"source": "wiki"})

# Combine documents
docs = [doc1, doc2, doc3]
combined = Document.combine(docs)

# Export
json_str = combined.to_json()
doc_dict = combined.to_dict()
```

### `EcharrParsers` – Chart Template Extraction

```python
from main import EcharrParsers

parser = EcharrParsers()
available_charts = parser.get_enum_list()  # List all charts
template = parser.get_full_chart_template("bar_chart")
prompt = parser.get_dynamic_prompt("Show sales by region", "bar_chart")
```

### Configuration

The `AsyncLLM.Config` class controls behavior:

```python
llm = AsyncLLM(
    queue_maxsize=10,               # Max pending requests
    default_timeout=160.0,           # Request timeout (seconds)
    available_ram=3000,              # Min free RAM (MB) before load fails
    graceful_shutdown=True,          # Wait for in-flight requests on stop
    enable_metrics=True,             # Track metrics
    enable_cancellation=True,        # Allow request cancellation
    reject_on_full_queue=True,       # Reject vs block on full queue
    auto_restart_worker=False,       # Auto-restart if worker crashes
    worker_exception_callback=None,  # Custom crash handler
)
```

---

## 📡 Advanced Features

### Streaming with Real-Time Token Yielding

```python
import asyncio
from async_llm_engine import AsyncLLM

async def stream_chat():
    llm = AsyncLLM()
    await llm._load_model(model_name="mistral-7b-instruct.Q4_K_M.gguf")
    await llm.start()
    
    print("Streaming response: ", end="", flush=True)
    async for token in llm.stream_llm("Tell me a joke"):
        print(token, end="", flush=True)
    print("\n")
    
    await llm.stop()

asyncio.run(stream_chat())
```

### Request Cancellation

```python
import asyncio

async def cancel_demo():
    llm = AsyncLLM()
    await llm._load_model(model_name="mistral-7b-instruct.Q4_K_M.gguf")
    await llm.start()
    
    # Start a request
    task = asyncio.create_task(llm.chat_llm("Solve this complex problem...", chat_id="req-123"))
    
    # Cancel it after 5 seconds
    await asyncio.sleep(5)
    await llm.cancel_request("req-123")
    
    try:
        result = await task
    except asyncio.CancelledError:
        print("Request was cancelled!")
    
    await llm.stop()

asyncio.run(cancel_demo())
```

### Structured Output with Pydantic

```python
from main import JsonOutputParser, AsyncLLM
from pydantic import BaseModel

class SentimentAnalysis(BaseModel):
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float

parser = JsonOutputParser(pydantic_object=SentimentAnalysis)

async def analyze():
    llm = AsyncLLM(output_parser=parser)
    await llm._load_model(model_name="mistral-7b-instruct.Q4_K_M.gguf")
    await llm.start()
    
    prompt = """Analyze the sentiment of this text:
    "I love using this framework, it's incredibly efficient!"
    
    Return JSON with {text, sentiment, confidence}"""
    
    result = await llm.chat_llm(prompt)  # Returns SentimentAnalysis instance
    print(f"Sentiment: {result.sentiment} ({result.confidence})")
    
    await llm.stop()

asyncio.run(analyze())
```

### ECharts Data Extraction

```python
from main import EcharrParsers, PromptTemplate, AsyncLLM

async def extract_chart_data():
    llm = AsyncLLM()
    await llm._load_model(model_name="mistral-7b-instruct.Q4_K_M.gguf")
    await llm.start()
    
    parser = EcharrParsers()
    
    # Get dynamic prompt for a specific chart
    user_input = "Sales data: Q1=100, Q2=150, Q3=200, Q4=180"
    chart_prompt = parser.get_dynamic_prompt(user_input, "bar_chart")
    
    response = await llm.chat_llm(chart_prompt)
    print(response)  # LLM returns JSON-formatted chart data
    
    await llm.stop()

asyncio.run(extract_chart_data())
```

### Metrics & Monitoring

```python
import asyncio
from async_llm_engine import AsyncLLM

async def monitor():
    llm = AsyncLLM()
    await llm._load_model(model_name="mistral-7b-instruct.Q4_K_M.gguf")
    await llm.start()
    
    # Start some concurrent requests
    tasks = [
        llm.chat_llm(f"Task {i}") 
        for i in range(5)
    ]
    
    # Monitor metrics while requests run
    for _ in range(3):
        await asyncio.sleep(1)
        metrics = await llm.get_metrics()
        print(f"Queue size: {metrics['queue_size']}")
        print(f"Total requests: {metrics['total_requests']}")
        print(f"Completed: {metrics['completed_requests']}")
        print(f"Errors: {metrics['total_errors']}")
        print("---")
    
    await asyncio.gather(*tasks, return_exceptions=True)
    await llm.stop()

asyncio.run(monitor())
```

---

## 📚 Core Components

### Prompt Templates

**PromptTemplate** – Simple text template with variables:
```python
template = PromptTemplate.from_template("Translate to {language}: {text}")
prompt = template.format(language="Spanish", text="Hello world")
```

**ChatPromptTemplate** – Multi-turn conversation format:
```python
chat = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{user_input}"),
])
```

### Output Parsers

- **StrOutputParser** – Extracts plain text from LLM responses
- **JsonOutputParser** – Extracts and validates JSON with optional Pydantic models
- **PydanticOutputParser** – Strict Pydantic model validation

### Document Class

Optimized for RAG pipelines with metadata, combining, and batching:

```python
# Create
doc = Document(page_content="...", metadata={"source": "wiki"})

# Combine multiple
combined = Document.combine([doc1, doc2, doc3])

# Export
doc.to_json()
doc.to_dict()
Document.from_json(json_str)
```

### EcharrParsers

Integrates JSON templates for ECharts and extracts data from LLM responses:

```python
parser = EcharrParsers()  # Loads Charts.json

# Get available charts
charts = parser.get_enum_list()

# Get template for a chart
template = parser.get_full_chart_template("bar_chart")

# Generate prompts
prompt = parser.get_dynamic_prompt(user_data, "bar_chart")
```

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

`AsyncLLM` and all prompt/parser classes implement the LangChain `Runnable` interface:

```python
# ✅ Works with | operator for chaining
chain = (
    ChatPromptTemplate.from_template("Summarize: {text}") 
    | llm 
    | JsonOutputParser(pydantic_object=SummaryModel)
)

result = await chain.ainvoke({"text": "Long document..."})
```

All components support:
- **sync invoke()** – Synchronous calls  
- **async ainvoke()** – Async/await support
- **Pipe operator (`|`)** – Chain components seamlessly
- **Metadata preservation** – chat_id, timeout, and custom kwargs flow through pipes

---

## � Resilience & Error Handling

### Worker Auto-Restart

Automatically recover from worker crashes:

```python
def on_worker_crash(exception: Exception):
    logger.error(f"Worker crashed: {exception}")
    # Send alert, log metrics, etc.

llm = AsyncLLM(
    auto_restart_worker=True,
    worker_exception_callback=on_worker_crash
)
```

### Graceful Shutdown

Wait for pending requests to complete or fail them safely:

```python
async def shutdown():
    llm = AsyncLLM(graceful_shutdown=True)
    # ... requests running ...
    await llm.stop()  # Waits for queue to finish
```

### Request Timeouts

Every request is protected with a timeout:

```python
try:
    result = await llm.chat_llm(
        "Complex query...",
        timeout=30.0  # 30 second timeout
    )
except asyncio.TimeoutError:
    print("Request timed out")
```

---

## �📊 Performance

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
