from main import EcharrParsers, AsyncLLM, StrOutputParser, PromptTemplate
import asyncio
import sys

async def main():
    # Initialize components
    ecp = EcharrParsers()
    llm = AsyncLLM()
    parser = StrOutputParser()
    
    pt = PromptTemplate(
        template="""<|im_start|>system
You are a helpful assistant that provides accurate, informative answers. Answer questions clearly and completely.
<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant

""",
        input_variables=["input"]
    )
    
    # Load and start LLM
    print("\n📦 Loading model...")
    await llm._load_model(
        model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        temperature=0.6,
        n_predict=2046,
        n_ctx=3024,
        stop=["<|im_end|>"]
    )
    
    print("🚀 Starting worker...")
    await llm.start()
    
    # Create chain
    chain = pt | llm | parser
    
    # Create queues
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    
    # Control flags
    running = True
    
    async def process_worker():
        """Worker that processes prompts from queue"""
        while running:
            try:
                # Get next prompt
                prompt = await input_queue.get()
                if prompt is None:
                    break
                
                # Process
                try:
                    response = await chain.ainvoke({"input": prompt})
                    await output_queue.put((prompt, response))
                except Exception as e:
                    await output_queue.put((prompt, f"Error: {e}"))
                
                input_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker error: {e}")
    
    async def output_printer():
        """Prints responses as they come"""
        while running:
            try:
                prompt, response = await asyncio.wait_for(output_queue.get(), timeout=0.1)
                print(f"\n{'='*60}")
                print(f"💬 Response to: {prompt}")
                print(f"{'='*60}")
                print(response)
                print(f"{'='*60}\n")
                print("You: ", end="", flush=True)
            except asyncio.TimeoutError:
                continue
    
    async def input_handler():
        """Handles user input"""
        nonlocal running
        while running:
            try:
                # Use asyncio.to_thread for non-blocking input
                ui = await asyncio.to_thread(input, "You: ")
                
                if ui.lower() == "exit":
                    running = False
                    break
                elif ui.lower() == "status":
                    print(f"\n📊 Queue size: {input_queue.qsize()}\n")
                    print("You: ", end="", flush=True)
                else:
                    await input_queue.put(ui)
                    print(f"📝 Queued: {ui}")
                    
            except Exception as e:
                print(f"Input error: {e}")
    
    print("\n" + "="*60)
    print("💬 CHAT MODE ACTIVE - Like ChatGPT web UI")
    print("="*60)
    print("Type your messages, press Enter to send")
    print("Type 'exit' to quit")
    print("Type 'status' to see queue")
    print("="*60 + "\n")
    
    # Start all tasks
    worker_task = asyncio.create_task(process_worker())
    printer_task = asyncio.create_task(output_printer())
    input_task = asyncio.create_task(input_handler())
    
    # Wait for input task to finish (when user types exit)
    await input_task
    
    # Clean shutdown
    running = False
    await input_queue.put(None)
    
    # Cancel remaining tasks
    worker_task.cancel()
    printer_task.cancel()
    
    # Wait for tasks to finish
    await asyncio.gather(worker_task, printer_task, return_exceptions=True)
    
    # Cleanup
    print("\n" + "="*60)
    print("CLEANING UP:")
    print("="*60)
    await llm.stop()
    llm.unload_all_models()
    
    print("\n✅ Done!")

if __name__ == "__main__":
    asyncio.run(main())
