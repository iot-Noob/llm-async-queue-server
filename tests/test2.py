import os
import sys
# Add parent directory to path so Python can find main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from main import PromptTemplate, AsyncLLM, StrOutputParser
import asyncio
from uuid import uuid4
import sys

async def main():
    try:
        llm = AsyncLLM()
        queue = asyncio.Queue(maxsize=11)
        
        # Load model
        await llm._load_model(   
            model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            temperature=0.6,
            n_predict=512,
            n_ctx=4096,
            stop=["<|im_end|>"]
        )
        await llm.start()
        
        parser = StrOutputParser()
        pt = PromptTemplate(
            template="""<|im_start|>system
You are a helpful assistant. Provide accurate, informative responses.<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
""",
            input_variables=["input"]
        )
        chain = pt | llm | parser
        
        running = True
        
        async def add_to_queue(msg: str):
            future = asyncio.Future()
            await queue.put({"message": msg, "future": future})
            print(f"[QUEUE] + {msg[:40]} (size: {queue.qsize()})")
            return await future
        
        async def worker():
            while running:
                item = None
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                    msg = item["message"]
                    future = item["future"]
                    print(f"[WORK] Processing: {msg[:40]}...")
                    response = await chain.ainvoke({"input": msg})
                    future.set_result(response)
                    print(f"[WORK] Done: {msg[:30]}...")
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if item and 'future' in item:
                        item['future'].set_exception(e)
                finally:
                    if item:
                        queue.task_done()
        
        # Non-blocking input reader
        async def async_input(prompt):
            """Read input without blocking the event loop"""
            return await asyncio.to_thread(input, prompt)
        
        print("\n" + "="*60)
        print("🤖 CONCURRENT CHATBOT - Type anytime! (like Gemini)")
        print("="*60)
        print("Commands: 'exit', 'status'")
        print("-"*60)
        
        # Start worker
        worker_task = asyncio.create_task(worker())
        
        # Main input loop - never blocks
        while True:
            # Show prompt and get input
            user_input = await async_input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if user_input.lower() == "status":
                print(f"Queue size: {queue.qsize()}")
                continue
            
            # 🔥 FIRE AND FORGET - response will print when ready
            async def respond(msg=user_input):
                response = await add_to_queue(msg)
                # Print response cleanly, then re-show prompt
                print(f"\nBot: {response}")
                print("You: ", end="", flush=True)
            
            asyncio.create_task(respond())
        
        # Cleanup
        print("\nShutting down...")
        running = False
        worker_task.cancel()
        await asyncio.gather(worker_task, return_exceptions=True)
        await llm.stop()
        llm.unload_all_models()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())