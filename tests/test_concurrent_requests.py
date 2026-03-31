"""
Test concurrent request handling with multiple workers
"""
import os
import sys
import asyncio
import time
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AsyncLLM, PromptTemplate, StrOutputParser

async def test_concurrent_requests():
    """Test multiple concurrent requests"""
    print("\n" + "="*70)
    print("🚀 CONCURRENT REQUESTS TEST")
    print("="*70)
    
    # Note: You'll need an actual model loaded for this test
    # This test assumes you have a model available
    
    try:
        llm = AsyncLLM()
        
        # Load model (adjust path as needed)
        await llm._load_model(
            model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            temperature=0.7,
            n_predict=100,  # Small for quick responses
            n_ctx=2048,
            repeat_penalty=1.1,
            stop=["<|im_end|>"]
        )
        await llm.start()
        
        parser = StrOutputParser()
        pt = PromptTemplate(
            template="""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
""",
            input_variables=["input"]
        )
        
        chain = pt | llm | parser
        
        # Send multiple requests concurrently
        questions = [
            "What is Python? Answer briefly.",
            "What is AI? Answer briefly.",
            "What is machine learning? Answer briefly.",
            "What is async programming? Answer briefly.",
        ]
        
        print(f"\n📤 Sending {len(questions)} concurrent requests...")
        start_time = time.time()
        
        tasks = [chain.ainvoke({"input": q}) for q in questions]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ All responses received in {elapsed:.2f} seconds")
        print("\n📋 Responses:")
        for i, (q, r) in enumerate(zip(questions, responses)):
            if isinstance(r, Exception):
                print(f"   {i+1}. {q[:30]}... → ❌ Error: {r}")
            else:
                print(f"   {i+1}. {q[:30]}... → ✅ {str(r)[:60]}...")
        
        await llm.stop()
        llm.unload_all_models()
        
        print("\n🎯 Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_queue_backlog():
    """Test queue handling with backlog"""
    print("\n" + "="*70)
    print("📊 QUEUE BACKLOG TEST")
    print("="*70)
    
    try:
        llm = AsyncLLM()
        
        await llm._load_model(
            model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            temperature=0.7,
            n_predict=50,
            n_ctx=2048,
            stop=["<|im_end|>"]
        )
        await llm.start()
        
        parser = StrOutputParser()
        pt = PromptTemplate(
            template="""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
""",
            input_variables=["input"]
        )
        
        chain = pt | llm | parser
        
        # Send many requests quickly to create backlog
        num_requests = 10
        print(f"\n📤 Sending {num_requests} requests to create backlog...")
        
        tasks = []
        for i in range(num_requests):
            tasks.append(chain.ainvoke({"input": f"Count to {i+1}"}))
        
        # Wait for all to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in responses if not isinstance(r, Exception))
        print(f"\n✅ {successful}/{num_requests} requests completed successfully")
        
        await llm.stop()
        llm.unload_all_models()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("Choose test:")
    print("1. Concurrent Requests")
    print("2. Queue Backlog")
    
    choice = input("\nEnter choice (1/2): ")
    
    if choice == "1":
        asyncio.run(test_concurrent_requests())
    elif choice == "2":
        asyncio.run(test_queue_backlog())
    else:
        print("Invalid choice")