"""
Test resource cleanup and memory management
"""
import os
import sys
import asyncio
import psutil
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AsyncLLM

async def test_resource_cleanup():
    """Test proper resource cleanup"""
    print("\n" + "="*70)
    print("🧹 RESOURCE CLEANUP TEST")
    print("="*70)
    
    process = psutil.Process(os.getpid())
    
    def get_memory():
        return process.memory_info().rss / 1024 / 1024
    
    print(f"📊 Initial memory: {get_memory():.2f} MB")
    
    test_cases = [
        ("Normal operation", test_normal_operation),
        ("Exception handling", test_exception_handling),
        ("Early cancellation", test_early_cancellation),
        ("Multiple load/unload", test_multiple_load_unload),
    ]
    
    for name, test_func in test_cases:
        print(f"\n📝 Testing: {name}")
        before_mem = get_memory()
        
        try:
            await test_func()
            print(f"   ✅ Test passed")
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
        
        gc.collect()
        await asyncio.sleep(1)
        after_mem = get_memory()
        
        print(f"   Memory before: {before_mem:.2f} MB")
        print(f"   Memory after:  {after_mem:.2f} MB")
        print(f"   Difference:    {after_mem - before_mem:+.2f} MB")

async def test_normal_operation():
    """Test normal operation with proper cleanup"""
    llm = AsyncLLM()
    try:
        await llm._load_model(
            model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            temperature=0.7,
            n_predict=10,
            n_ctx=1024
        )
        await llm.start()
        await asyncio.sleep(0.5)
    finally:
        await llm.stop()
        llm.unload_all_models()

async def test_exception_handling():
    """Test cleanup after exception"""
    llm = AsyncLLM()
    try:
        await llm._load_model(
            model_name="nonexistent-model.gguf",
            temperature=0.7
        )
    except ValueError:
        pass  # Expected exception
    finally:
        await llm.stop()
        llm.unload_all_models()

async def test_early_cancellation():
    """Test cleanup after early cancellation"""
    llm = AsyncLLM()
    try:
        await llm._load_model(
            model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            temperature=0.7,
            n_predict=10,
            n_ctx=1024
        )
        await llm.start()
        
        # Create a task and cancel it
        task = asyncio.create_task(asyncio.sleep(2))
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
            
    finally:
        await llm.stop()
        llm.unload_all_models()

async def test_multiple_load_unload():
    """Test multiple load/unload cycles"""
    for i in range(3):
        llm = AsyncLLM()
        try:
            await llm._load_model(
                model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
                temperature=0.7,
                n_predict=10,
                n_ctx=1024
            )
            await llm.start()
            await asyncio.sleep(0.2)
        finally:
            await llm.stop()
            llm.unload_all_models()
            await asyncio.sleep(0.3)

if __name__ == "__main__":
    asyncio.run(test_resource_cleanup())