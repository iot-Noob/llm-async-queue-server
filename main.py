import llama_cpp
from pydantic import BaseModel, Field,field_validator,ConfigDict
from pydantic_settings import BaseSettings
import gc
import ctypes
import os
from langchain.embeddings import Embeddings
from langchain_community.retrievers import WikipediaRetriever
from uuid import uuid4
import asyncio
from typing import Dict,List
import uvloop
from queue import Queue
import psutil
 

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow"   
    )
    MODEL_PATH:str=Field(...)

    @field_validator("MODEL_PATH",mode="before")
    @classmethod
    def validate_path(cls,v:str)->str:
        if not os.path.exists(v):
            raise ValueError(f"Error path not found {v}")
        return v

 

class AsyncLLM:
 
    def __init__(self):
       
        self.futures:Dict[str,asyncio.Future]={}    
        self.current_model:Dict[str,llama_cpp.Llama]={}
        self.setting=Settings()
        self.path=self.setting.MODEL_PATH
        self.loop=uvloop.new_event_loop()
        self.async_lock=asyncio.Lock()
        self.queue=asyncio.Queue()
        self._running=False
        self._worker_task=None
        asyncio.set_event_loop(self.loop)

    async def _get_model_in_current_dirs(self):
        """Get all .gguf models in directory"""
        try:
            
            # Run blocking os.walk in thread to avoid blocking event loop
            all_files = await asyncio.to_thread(self._walk_directory)
            return all_files
        except Exception as e:
            raise ValueError(f"Error getting models: {e}")
    
    def _walk_directory(self):
        """Blocking directory walk - runs in thread"""
        all_files = []
        for root, _, files in os.walk(self.path):
            for f in files:
                if f.endswith(".gguf"):
                    bp = os.path.join(root, f)
                    all_files.append({
                        "path": bp,
                        "filename": f,
                        "directory": root
                    })
        return all_files
 
    async def _load_model(self, **kwargs: dict):
        """
        # GGUF Model loader

        ### kwargs contain param:

        **model_name:** name of gguf model just name of it it will automatically search within dir

        **temperature:** llm temperature for random answers (0.0 = deterministic, 2.0 = very random)

        **top_p:** nucleus sampling threshold - only sample from tokens with cumulative probability >= top_p (0.0 to 1.0)

        **top_k:** top-k sampling - only sample from the top K tokens (1 to 100)

        **streaming:** if True, stream tokens as they're generated token by token

        **repeat_penalty:** penalty for repeating tokens (1.0 = no penalty, >1.0 = penalize repeats)

        **max_tokens:** maximum number of tokens to generate (1 to 4096)

        **n_batch:** batch size for prompt processing (higher = faster but more memory)

        **n_ctx:** context window size - maximum tokens model can remember (512 to 32768)

        **n_threads:** number of CPU threads for inference

        **n_gpu_layers:** number of layers to offload to GPU (-1 = all layers, 0 = CPU only)

        **verbose:** if True, print detailed debug logs during inference

        **stop:** list of stop sequences where generation should end (e.g., ["<|endoftext|>", "<|im_end|>"])
        """
        try:
            # Get required parameters
            model_name = kwargs.get("model_name")
            if not model_name:
                raise ValueError("model_name is required")
            
            # Get optional parameters with defaults
            temperature = kwargs.get("temperature", 0.4)
            top_p = kwargs.get("top_p", 0.9)
            top_k = kwargs.get("top_k", 30)
            streaming = kwargs.get("streaming", False)
            repeat_penalty = kwargs.get("repeat_penalty", 1.15)
            max_tokens = kwargs.get("max_tokens", 1024)
            n_batch = kwargs.get("n_batch", 128)
            n_ctx = kwargs.get("n_ctx", 2048)
            n_threads = kwargs.get("n_threads", 6)
            n_gpu_layers = kwargs.get("n_gpu_layers", -1)
            verbose = kwargs.get("verbose", False)
            stop = kwargs.get("stop", ["<|endoftext|>", "<|im_end|>"])
            
            # Validate model_name
            if not model_name or not isinstance(model_name, str):
                raise ValueError(f"Invalid model_name: {model_name}")
            
            # Check if model already loaded
            if model_name in self.current_model:
                raise ValueError(f"Model '{model_name}' is already loaded")
            
            # Build model path
            fmp = os.path.join(self.path, model_name)
            
            # Check if file exists directly
            if not os.path.exists(fmp):
                # Try to search in subdirectories
                found_path = None
                models = await self._get_model_in_current_dirs()
                for model in models:
                    if model["filename"] == model_name:
                        found_path = model["path"]
                        break
                
                if found_path:
                    fmp = found_path
                else:
                    raise FileNotFoundError(f"Model file '{model_name}' not found in '{self.path}'")
            
            print(f"📦 Loading model: {fmp}")
            print(f"   Temperature: {temperature}")
            print(f"   Max tokens: {max_tokens}")
            print(f"   GPU layers: {n_gpu_layers}")
            
            # Load model in thread to avoid blocking
            llm = await asyncio.to_thread(
                llama_cpp.Llama,
                model_path=fmp,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                streaming=streaming,
                repeat_penalty=repeat_penalty,
                max_tokens=max_tokens,
                n_batch=n_batch,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                stop=stop
            )
            
            # Store loaded model
            self.current_model[model_name] = llm
            
            print(f"✅ Model '{model_name}' loaded successfully!")
            
            return llm
            
        except FileNotFoundError as e:
            raise ValueError(f"Model not found: {e}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def run_task(self,tname,**kw):
        try:
            if callable(tname):
                res= self.loop.run_until_complete(tname(**kw))
                return res
        except Exception as e:
            raise ValueError(f"Error run task due to {e}")

    def get_loaded_models(self) -> List[str]:
    
        """
        Get a list of currently loaded model names.
        
        Returns:
            List[str]: List of loaded model names. Returns empty list if no models loaded.
        
        Example:
            >>> loaded = llm.get_loaded_models()
            >>> print(loaded)  # ['mistral-7b.gguf', 'llama-2.gguf']
            >>> if loaded:
            ...     print(f"Active models: {', '.join(loaded)}")
        """
        try:
            # Convert keys view to list for easier handling
            return list(self.current_model.keys())
            
        except Exception as e:
            print(f"❌ Error getting loaded models: {e}")
            return []  # Return empty list on error

    def unload_model(self, model_name: str = None):
        """
        Unload a model to free memory and GPU resources.
        
        Args:
            model_name (str, optional): Name of the model to unload.
                If None, unloads all loaded models.
        
        Returns:
            bool: True if unloaded successfully, False otherwise.
        
        Example:
            >>> llm.unload_model("mistral-7b.gguf")
            >>> llm.unload_model()  # Unload all
        """
        try:
            # If no model name provided, unload all
            if model_name is None:
                if not self.current_model:
                    print("No models loaded to unload")
                    return True
                
                count = len(self.current_model)
                print(f"Unloading {count} model(s)...")
                
                for name in list(self.current_model.keys()):
                    self._unload_single_model(name)
                
                self.current_model.clear()
                print(f"✅ Unloaded {count} model(s)")
                return True
            
            # Unload specific model
            if model_name not in self.current_model:
                raise ValueError(f"Model '{model_name}' not loaded. Available: {list(self.current_model.keys())}")
            
            return self._unload_single_model(model_name)
            
        except Exception as e:
            print(f"❌ Error unloading model: {e}")
            return False

    def _unload_single_model(self, model_name: str) -> bool:
        """
        Internal method to unload a single model.
        
        Args:
            model_name: Name of the model to unload
        
        Returns:
            bool: True if unloaded successfully
        """
        try:
            print(f"📤 Unloading model: {model_name}")
            
            # Get the model instance
            model = self.current_model.get(model_name)
            
            if model:
                # Close any open resources
                if hasattr(model, 'close'):
                    try:
                        model.close()
                    except:
                        pass
                
                # Delete the reference
                del self.current_model[model_name]
                print(f"   ✅ Model reference deleted")
            
      
            gc.collect()
            print(f"   🧹 Garbage collection ran")
            
            # Try to free memory (Linux only)
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
                print(f"   💾 Memory trimmed")
            except:
                pass  # Not available on all platforms
            
            print(f"✅ Model '{model_name}' unloaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error unloading model '{model_name}': {e}")
            return False

    def unload_all_models(self):
        """
        Unload all loaded models.
        
        Returns:
            int: Number of models unloaded
        """
        count = len(self.current_model)
        if count == 0:
            print("No models loaded")
            return 0
        
        print(f"Unloading all {count} model(s)...")
        
        for model_name in list(self.current_model.keys()):
            self._unload_single_model(model_name)
        
        self.current_model.clear()
        print(f"✅ Unloaded {count} model(s)")
        return count

    def get_memory_usage(self):
        """
        Get current memory usage information.
        
        Returns:
            dict: Memory usage stats
        """
        
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "models_loaded": len(self.current_model),
            "model_names": list(self.current_model.keys())
        }
    
    # ========== FIXED WORKER ==========
    async def _worker(self):
        """Background worker - processes queue"""
        print("👷 Worker started, waiting for tasks...")
        
        while self._running: 
            try:
       
                task = await asyncio.wait_for(self.queue.get(), timeout=60.0)
                
                task_id = task["task_id"]
                prompt = task["prompt"]
                future = task["future"]
                
                print(f"👷 Processing: {task_id}")

                if not self.current_model:
                    future.set_exception(ValueError("No model loaded"))
                    self.queue.task_done()
                    continue
         
                model = list(self.current_model.values())[0]
                response = await asyncio.to_thread(model.invoke, prompt)

                result = response.content if hasattr(response, 'content') else str(response)

                future.set_result(result)
                self.queue.task_done() 
                print(f"✅ Completed: {task_id}")
                self.queue.task_done() 
            except asyncio.TimeoutError:
                print(f"Timeout on task {task_id}")
                continue
            except asyncio.CancelledError:
                print("Worker cancelled")
                break
            except Exception as e:
                print(f"❌ Worker error: {e}")
                if 'future' in locals():
                    future.set_exception(e)
                if 'task' in locals():
                    self.queue.task_done()
    
    # ========== START METHOD ==========
    async def start(self):
        """Start the worker"""
        if self._running:
            print("Worker already running")
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        print("🚀 Service started")
    
    # ========== STOP METHOD ==========
    async def stop(self):
        """Stop the worker gracefully"""
        if not self._running:
            return
        
        self._running = False
   
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
 
        for task_id, future in self.futures.items():
            if not future.done():
                future.set_exception(Exception("Service stopped"))
        self.futures.clear()
        
        print("🛑 Service stopped")
    
    # ========== CHAT METHOD ==========
    async def chat_llm(self, chat: str) -> str:
        """Send a message to the LLM"""
        try:
            if not self.current_model:
                raise ValueError("No model loaded. Call _load_model() first.")
            
            if not self._running:
                raise RuntimeError("Service not started. Call start() first.")
            
            task_id = str(uuid4())[:12]
            future = asyncio.Future()
            
            self.futures[task_id] = future
            
            await self.queue.put({
                "task_id": task_id,
                "prompt": chat,
                "future": future
            })
            
            print(f"📝 [{task_id}] Queued (position: {self.queue.qsize()})")
            
            return await asyncio.wait_for(future, timeout=60.0)

            
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            raise

# ========== CORRECT USAGE ==========
async def main():
    print("="*60)
    print("ASYNC LLM SERVICE TEST")
    print("="*60)
    
    # 1. Create service
    llm = AsyncLLM()
    
    # 2. Load model (MUST AWAIT!)
    await llm._load_model(model_name="glm-4-9b-chat-IQ4_XS.gguf")
    
    # 3. Start the worker
    await llm.start()
    
    # 4. Chat with the LLM
    print("\n" + "="*60)
    print("CHATTING...")
    print("="*60)
    
    response = await llm.chat_llm("What is Python?")
    print(f"Response: {response[:200]}...")
    
    # 5. Multiple users
    print("\n" + "="*60)
    print("3 USERS CONCURRENTLY")
    print("="*60)
    
    results = await asyncio.gather(
        llm.chat_llm("Tell me a joke"),
        llm.chat_llm("Explain async programming"),
        llm.chat_llm("Write 500 lin eessay on pakistani village life"),
        llm.chat_llm("What is machine learning?")
    )
    
    for i, r in enumerate(results):
        print(f"\nUser {i+1}: {r[:100]}...")
    
    # 6. Stop service
    await llm.stop()
    
    # 7. Unload model
    llm.unload_model("glm-4-9b-chat-IQ4_XS.gguf")
    
    print("\n✅ All done!")

if __name__ == "__main__":
    asyncio.run(main())
