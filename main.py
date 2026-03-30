import llama_cpp
from pydantic import BaseModel, Field,field_validator,ConfigDict,ValidationError
from pydantic_settings import BaseSettings
import gc
import ctypes
import os
from langchain.embeddings import Embeddings
from langchain_community.retrievers import WikipediaRetriever
from uuid import uuid4
import asyncio
from typing import Dict,List,Union,Any,Tuple,Type,TypeVar,Optional
import uvloop
from queue import Queue
import psutil
import json
import shutil
import re
from enum import Enum
class EcharrParsers:
    def __init__(self,**kwargs):
        self.cj: Dict = {}
        cdir = os.getcwd()
        # Fix: Initialize as an empty list, not the type 'List[str]'
        self.all_charts_list: List[str] = [] 
        
        fp = os.path.join(cdir, "Charts.json")
        if not os.path.exists(fp):
            raise ValueError(f"Error file not found Charts.json")

        with open(fp, 'r') as f:
            # Note: parse_int=3 is unusual; standard json.load() is safer
            self.cj = json.load(f)
    
        self.all_charts_list = list(self.cj.get("templates", {}).keys())
        self.DEC = Enum('DEC', {key.upper(): key.lower() for key in self.all_charts_list})
        self.temperature = kwargs.get("temperature", 0.1)
        self.top_p = kwargs.get("top_p", 0.9)
        self.top_k = kwargs.get("top_k", 30)
        self.streaming = kwargs.get("streaming", False)
        self.repeat_penalty = kwargs.get("repeat_penalty",None)
        self.n_predict = kwargs.get("n_predict", kwargs.get("n_predict", 2048))
        self.n_batch = kwargs.get("n_batch", 128)
        self.n_ctx = kwargs.get("n_ctx", 2048)
        self.n_threads = kwargs.get("n_threads", 6)
        self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)
        self.verbose = kwargs.get("verbose", False)
        self.stops = kwargs.get("stop", ["<|endoftext|>", "<|im_end|>"])
    def get_tool_metadata(self,capitalise=False) -> str:
        """
        Returns a string describing available charts.
        This is what you 'feed' the LLM so it knows its powers.
        """
        metadata = []
        templates = self.cj.get("templates", {})
        for name, info in templates.items():
            desc = info.get("description", "No description")
            metadata.append(f"- {name}: {desc}")
        return "\n".join(metadata)

    def get_enum_list(self, capitalise: bool = False):
        """
        Returns list of chart names.
        If capitalise=True: returns uppercase names (enum member names)
        If capitalise=False: returns lowercase values (original keys)
        """
        if capitalise:
            # Return enum member names (already uppercase)
            return [member.name for member in self.DEC]
        else:
            # Return enum member values (original lowercase keys)
            return [member.value for member in self.DEC]
        
    def get_template_placeholders(self, chart_name:'DEC') -> List[str]:
        """Returns the specific blanks the LLM needs to fill"""
        template = self.cj.get("templates", {}).get(chart_name)
        return template.get("placeholders", []) if template else []
    
    def get_full_chart_template(self, chart_name: 'DEC') -> Dict[str, Any]:
        """
        Returns the ENTIRE JSON block for a specific chart.
        """
        # ✅ Correct: Use chart_name.value to get the string key
        chart_key = chart_name.value if hasattr(chart_name, 'value') else chart_name
        template = self.cj.get("templates", {}).get(chart_key)
        
        if not template:
            # ✅ Correct formatting
            raise ValueError(f"Chart '{chart_key}' not found. Available charts: {list(self.cj.get('templates', {}).keys())}")
        
        return template
    
    def get_dynamic_prompt(self, user_input: str, selected_chart:'DEC'):
            """
            Injected Prompt: 
            Only tells the LLM about the ONE chart it needs to fill.
            """
            # 1. Get the specific placeholders for the chosen chart
            placeholders = self.get_template_placeholders(selected_chart)
            
            # 2. Create a string representation of the keys the LLM must return
            # e.g., '"xAxisData": [...], "seriesData": [...]'
            placeholder_str = ", ".join([f'"{p}": [...]' for p in placeholders])

            return f"""
            You are a Data Extraction Bot for the chart: {selected_chart}.
            
            REQUIRED KEYS:
            You must return a JSON object with exactly these keys: {placeholders}
            
            USER DATA:
            {user_input}
            
            STRICT OUTPUT FORMAT:
            {{
                "chart_name": "{selected_chart}",
                "data": {{ {placeholder_str} }}
            }}
            """
    def get_full_injection_prompt(self, user_input: str, selected_chart: 'DEC'):
                full_template = self.get_full_chart_template(selected_chart)
                
                return f"""
                You are a JSON Engineer.
                
                TEMPLATE:
                {json.dumps(full_template["options"])}
                
                USER DATA:
                {user_input}
                
                CRITICAL RULES:
                1. Return the FULL JSON exactly as provided in the TEMPLATE.
                2. ONLY replace the "{{{{placeholders}}}}" with values from User Data.
                3. DO NOT add new keys like 'title', 'tooltip', 'legend', or 'color' if they are not in the template.
                4. If data for a placeholder is missing, return an empty list [].
                """
    
    def get_designer_prompt(self, user_input: str, selected_chart: 'DEC'):
            full_template = self.get_full_chart_template(selected_chart)
            
            return f"""
            You are a Senior Chart Designer for an ERP system.
            
            TEMPLATE:
            {json.dumps(full_template["options"])}
            
            USER DATA: {user_input}
            
            DESIGN RULES:
            1. Fill the "{{{{placeholders}}}}" with the correct data.
            2. If the user mentions a specific TITLE, add/update the "title": {{"text": "..."}} key.
            3. If the user mentions a COLOR (e.g., 'make it red'), update the "itemStyle": {{"color": "..."}} inside the series.
            4. If the user asks for a 'smooth' line, set "smooth": true in the series.
            
            STRICT REQUIREMENT: 
            Only modify keys that exist in the ECharts standard. Do not invent custom keys.
            """
class Runnable:
    def __init__(self, func):
        self.func = func
    
    async def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    
    # ADD THESE METHODS:
    def invoke(self, input_data):
        """Sync invoke"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input_data))
        else:
            return loop.run_until_complete(self.ainvoke(input_data))
    
    async def ainvoke(self, input_data):
        """Async invoke"""
        return await self(input_data)
    
    def __or__(self, other):
        async def chained(value):
            result = await self.ainvoke(value)  # CHANGE: use ainvoke
            if hasattr(other, 'ainvoke'):
                return await other.ainvoke(result)
            elif hasattr(other, 'invoke'):
                return other.invoke(result)
            elif callable(other):
                return await other(result) if asyncio.iscoroutinefunction(other) else other(result)
            return result
        return Runnable(chained)

 

class StrOutputParser:
    def parse(self, response):
        # CHANGE: Return raw string, don't try to parse JSON
        if isinstance(response, str):
            return response.strip()  # Just strip, don't json.loads
        
        if isinstance(response, dict):
            if "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0].get("text", "").strip()
            if "text" in response:
                return response["text"].strip()
            return json.dumps(response)  # Convert dict to string
        
        return str(response).strip()
    
    # ADD these methods for consistency
    def invoke(self, input_data):
        return self.parse(input_data)
    
    async def ainvoke(self, input_data):
        return self.parse(input_data)
    
    def __call__(self, response):
        return self.parse(response)
    
    def __or__(self, other):
        def chained(value):
            parsed = self.parse(value)
            return other(parsed) if callable(other) else parsed
        return Runnable(chained)
    

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

class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str] = None,
                 partial_variables: Dict[str, Any] = None,
                 validate_template: bool = True):
        self.template = template
        self.partial_variables = partial_variables or {}
        self.input_variables = input_variables or self._extract_variables(template)
        self.validate_template = validate_template
        if validate_template:
            self._validate()

    def _extract_variables(self, template: str) -> List[str]:
        
        return re.findall(r'\{([^{}]+)\}', template)

    def _validate(self):
        """Check that all input variables are accounted for."""
        pass

    def format(self, **kwargs) -> str:
        """Combine partial and passed variables."""
        all_vars = {**self.partial_variables, **kwargs}
        if self.validate_template:
            missing = set(self.input_variables) - set(all_vars)
            if missing:
                raise ValueError(f"Missing variables: {missing}")
        return self.template.format(**all_vars)

    def partial(self, **kwargs):
        """Return a new template with pre‑filled variables."""
        new_partials = {**self.partial_variables, **kwargs}
        return PromptTemplate(
            template=self.template,
            partial_variables=new_partials,
            validate_template=self.validate_template
        )

    def invoke(self, input_data):
        """Sync invoke - format prompt"""
        if isinstance(input_data, dict):
            return self.format(**input_data)
        return self.format(input=input_data)
    
    async def ainvoke(self, input_data):
        """Async invoke - format prompt"""
        return self.invoke(input_data)
    
    def __or__(self, other):
        async def chained(input_data):
            if isinstance(input_data, dict):
                prompt = self.format(**input_data)
            else:
                prompt = self.format(input=input_data)
            
            if hasattr(other, 'ainvoke'):
                return await other.ainvoke(prompt)
            elif hasattr(other, 'invoke'):
                return other.invoke(prompt)
            elif callable(other):
                return await other(prompt) if asyncio.iscoroutinefunction(other) else other(prompt)
            return prompt
        return Runnable(chained)
    
    @classmethod
    def from_template(cls, template: str):
        """Convenience constructor."""
        return cls(template)

class ChatPromptTemplate:

    def __init__(self, messages: List[Tuple[str, str]]):
        """# Store the messages for later formatting"""
        self.messages = messages

    def format_prompt(self, **kwargs):
        """Replace placeholders and return prompt string"""
        result = ""
        
        for role, template in self.messages:
            # Replace placeholders like {name} with values
            content = template.format(**kwargs)
            
            # Convert role to format LLM understands
            if role == "system":
                result += f"[SYSTEM] {content}\n"
            elif role == "human":
                result += f"[USER] {content}\n"
            elif role == "assistant":
                result += f"[ASSISTANT] {content}\n"
        
        # Add final marker for LLM to start generating
        result += "[ASSISTANT] "
        return result
    def __call__(self, **kwargs):
        """Allow: template(topic="Python")"""
        return self.format_prompt(**kwargs)
        
    def __or__(self, other):
        async def chained(input_data):
            if isinstance(input_data, dict):
                prompt = self.format_prompt(**input_data)
            else:
                prompt = self.format_prompt(input=input_data)
            return await other(prompt)   # <-- always await
        return Runnable(chained)
    
    @classmethod
    def from_messages(cls, messages:List[Tuple[str,str]]):
        """Create template from message list"""
        return cls(messages)

    @classmethod
    def from_template(cls, template: str):
        """Create simple template from one string"""
        return cls([("human", template)])
class AsyncLLM:
 
    def __init__(self):
       
        self.futures:Dict[str,asyncio.Future]={}    
        self.current_model:Dict[str,llama_cpp.Llama]={}
        self.setting=Settings()
        self.path=self.setting.MODEL_PATH
        self.loop=uvloop.new_event_loop()
        self.async_lock=asyncio.Lock()
        self.queue=asyncio.Queue(maxsize=10)
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
            avail = psutil.virtual_memory().available / (1024**2)
            if avail < 3000:
                raise MemoryError(f"Need 3GB+ free, have {avail:.0f}MB")
            # Get optional parameters with defaults
            self.temperature = kwargs.get("temperature", 0.1)
            self.top_p = kwargs.get("top_p", 0.9)
            self.top_k = kwargs.get("top_k", 30)
            self.streaming = kwargs.get("streaming", False)
            self.repeat_penalty = kwargs.get("repeat_penalty",None)
            self.n_predict = kwargs.get("n_predict", kwargs.get("n_predict", 2048))
            self.n_batch = kwargs.get("n_batch", 128)
            self.n_ctx = kwargs.get("n_ctx", 2048)
            self.n_threads = kwargs.get("n_threads", 6)
            self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)
            self.verbose = kwargs.get("verbose", False)
            self.stops = kwargs.get("stop", ["<|endoftext|>", "<|im_end|>"])
            print("max_token:::")
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
            print(f"   Temperature: {self.temperature}")
            print(f"   Max tokens: {self.n_predict}")
            print(f"   GPU layers: {self.n_gpu_layers}")
       
            # Load model in thread to avoid blocking
            llm = await asyncio.to_thread(
                llama_cpp.Llama,
                model_path=fmp,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                streaming=self.streaming,
                repeat_penalty=self.repeat_penalty,
                n_predict=self.n_predict,
                n_batch=self.n_batch,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                stop=self.stops,
            )
            
            # Store loaded model
            self.current_model[model_name] = llm
            
            print(f"✅ Model '{model_name}' loaded successfully!")
            
            return llm
            
        except FileNotFoundError as e:
            raise ValueError(f"Model not found: {e}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    # def run_task(self,tname,**kw):
    #     try:
    #         if callable(tname):
    #             res= self.loop.run_until_complete(tname(**kw))
    #             return res
    #     except Exception as e:
    #         raise ValueError(f"Error run task due to {e}")

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
        """Background worker - returns RAW response"""
        while self._running:
            task = None
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=60.0)
                task_id = task["task_id"]
                prompt = task["prompt"]
                future = task["future"]
                
                if not self.current_model:
                    future.set_exception(ValueError("No model loaded"))
                    continue
                
                model = list(self.current_model.values())[0]
                
                # FIX: Pass explicit generation parameters
                raw_response = await asyncio.to_thread(
                    model, 
                    prompt, 
                    max_tokens=2048,        # ← ADD THIS
                    temperature=0.7,        # ← ADD THIS (or get from config)
                    stop=["<|im_end|>", "User:", "Assistant:", "Human:", "</s>"]  # ← ADD stop tokens
                )
                
                print("raw_response:::", raw_response)
                future.set_result(raw_response)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if task and 'future' in locals() and not future.done():
                    future.set_exception(e)
            finally:
                if task:
                    self.queue.task_done()
    # async def _worker(self):
    #     """Background worker - processes queue"""
    #     print("👷 Worker started, waiting for tasks...")
        
    #     while self._running: 
    #         try:
    #             parser=StrOutputParser()
    #             task = await asyncio.wait_for(self.queue.get(), timeout=60.0)
    #             if psutil.virtual_memory().percent > 95:
    #                 future.set_exception(MemoryError("System RAM critical"))
    #                 continue
    #             task_id = task["task_id"]
    #             prompt = task["prompt"]
    #             future = task["future"]
                
    #             print(f"👷 Processing: {task_id}")

    #             if not self.current_model:
    #                 future.set_exception(ValueError("No model loaded"))
                     
    #                 continue
         
    #             model = list(self.current_model.values())[0]
    #             response = await asyncio.to_thread(model, prompt)
    #             pr=parser(response=response)

    #             result = response.content if hasattr(response, 'content') else str(response)
                 
    #             future.set_result(response)
                
    #             print(f"✅ Completed: {task_id}")
              
    #         except asyncio.TimeoutError:
    #             print(f"Timeout on task {task_id}")
    #             continue
    #         except asyncio.CancelledError:
    #             print("Worker cancelled")
    #             break
    #         except Exception as e:
    #             print(f"❌ Worker error: {e}")
    #             # if 'future' in locals():
    #             #     future.set_exception(e)
    #             # if 'task' in locals():
    #             #     self.queue.task_done()
    #         finally:
    #             self.queue.task_done()
    #             gc.collect()
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
 
        # Clear pending futures from queue (not self.futures dict)
        while not self.queue.empty():
            try:
                task = self.queue.get_nowait()
                future = task.get("future")
                if future and not future.done():
                    future.set_exception(Exception("Service stopped"))
            except:
                break
        
        print("🛑 Service stopped")
    
    # ========== CHAT METHOD ==========
    async def chat_llm(self, chat: str) -> str:
        """Send a message to the LLM"""
        try:
            task_id=None
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
        finally:
            self.futures.pop(task_id, None) 
            print(f"🗑️ [{task_id}] Future cleaned from memory")
    async def ainvoke(self, input: Union[str, Dict]) -> str:
        """Async invoke for chaining"""
        if isinstance(input, dict):
            prompt = input.get("input", str(input))
        else:
            prompt = str(input)
        return await self.chat_llm(prompt)
    
    def invoke(self, input: Union[str, Dict]) -> str:
        """Sync invoke - runs LLM"""
        if isinstance(input, dict):
            prompt = input.get("input", str(input))
        else:
            prompt = str(input)
        
        try:
            # Try to run in existing loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create new one
            return asyncio.run(self.chat_llm(prompt))
        else:
            # Already in async context
            return loop.run_until_complete(self.chat_llm(prompt))
    
    def __or__(self, other):
        async def chained(value):
            response = await self.ainvoke(value)  # Use ainvoke
            if hasattr(other, 'ainvoke'):
                return await other.ainvoke(response)
            elif hasattr(other, 'invoke'):
                return other.invoke(response)
            elif callable(other):
                return await other(response) if asyncio.iscoroutinefunction(other) else other(response)
            return response
        return Runnable(chained)
    
    def __ror__(self, other):
        """prompt | llm"""
        if callable(other):
            async def chained(value):
                processed = other(value)
                return await self.ainvoke(processed)
            return chained
        return self
    async def __call__(self, input: Union[str, Dict]) -> str:
        """Make the instance callable directly"""
        return await self.ainvoke(input)
