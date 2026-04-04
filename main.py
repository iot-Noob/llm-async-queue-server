import llama_cpp
from pydantic import BaseModel, Field,field_validator,ConfigDict,ValidationError
from pydantic import ValidationError as PydanticValidationError
from pydantic_settings import BaseSettings
import gc
import ctypes
import os
from langchain.embeddings import Embeddings
from langchain_community.retrievers import WikipediaRetriever
from uuid import uuid4
import asyncio
from typing import Dict, List, Union, Any, Optional, Tuple, Type, TypeVar, Generic
from queue import Queue
import psutil
import json
import shutil
import re
from enum import Enum
from dataclasses import dataclass,field
import time
import logging

logger = logging.getLogger(__name__)

class EcharrParsers:
    def __init__(self):
        self.cj: Dict = {}
        cdir = os.getcwd()
        # Fix: Initialize as an empty list, not the type 'List[str]'
        self.all_charts_list: List[str] = [] 
        fp = os.path.join(cdir, "Charts.json")
        if not os.path.exists(fp):
            raise ValueError(f"Error file not found Charts.json")
        try:
            with open(fp, 'r') as f:
                self.cj = json.load(f)
            self.all_charts_list = list(self.cj.get("templates", {}).keys())
        except json.JSONDecodeError as je:
            raise ValueError(f"Error invalid json  in echart{je}")
            # 2. Extract the keys into the list
        
        # 3. Create the Enum LAST using the populated list
        # Using the Functional API (Enum) or the type() method you liked:
        self.DEC = Enum('DEC', {key.upper(): key for key in self.all_charts_list})
        """
        Returns a string describing available charts.
        This is what you 'feed' the LLM so it knows its powers.
        """
        metadata = []
        templates = self.cj.get("templates", {})
        for name, info in templates.items():
            desc = info.get("description", "No description")
            metadata.append(f"- {name}: {desc}")
       

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
    
    # def __or__(self, other):
    #     async def chained(value):
    #         result = await self.ainvoke(value)  # CHANGE: use ainvoke
    #         if hasattr(other, 'ainvoke'):
    #             return await other.ainvoke(result)
    #         elif hasattr(other, 'invoke'):
    #             return other.invoke(result)
    #         elif callable(other):
    #             return await other(result) if asyncio.iscoroutinefunction(other) else other(result)
    #         return result
    #     return Runnable(chained)
    def __or__(self, other):
        async def chained(value):
            result = await self.ainvoke(value)
            if asyncio.iscoroutinefunction(other):
                return await other(result)
            elif callable(other):
                return other(result)
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
    
    # def __or__(self, other):
    #     async def chained(input_data):
    #         if isinstance(input_data, dict):
    #             prompt = self.format(**input_data)
    #         else:
    #             prompt = self.format(input=input_data)
            
    #         if hasattr(other, 'ainvoke'):
    #             return await other.ainvoke(prompt)
    #         elif hasattr(other, 'invoke'):
    #             return other.invoke(prompt)
    #         elif callable(other):
    #             return await other(prompt) if asyncio.iscoroutinefunction(other) else other(prompt)
    #         return prompt
    #     return Runnable(chained)

    def __or__(self, other):
        async def chained(input_data):
            if isinstance(input_data, dict):
                # Format prompt
                prompt = self.format(**input_data)
                # Pass through other keys (like chat_id, timeout)
                result = {**input_data, "input": prompt}
            else:
                prompt = self.format(input=input_data)
                result = {"input": prompt}
            
            if hasattr(other, 'ainvoke'):
                return await other.ainvoke(result)
            elif hasattr(other, 'invoke'):
                return other.invoke(result)
            elif callable(other):
                return await other(result) if asyncio.iscoroutinefunction(other) else other(result)
            return result
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
        
    # def __or__(self, other):
    #     async def chained(input_data):
    #         if isinstance(input_data, dict):
    #             prompt = self.format_prompt(**input_data)
    #         else:
    #             prompt = self.format_prompt(input=input_data)
    #         return await other(prompt)   # <-- always await
    #     return Runnable(chained)

    def __or__(self, other):
        """Chain with metadata preservation (keeps chat_id, timeout, etc.)"""
        async def chained(input_data):
            if isinstance(input_data, dict):
                # Format the prompt using all values from the dict
                prompt = self.format_prompt(**input_data)
                # ✅ PRESERVE all original data + add formatted prompt
                result = {**input_data, "input": prompt}
            else:
                # Input is a string, treat as user input
                prompt = self.format_prompt(input=input_data)
                result = {"input": prompt}
            
            # Pass to next component with metadata preserved
            if hasattr(other, 'ainvoke'):
                return await other.ainvoke(result)
            elif hasattr(other, 'invoke'):
                return other.invoke(result)
            elif callable(other):
                return await other(result) if asyncio.iscoroutinefunction(other) else other(result)
            return result
        return Runnable(chained)
    @classmethod
    def from_messages(cls, messages:List[Tuple[str,str]]):
        """Create template from message list"""
        return cls(messages)

    @classmethod
    def from_template(cls, template: str):
        """Create simple template from one string"""
        return cls([("human", template)])

T = TypeVar('T', bound=BaseModel)

class JsonOutputParser:
    """
    LangChain-style JSON Output Parser.
    Extracts JSON from LLM responses, handles markdown code blocks, and returns dict.
    """
    
    def __init__(self, pydantic_object: Optional[Type[BaseModel]] = None):
        """
        Args:
            pydantic_object: Optional Pydantic model for validation.
                            If provided, returns validated model instead of dict.
        """
        self.pydantic_object = pydantic_object
    
    def parse(self, response):
        """Parse raw LLM response into JSON/dict"""
        import re
        
        if isinstance(response, str):
            text = response.strip()
            
            # Remove markdown code blocks - same regex pattern LangChain uses [citation:1]
            # Pattern: ```json ... ``` or ``` ... ```
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            
            # Try to find JSON directly if no markdown blocks
            else:
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    text = json_match.group()
            
            # Parse JSON
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {e}\nResponse: {text[:200]}")
            
            # Validate with Pydantic if provided [citation:5][citation:8]
            if self.pydantic_object:
                try:
                    return self.pydantic_object(**parsed)
                except PydanticValidationError as e:
                    raise ValueError(f"Pydantic validation failed: {e}")
            return parsed
        
        elif isinstance(response, dict):
            if self.pydantic_object:
                return self.pydantic_object(**response)
            return response
        
        return response
    
    def get_format_instructions(self) -> str:
        """Return formatting instructions for the LLM - similar to LangChain's [citation:5]"""
        if self.pydantic_object:
            schema = self.pydantic_object.model_json_schema()
            return f"""Respond with a valid JSON object matching this schema:
{json.dumps(schema, indent=2)}"""
        return "Respond with a valid JSON object. Do not include any explanatory text."
    
    def invoke(self, input_data):
        return self.parse(input_data)
    
    async def ainvoke(self, input_data):
        return self.parse(input_data)
    
    def __call__(self, response):
        return self.parse(response)

@dataclass(slots=True)  # ✅ slots=True reduces memory usage (~40% less)
class Document:
    """
    Optimized LangChain-style Document class for RAG pipelines.
    
    Performance Optimizations:
    - __slots__ reduces memory footprint by ~40%
    - Lazy JSON serialization (only when needed)
    - Efficient string concatenation with join
    - Early returns for common operations
    - Minimal overhead for metadata access
    """
    
    # Core fields with type hints
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    # Class-level constants
    _TRUNCATE_SUFFIX = "..."
    _MAX_PREVIEW_LEN = 50

    def __post_init__(self):
        """Auto-generate ID if not provided (minimal overhead)"""
        if self.id is None:
            # ✅ Use faster UUID generation for small IDs
            self.id = uuid4().hex[:8]
    
    def __str__(self) -> str:
        """Fast string conversion"""
        return self.page_content
    
    def __repr__(self) -> str:
        """Optimized debug representation"""
        content = self.page_content
        if len(content) > self._MAX_PREVIEW_LEN:
            content = content[:self._MAX_PREVIEW_LEN] + self._TRUNCATE_SUFFIX
        return f"Document(id='{self.id}', page_content='{content}', metadata={self.metadata})"
    
    def __len__(self) -> int:
        """O(1) length operation"""
        return len(self.page_content)
    
    def __bool__(self) -> bool:
        """Truthy if has content"""
        return bool(self.page_content)
    
    def __eq__(self, other: object) -> bool:
        """Equality check by ID"""
        if not isinstance(other, Document):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID for dict/set usage"""
        return hash(self.id)
    
    def __add__(self, other: 'Document') -> 'Document':
        """Fast document combination"""
        if not isinstance(other, Document):
            return NotImplemented
        
        # ✅ Use efficient string concatenation
        return Document(
            page_content=f"{self.page_content}\n{other.page_content}",
            metadata={
                "sources": [self.metadata, other.metadata],
                "original_ids": [self.id, other.id]
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with minimal copying"""
        return {
            "id": self.id,
            "page_content": self.page_content,
            "metadata": dict(self.metadata)  # Copy to prevent modification
        }
    
    def to_json(self, indent: int = None, compact: bool = False) -> str:
        """
        Convert to JSON with options for performance.
        
        Args:
            indent: Pretty print indent (None = compact)
            compact: Force compact JSON even with indent (ignores indent)
        """
        if compact or indent is None:
            # ✅ Fastest JSON generation (no whitespace)
            return json.dumps(self.to_dict(), separators=(',', ':'), ensure_ascii=False)
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Fast dict creation with defaults"""
        return cls(
            page_content=data.get("page_content", ""),
            metadata=data.get("metadata", {}),
            id=data.get("id")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Document':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def combine(cls, documents: List['Document'], separator: str = "\n") -> 'Document':
        """
        Optimized combine using list comprehension and efficient join.
        """
        if not documents:
            raise ValueError("Cannot combine empty list of documents")
        
        # ✅ Single pass through documents for content and metadata
        contents = []
        original_ids = []
        sources = []
        
        for doc in documents:
            contents.append(doc.page_content)
            original_ids.append(doc.id)
            sources.append(doc.metadata)
        
        # ✅ Efficient string joining
        combined_content = separator.join(contents)
        
        return cls(
            page_content=combined_content,
            metadata={
                "combined": True,
                "original_count": len(documents),
                "original_ids": original_ids,
                "sources": sources
            }
        )
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Fast metadata access with default"""
        return self.metadata.get(key, default)
    
    def has_metadata(self, key: str) -> bool:
        """Fast metadata existence check"""
        return key in self.metadata
    
    def add_metadata(self, key: str, value: Any) -> 'Document':
        """
        Add metadata and return new document (immutable).
        Optimized to avoid copying large metadata dict when possible.
        """
        # ✅ Create new dict with single copy operation
        new_metadata = {**self.metadata, key: value}
        return Document(
            page_content=self.page_content,
            metadata=new_metadata,
            id=self.id
        )
    
    def truncate(self, max_chars: int = 500, suffix: str = "...") -> 'Document':
        """
        Optimized truncation with early return.
        """
        # ✅ Early return if no truncation needed
        if len(self.page_content) <= max_chars:
            return self
        
        # ✅ Calculate truncation point
        trunc_len = max_chars - len(suffix)
        if trunc_len <= 0:
            # If max_chars is too small, return just the suffix
            truncated_content = suffix
        else:
            truncated_content = self.page_content[:trunc_len] + suffix
        
        # ✅ Only add truncation metadata if changed
        return Document(
            page_content=truncated_content,
            metadata={
                **self.metadata,
                "truncated": True,
                "original_length": len(self.page_content),
                "truncated_length": max_chars
            },
            id=self.id
        )
    
    def copy(self, **kwargs) -> 'Document':
        """
        Create a copy with optional field updates.
        Optimized for common use cases.
        """
        return Document(
            page_content=kwargs.get("page_content", self.page_content),
            metadata=kwargs.get("metadata", self.metadata.copy()),
            id=kwargs.get("id", self.id)
        )
    
    def batch_to_dicts(self, documents: List['Document']) -> List[Dict[str, Any]]:
        """Convert multiple documents to dicts efficiently"""
        return [doc.to_dict() for doc in documents]
    
    @staticmethod
    def batch_from_dicts(dicts: List[Dict[str, Any]]) -> List['Document']:
        """Create multiple documents from dicts efficiently"""
        return [Document.from_dict(d) for d in dicts]
 
class PydanticOutputParser(JsonOutputParser, Generic[T]):
    """
    LangChain-style Pydantic Output Parser.
    Parses LLM output directly into a Pydantic model [citation:5].
    """
    
    def __init__(self, pydantic_object: Type[T]):
        super().__init__(pydantic_object=pydantic_object)
        self.pydantic_object = pydantic_object
    
    def parse(self, response):
        """Parse and return validated Pydantic model"""
        return super().parse(response)



class AsyncLLM:
    class Config:
        """Central configuration for AsyncLLM."""
        def __init__(self, **kwargs):
            self.available_ram = kwargs.get("available_ram", 3000)          # MB
            self.queue_maxsize = kwargs.get("queue_maxsize", 10)
            self.default_timeout = kwargs.get("default_timeout", 160.0)   # seconds
            self.graceful_shutdown = kwargs.get("graceful_shutdown", True) # wait for in‑flight requests
            self.enable_metrics = kwargs.get("enable_metrics", True)
            self.enable_cancellation = kwargs.get("enable_cancellation", True)
            self.reject_on_full_queue = kwargs.get("reject_on_full_queue", True)

    def __init__(self, **kwargs):
        # Load configuration
        self.config = self.Config(**kwargs)

        # Setup logging (you may want to configure level/handlers externally)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        self._metrics_lock=asyncio.Lock()
        self.setting = Settings()
        self.model_path = kwargs.get("model_path")
        self._worker_exception_callback = kwargs.get("worker_exception_callback", None)
        self._auto_restart_worker = kwargs.get("auto_restart_worker", False)
        self._stream_cancel_flags: Dict[str, asyncio.Event] = {}
        self.futures: Dict[str, asyncio.Future] = {}
        self._active_streams: set = set()
        self.current_model: Dict[str, llama_cpp.Llama] = {}
        self._matric_lock=asyncio.Lock()
        # Metrics
        self._metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "streaming_requests": 0,
            "completed_requests": 0,
        }

        if self.model_path:
            self.path = self.model_path
        else:
            try:
                self.path = self.setting.MODEL_PATH
            except ValidationError:
                raise ValueError("MODEL_PATH not set. Provide it via .env or pass model_path argument.")

        self.queue = asyncio.Queue(maxsize=self.config.queue_maxsize)
        self._running_event = asyncio.Event()
        self._worker_task = None
        self._shutdown_event = asyncio.Event()
        self.lock=asyncio.Lock()
        self._future_lock=asyncio.Lock()
        # Default generation parameters (can be overridden per request)
        self.temperature = kwargs.get("temperature", 0.1)
        self.top_p = kwargs.get("top_p", 0.9)
        self.top_k = kwargs.get("top_k", 30)
        self.streaming = kwargs.get("streaming", False)
        self.repeat_penalty = kwargs.get("repeat_penalty", None)
        self.n_predict = kwargs.get("n_predict", 2048)
        self.n_batch = kwargs.get("n_batch", 128)
        self.n_ctx = kwargs.get("n_ctx", 2048)
        self.n_threads = kwargs.get("n_threads", 6)
        self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)
        self.verbose = kwargs.get("verbose", False)
        self.stops = kwargs.get("stop", ["<|endoftext|>", "<|im_end|>"])
        self.output_parser = kwargs.get("output_parser", StrOutputParser())
        self._stream_lock = asyncio.Lock()
        self._stream_tasks: Dict[str, asyncio.Task] = {}
    # ------------------------ Health & Metrics ------------------------
    async def is_healthy(self) -> bool:
        async with self.lock:
            return self._running_event.is_set() and len(self.current_model) > 0

    # async def get_metrics(self) -> Dict[str, Any]:
    #     """Return current metrics."""
    #     async with self.lock:
    #         metrics = self._metrics.copy()
    #         metrics["queue_size"] = self.queue.qsize()
    #         metrics["active_streams"] = len(self._active_streams)
    #         metrics["models_loaded"] = len(self.current_model)
    #         return metrics

    async def get_metrics(self) -> Dict[str, Any]:
        async with self._metrics_lock:
            metrics = self._metrics.copy()
            metrics["queue_size"] = self.queue.qsize()
            
            # These are safe to read without lock (len is atomic)
            metrics["active_streams"] = len(self._active_streams)
            metrics["models_loaded"] = len(self.current_model)
            return metrics

    # # ------------------------ Request Cancellation ------------------------
    # async def cancel_request(self, request_id: str):
    #     """Cancel a pending or in‑progress request (non‑streaming) or stop a stream."""
        
    #     if request_id in self.futures and not self.futures[request_id].done():
    #         self.futures[request_id].set_exception(asyncio.CancelledError(f"Request {request_id} cancelled"))
    #         logger.info(f"Cancelled request {request_id}")
    #     elif request_id in self._active_streams:
    #         # For streaming, we need to force the worker to stop. Simpler: remove the future and close stream.
    #         # We'll rely on the monitor task to abort.
    #         # Actually, we can add a cancellation queue. For brevity, we raise.
    #         raise NotImplementedError("Stream cancellation not implemented yet – use the monitor's future exception")
    #     else:
    #         logger.warning(f"Request {request_id} not found or already completed")
    async def cancel_request(self, request_id: str):
        async with self._future_lock:
            if request_id in self.futures and not self.futures[request_id].done():
                self.futures[request_id].set_exception(asyncio.CancelledError(f"Request {request_id} cancelled"))
                logger.info(f"Cancelled request {request_id}")
                return
        
        # ✅ Check streaming cancellation flags
        if request_id in self._stream_cancel_flags:
            self._stream_cancel_flags[request_id].set()
            logger.info(f"Cancelled stream {request_id}")
            return
        
        logger.warning(f"Request {request_id} not found")
    # ------------------------ Model Management ------------------------
    async def _get_model_in_current_dirs(self):
        try:
            return await asyncio.to_thread(self._walk_directory)
        except Exception as e:
            raise ValueError(f"Error getting models: {e}")

    def _walk_directory(self):
        all_files = []
        for root, _, files in os.walk(self.path):
            for f in files:
                if f.endswith(".gguf"):
                    bp = os.path.join(root, f)
                    all_files.append({"path": bp, "filename": f, "directory": root})
        return all_files

    async def _load_model(self, **kwargs: dict):
        async with self.lock:
            try:
                
                model_name = kwargs.get("model_name")
                if not model_name:
                    raise ValueError("model_name is required")
                avail = psutil.virtual_memory().available / (1024**2)
                if avail < self.config.available_ram:
                    raise MemoryError(f"Need {self.config.available_ram/1000}GB free, have {avail:.0f}MB")

                # Override default params for this model
                self.temperature = kwargs.get("temperature", 0.1)
                self.top_p = kwargs.get("top_p", 0.9)
                self.top_k = kwargs.get("top_k", 30)
                self.streaming = kwargs.get("streaming", False)
                self.repeat_penalty = kwargs.get("repeat_penalty", None)
                self.n_predict = kwargs.get("n_predict", 2048)
                self.n_batch = kwargs.get("n_batch", 128)
                self.n_ctx = kwargs.get("n_ctx", 2048)
                self.n_threads = kwargs.get("n_threads", 6)
                self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)
                self.verbose = kwargs.get("verbose", False)
                self.stops = kwargs.get("stop", ["<|endoftext|>", "<|im_end|>"])

                if model_name in self.current_model:
                    raise ValueError(f"Model '{model_name}' already loaded")

                fmp = os.path.join(self.path, model_name)
                if not os.path.exists(fmp):
                    models = await self._get_model_in_current_dirs()
                    found = next((m for m in models if m["filename"] == model_name), None)
                    if found:
                        fmp = found["path"]
                    else:
                        raise FileNotFoundError(f"Model '{model_name}' not found in '{self.path}'")

                logger.info(f"Loading model: {fmp} (temp={self.temperature}, max_tokens={self.n_predict}, gpu_layers={self.n_gpu_layers})")
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
                self.current_model[model_name] = llm
                logger.info(f"Model '{model_name}' loaded successfully")
                return llm
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    # ------------------------ Worker Lifecycle ------------------------
    def _worker_done_callback(self, task):
        if task.cancelled():
            logger.warning("Worker task cancelled")
            return
        exception = task.exception()
        if exception:
            logger.error(f"Worker crashed: {exception}", exc_info=True)
            if self._worker_exception_callback:
                try:
                    self._worker_exception_callback(exception)
                except Exception as e:
                    logger.error(f"Exception callback failed: {e}")
            if self._auto_restart_worker:
                logger.info("Auto‑restarting worker...")
                self._worker_task = asyncio.create_task(self._worker())
                self._worker_task.add_done_callback(self._worker_done_callback)

    async def _worker(self):
        while self._running_event.is_set():
            task = None
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=5.0)
                task_id = task["task_id"]
                prompt = task["prompt"]
                future = task["future"]
                stream = task.get("stream", False)
                stream_queue = task.get("stream_queue")
                async with self.lock:
                    if not self.current_model:
                        future.set_exception(ValueError("NoSS model loaded"))
                        continue

                    model = list(self.current_model.values())[0]

                if stream:
                    loop = asyncio.get_running_loop()
                    repeat_penalty_val = task.get("repeat_penalty", self.repeat_penalty)
                    if repeat_penalty_val is None:
                        repeat_penalty_val = 1.0
                    def generate():
                        generator = model(
                            prompt,
                            max_tokens=task.get("max_tokens", self.n_predict),
                            temperature=task.get("temperature", self.temperature),
                            top_p=task.get("top_p", self.top_p),
                            top_k=task.get("top_k", self.top_k),
                            stream=True,
                            stop=task.get("stop", self.stops),
                            repeat_penalty=repeat_penalty_val,
                        )
                        try:
                            for chunk in generator:
                                token = chunk["choices"][0]["text"]
                                asyncio.run_coroutine_threadsafe(stream_queue.put(token), loop)
                        except Exception as e:
                            asyncio.run_coroutine_threadsafe(stream_queue.put(e), loop)
                        finally:
                            asyncio.run_coroutine_threadsafe(stream_queue.put(None), loop)
                            if not future.done():
                                loop.call_soon_threadsafe(future.set_result, None)
                    await asyncio.to_thread(generate)
                else:
                    logger.info(f"Processing request {task_id}: {prompt[:50]}...")
                    repeat_penalty_val = task.get("repeat_penalty", self.repeat_penalty)
                    if repeat_penalty_val is None:
                        repeat_penalty_val = 1.0
                    raw_response = await asyncio.to_thread(
                        model,
                        prompt,
                        max_tokens=task.get("max_tokens", self.n_predict),
                        temperature=task.get("temperature", self.temperature),
                        top_p=task.get("top_p", self.top_p),
                        top_k=task.get("top_k", self.top_k),
                        stream=False,
                        stop=task.get("stop", self.stops),
                        repeat_penalty=repeat_penalty_val,
                    )
                    if isinstance(raw_response, dict) and "choices" in raw_response:
                        finish_reason = raw_response["choices"][0].get("finish_reason")
                        if finish_reason == "length":
                            logger.warning(f"Request {task_id} stopped due to max_tokens limit")
                        elif finish_reason == "stop":
                            logger.info(f"Request {task_id} stopped by stop token")
                    text_output = self._parse_raw_response(raw_response)
                    if not future.done():
                        future.set_result(text_output)
                    logger.info(f"Request {task_id} completed")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                if task and "future" in task and not task["future"].done():
                    task["future"].set_exception(e)
            finally:
                if task:
                    self.queue.task_done()

    async def start(self):
        if self._running_event.is_set():
            logger.warning("Worker already running")
            return
        
        if not self.current_model:
            raise RuntimeError("No model loaded. Call load_model() before start().")
        
        self._running_event.set()
        self._worker_task = asyncio.create_task(self._worker())
        self._worker_task.add_done_callback(self._worker_done_callback)
        logger.info("Service started")

    # async def stop(self):
    #     if not self._running:
    #         return
    #     logger.info("Stopping service...")
    #     self._running = False
    #     if self.config.graceful_shutdown:
    #         # Wait for all pending tasks to finish
    #         await self.queue.join()
    #     if self._worker_task:
    #         self._worker_task.cancel()
    #         try:
    #             await self._worker_task
    #         except asyncio.CancelledError:
    #             pass
    #         self._worker_task = None
    #     # Clear queue and fail pending futures
    #     while not self.queue.empty():
    #         try:
    #             task = self.queue.get_nowait()
    #             future = task.get("future")
    #             if future and not future.done():
    #                 future.set_exception(Exception("Service stopped"))
    #         except:
    #             break
    #     logger.info("Service stopped")

    async def stop(self):
        if not self._running_event.is_set():
            return
        
        logger.info("Stopping service...")
        for flag in self._stream_cancel_flags.values():
            flag.set()
        self._stream_cancel_flags.clear()
        # 1. Stop accepting new requests
        self._running_event.clear()
        
        # 2. Cancel all active streams
        for task_id, task in list(self._stream_tasks.items()):
            task.cancel()
            logger.info(f"Cancelled stream {task_id}")
        if self._stream_tasks:
            await asyncio.gather(*self._stream_tasks.values(), return_exceptions=True)
        self._stream_tasks.clear()
        
        # 3. Stop the worker
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._worker_task = None
        
        # 4. Fail all pending futures in the queue
        failed_count = 0
        while not self.queue.empty():
            try:
                task = self.queue.get_nowait()
                future = task.get("future")
                if future and not future.done():
                    future.set_exception(RuntimeError("Service stopped"))
                    failed_count += 1
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # 5. Also fail any futures still in self.futures (not yet queued? unlikely)
        async with self._future_lock:
            for fid, fut in self.futures.items():
                if not fut.done():
                    fut.set_exception(RuntimeError("Service stopped"))
            self.futures.clear()
        
        logger.info(f"Service stopped. Failed {failed_count} pending requests.")

    # ------------------------ Core APIs ------------------------
    async def chat_llm(self, chat: str, chat_id: Optional[str] = None, timeout: float = None, **gen_kwargs) -> str:
        if timeout is None:
            timeout = self.config.default_timeout
        
        async with self.lock:
            if not self.current_model:
                raise ValueError("No model loaded. Call _load_model() first.")
        
        if not self._running_event.is_set():
            raise RuntimeError("Service not started")
        
        task_id = chat_id or str(uuid4())[:12]
        future = asyncio.Future()
        
        async with self._future_lock:
            if task_id in self.futures:
                raise ValueError(f"Request ID '{task_id}' already in use")
            self.futures[task_id] = future
        
        # ✅ Update metrics with proper lock
        async with self._metrics_lock:
            self._metrics["total_requests"] += 1
        
        # Build task_params BEFORE queue put
        task_params = {
            "task_id": task_id,
            "prompt": chat,
            "future": future,
            "timestamp": time.time(),
            "stream": False,
        }
        allowed_params = ["temperature", "top_p", "top_k", "max_tokens", "stop", "repeat_penalty"]
        for p in allowed_params:
            if p in gen_kwargs:
                task_params[p] = gen_kwargs[p]
        
        # ✅ Queue put with proper full handling
        try:
            if self.config.reject_on_full_queue:
                self.queue.put_nowait(task_params)
            else:
                await self.queue.put(task_params)
        except asyncio.QueueFull:
            async with self._future_lock:
                self.futures.pop(task_id, None)
            raise asyncio.QueueFull(f"Queue is full (max {self.config.queue_maxsize})")
        
        logger.info(f"Request {task_id} queued (size: {self.queue.qsize()})")
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            async with self._metrics_lock:
                self._metrics["completed_requests"] += 1
            return result
        except Exception as e:
            async with self._metrics_lock:
                self._metrics["total_errors"] += 1
            logger.error(f"Request {task_id} failed: {e}")
            raise
        finally:
            async with self._future_lock:
                self.futures.pop(task_id, None)

    async def stream_llm(self, chat: str, chat_id: Optional[str] = None, timeout: float = None, **gen_kwargs):
        if timeout is None:
            timeout = self.config.default_timeout
        async with self.lock:
            if not self.current_model:
                raise ValueError("No model loaded")
        if not self._running_event.is_set():
            raise RuntimeError("Service not started")

        # if self.config.reject_on_full_queue:
        #     try:
        #         self.queue.put_nowait(task_params)
        #     except asyncio.QueueFull:
        #         raise asyncio.QueueFull(f"Queue is full (max {self.config.queue_maxsize}) – try again later")
        # else:
        #     await self.queue.put(task_params)
        task_id = chat_id or str(uuid4())[:12]
        cancel_flag = asyncio.Event()
        self._stream_cancel_flags[task_id] = cancel_flag
        async with self._stream_lock:
            if task_id in self._active_streams:
                raise ValueError(f"Stream ID '{task_id}' already active")
            
            self._active_streams.add(task_id)
        async with self._metrics_lock:
            self._metrics["streaming_requests"] += 1
            self._metrics["total_requests"] += 1

        stream_queue = asyncio.Queue()
        future = asyncio.Future()

        task_params = {
            "task_id": task_id,
            "prompt": chat,
            "future": future,
            "timestamp": time.time(),
            "stream": True,
            "stream_queue": stream_queue,
        }
        allowed_params = ["temperature", "top_p", "top_k", "max_tokens", "stop", "repeat_penalty"]
        for p in allowed_params:
            if p in gen_kwargs:
                task_params[p] = gen_kwargs[p]

        await self.queue.put(task_params)

        async def monitor():
            try:
                await future
            except Exception as e:
                await stream_queue.put(e)
                await stream_queue.put(None)
            finally:
                async with self._stream_lock:
                    self._active_streams.discard(task_id)

        mt=asyncio.create_task(monitor())

        # Yield tokens with timeout per token
        # while True:
        #     try:
        #         token = await asyncio.wait_for(stream_queue.get(), timeout=timeout)
        #     except asyncio.TimeoutError:
        #         # Timeout while waiting for next token – stop the stream
        #         if not future.done():
        #             future.set_exception(asyncio.TimeoutError(f"Stream timed out after {timeout}s"))
        #         break
        #     finally:
        #         mt.cancel()
        #         try:
        #             await mt
        #         except asyncio.CancelledError:
        #             pass
        #         self._active_streams.discard(task_id)
        #     if token is None:
        #         break
        #     if isinstance(token, Exception):
        #         raise token
        #     yield token 
        try:
            while True:
                if cancel_flag.is_set():
                    raise asyncio.CancelledError(f"Stream {task_id} cancelled")
                try:
                    token = await asyncio.wait_for(stream_queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    if not future.done():
                        future.set_exception(asyncio.TimeoutError(f"Stream timed out after {timeout}s"))
                    break
                
                if token is None:
                    break
                if isinstance(token, Exception):
                    raise token
                yield token
        finally:
            self._stream_cancel_flags.pop(task_id, None)
            # Cleanup happens once, after loop exits
            mt.cancel()
            try:
                await mt
            except asyncio.CancelledError:
                pass
            async with self._stream_lock:
                self._active_streams.discard(task_id)
    # ------------------------ Parser helpers ------------------------
    def _extract_response_text(self, response):
        try:
            if isinstance(response, str):
                return response.strip()
            elif isinstance(response, dict):
                if "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    if "text" in choice:
                        return choice["text"].strip()
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"].strip()
                return json.dumps(response)
            return str(response)
        except Exception as e:
            return f"[Error extracting response: {e}]"

    def _parse_raw_response(self, raw_response):
        try:
            if hasattr(self.output_parser, "parse"):
                return self.output_parser.parse(raw_response)
            elif callable(self.output_parser):
                return self.output_parser(raw_response)
            else:
                raise ValueError("output_parser not callable and has no parse method")
        except Exception as e:
            logger.warning(f"Parser failed, using fallback: {e}")
            return self._extract_response_text(raw_response)

    # ------------------------ Runnable interface ------------------------
    async def ainvoke(self, input: Union[str, Dict]) -> str:
        if isinstance(input, dict):
            prompt = input.get("input", str(input))
            chat_id = input.get("chat_id")
            timeout = input.get("timeout", self.config.default_timeout)
            # Extract any generation kwargs from the dict
            gen_kwargs = {k: v for k, v in input.items() if k in ["temperature", "top_p", "top_k", "max_tokens", "stop", "repeat_penalty"]}
        else:
            prompt = str(input)
            chat_id = None
            timeout = self.config.default_timeout
            gen_kwargs = {}
        return await self.chat_llm(prompt, chat_id=chat_id, timeout=timeout, **gen_kwargs)

    def invoke(self, input: Union[str, Dict]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input))
        else:
            return loop.run_until_complete(self.ainvoke(input))

    def __or__(self, other):
        async def chained(value):
            response = await self.ainvoke(value)
            if hasattr(other, "ainvoke"):
                return await other.ainvoke(response)
            elif hasattr(other, "invoke"):
                return other.invoke(response)
            elif callable(other):
                return await other(response) if asyncio.iscoroutinefunction(other) else other(response)
            return response
        return Runnable(chained)

    def __ror__(self, other):
        if callable(other):
            async def chained(value):
                processed = other(value)
                return await self.ainvoke(processed)
            return chained
        return self

    async def __call__(self, input: Union[str, Dict]) -> str:
        return await self.ainvoke(input)

    # ------------------------ Unload and memory ------------------------
    async def get_loaded_models(self) -> List[str]:
        async with self.lock:
            return list(self.current_model.keys())

    async def get_memory_usage(self):
        async with self.lock:
            process = psutil.Process(os.getpid())
            # Quick sync call - memory_info is fast enough
            mem = process.memory_info()
            return {
                "rss_mb": mem.rss / (1024 * 1024),
                "vms_mb": mem.vms / (1024 * 1024),
                "models_loaded": len(self.current_model),
                "model_names": list(self.current_model.keys())
            }
    
    
    # def unload_model(self, model_name: str = None):
    #     if model_name is None:
    #         for name in list(self.current_model.keys()):
    #             self._unload_single_model(name)
    #         self.current_model.clear()
    #         logger.info("Unloaded all models")
    #         return True
    #     if model_name not in self.current_model:
    #         raise ValueError(f"Model '{model_name}' not loaded")
    #     return self._unload_single_model(model_name)

    async def unload_model(self, model_name: str = None):
        async with self.lock:
            if model_name is None:
                for name in list(self.current_model.keys()):
                    await self._unload_single_model_async(name)  # Use async version
                self.current_model.clear()
                logger.info("Unloaded all models")
                return True
            if model_name not in self.current_model:
                raise ValueError(f"Model '{model_name}' not loaded")
            return await self._unload_single_model_async(model_name)


    # Remove the sync _unload_single_model entirely, or make it private with _ prefix
    async def _unload_single_model_async(self, model_name: str) -> bool:
        # Caller MUST hold self.lock! Do NOT acquire here.
        try:
            logger.info(f"Unloading model: {model_name}")
            model = self.current_model.pop(model_name, None)
            if model and hasattr(model, "close"):
                try:
                    model.close()
                except:
                    pass
            # Run gc in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._gc_cleanup)
            logger.info(f"Model {model_name} unloaded")
            return True
        except Exception as e:
            logger.error(f"Error unloading {model_name}: {e}")
            return False
    def _gc_cleanup(self):
        import gc
        gc.collect()
        import sys
        if sys.platform.startswith('linux'):
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass
    
    
    async def unload_all_models(self):
        """Unload all models. Convenience method."""
        return await self.unload_model(None)  # Just delegate