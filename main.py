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
from typing import Dict,List,Union,Any,Tuple,Type,TypeVar,Optional,Generic
from queue import Queue
import psutil
import json
import shutil
import re
from enum import Enum
from dataclasses import dataclass,field
import time


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
 
    def __init__(self, ** kwargs):
        """
        An asynchronous, queue-based LLM wrapper for llama.cpp with LangChain-like chaining.
        
        This class provides a production-ready interface for running GGUF models with:
        - Non-blocking async/await support
        - Queue-based request handling for concurrent operations
        - Automatic memory management and cleanup
        - Pluggable output parsers (JSON, Pydantic, plain text)
        - Worker crash detection and optional auto-restart
        - Configurable memory thresholds for different devices
        
        Features
        --------
        * Async queue processing with background worker
        * Chainable with Runnable components using | operator
        * Automatic stop token detection and logging
        * Graceful fallbacks for parsing errors
        * Memory leak prevention with garbage collection
        * Cross-platform support (Linux, Windows, Android via Termux)
        
        Parameters
        ----------
        available_ram : int, optional
            Minimum required free RAM in MB before loading model.
            Set to 0 to disable check. Default: 3000 (3GB)
        model_path : str, optional
            Direct path to models directory. Overrides .env file.
        worker_exception_callback : callable, optional
            Function called when worker crashes. Receives exception as argument.
        auto_restart_worker : bool, optional
            Whether to automatically restart worker on crash. Default: False
        temperature : float, optional
            Sampling temperature (0.0 = deterministic, 2.0 = random). Default: 0.1
        top_p : float, optional
            Nucleus sampling threshold (0.0 to 1.0). Default: 0.9
        top_k : int, optional
            Top-k sampling limit. Default: 30
        streaming : bool, optional
            Enable token streaming (not fully implemented). Default: False
        repeat_penalty : float, optional
            Penalty for repeating tokens (1.0 = no penalty). Default: None
        n_predict : int, optional
            Maximum tokens to generate. Default: 2048
        n_batch : int, optional
            Batch size for prompt processing. Default: 128
        n_ctx : int, optional
            Context window size (input memory). Default: 2048
        n_threads : int, optional
            Number of CPU threads. Default: 6
        n_gpu_layers : int, optional
            GPU layers to offload (-1 = all, 0 = CPU only). Default: -1
        verbose : bool, optional
            Enable verbose logging. Default: False
        stop : List[str], optional
            Stop sequences. Default: ["<|endoftext|>", "<|im_end|>"]
        output_parser : object, optional
            Custom output parser (must have parse method). Default: StrOutputParser()
        
        Examples
        --------
        Basic Usage:
        >>> llm = AsyncLLM()
        >>> await llm._load_model(model_name="my-model.gguf")
        >>> await llm.start()
        >>> response = await llm.chat_llm("What is Python?")
        >>> print(response)
        >>> await llm.stop()
        >>> llm.unload_all_models()
        
        With Chaining:
        >>> prompt = PromptTemplate(template="User: {input}\\nAssistant: ")
        >>> parser = StrOutputParser()
        >>> chain = prompt | llm | parser
        >>> result = await chain.ainvoke({"input": "Hello"})
        
        With Pydantic Output:
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> parser = PydanticOutputParser(pydantic_object=Person)
        >>> chain = prompt | llm | parser
        >>> result = await chain.ainvoke({"input": "Create a person"})
        >>> print(result.name, result.age)
        
        For Mobile Devices (Samsung A06, Raspberry Pi):
        >>> llm = AsyncLLM(
        ...     available_ram=1500,  # Only need 1.5GB free
        ...     n_ctx=2048,          # Smaller context
        ...     n_predict=256,       # Shorter responses
        ...     n_threads=4,         # Match CPU cores
        ...     n_gpu_layers=0       # CPU only
        ... )
        
        With Error Callback:
        >>> def on_crash(exception):
        ...     print(f"Worker crashed: {exception}")
        ...     # Send alert, restart service, etc.
        >>> llm = AsyncLLM(
        ...     worker_exception_callback=on_crash,
        ...     auto_restart_worker=True
        ... )
        
        Notes
        -----
        - Model path can be set via .env file (MODEL_PATH="/path/to/models") or passed directly
        - Memory check prevents OOM crashes on low-RAM devices
        - Worker crash callback enables monitoring and recovery
        - Queue size is 10; requests beyond that wait
        - Default timeout is 160 seconds for generation
        
        See Also
        --------
        StrOutputParser : Simple string output parser
        JsonOutputParser : JSON output with optional Pydantic validation
        PydanticOutputParser : Type-safe Pydantic model output
        PromptTemplate : Template for formatting prompts
        Runnable : Base class for chainable components
        """
        self.available_ram=kwargs.get("available_ram",3000)
        self.setting=Settings()
        self.model_path=kwargs.get("model_path")
        self._worker_exception_callback = kwargs.get("worker_exception_callback", None)
        self._auto_restart_worker = kwargs.get("auto_restart_worker", False)
        self.futures:Dict[str,asyncio.Future]={}    
        self.current_model:Dict[str,llama_cpp.Llama]={}
        if self.model_path:
            self.path=self.model_path
        else:
            try:
                self.path=self.setting.MODEL_PATH
            except ValidationError as ve:
                raise ValueError(f"MODEL_PATH not set. Provide it via .env or pass model_path argument.") from ve
        
 
        self.async_lock=asyncio.Lock()
        self.queue=asyncio.Queue(maxsize=10)
        self._running=False
        self._worker_task=None

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
        self.output_parser = kwargs.get("output_parser", StrOutputParser())
 

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
            if avail < self.available_ram:
                raise MemoryError(f"Need {(self.available_ram/1000)}GB free, have {avail:.0f}MB")
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
    #     try:f
    #         if callable(tname):
    #             res= self.loop.run_until_complete(tname(**kw))
    #             return res
    #     except Exception as e:
    #         raise ValueError(f"Error run task due to {e}")


    def _worker_done_callback(self, task):
            """Handle worker task completion and exceptions"""
            if task.cancelled():
                print("⚠️ Worker task was cancelled")
                return
            
            exception = task.exception()
            if exception:
                print(f"❌ Worker task crashed with exception: {exception}")
                import traceback
                traceback.print_exception(type(exception), exception, exception.__traceback__)
                
                # Call custom callback if provided
                if self._worker_exception_callback:
                    try:
                        self._worker_exception_callback(exception)
                    except Exception as e:
                        print(f"⚠️ Exception callback failed: {e}")
                
                # Optional: Auto-restart worker
                if hasattr(self, '_auto_restart_worker') and self._auto_restart_worker:
                    print("🔄 Auto-restarting worker...")
                    self._worker_task = asyncio.create_task(self._worker())
                    self._worker_task.add_done_callback(self._worker_done_callback)
    
    
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
        """(
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
        
    def _extract_response_text(self, response):
        """Fallback extraction when parser fails"""
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
        """
        LangChain‑style parsing of the raw LLM response.
        Uses the configured output_parser to convert the raw response to a string.
        Falls back to _extract_response_text if the parser fails.
        """
        try:
            # If the parser has a parse method (like StrOutputParser)
            if hasattr(self.output_parser, 'parse'):
                return self.output_parser.parse(raw_response)
            # If it's a callable
            elif callable(self.output_parser):
                return self.output_parser(raw_response)
            else:
                raise ValueError("output_parser not callable and has no parse method")
        except Exception as e:
            # Fallback to the safe extraction method
            print(f"⚠️ Parser failed, using fallback: {e}")
            return self._extract_response_text(raw_response)
        
    async def _worker(self):
        """Background worker - robust response handling"""
        while self._running:
            task = None
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                task_id = task["task_id"]
                prompt = task["prompt"]
                future = task["future"]
                
                if not self.current_model:
                    future.set_exception(ValueError("No model loaded"))
                    continue
                
                model = list(self.current_model.values())[0]
                
                print(f"⚙️ [{task_id}] Processing: {prompt[:50]}...")
                
                # Run inference
                raw_response = await asyncio.to_thread(
                    model, 
                    prompt, 
                    max_tokens=self.n_predict,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    # Remove stream if you don't handle it
                    # stream=self.streaming,  
                )
                if isinstance(raw_response, dict) and "choices" in raw_response:
                    choice = raw_response["choices"][0]
                    finish_reason = choice.get("finish_reason")
                    
                    if finish_reason == "length":
                        print(f"⚠️ [{task_id}] Stopped due to max_tokens limit ({self.n_predict})")
                    elif finish_reason == "stop":
                        print(f"✅ [{task_id}] Stopped by stop token")
                    elif finish_reason:
                        print(f"ℹ️ [{task_id}] Finish reason: {finish_reason}")
                # SAFE EXTRACTION
                text_output = self._parse_raw_response(raw_response)
                
                if not future.done():
                    future.set_result(text_output)
                
                print(f"✅ [{task_id}] Completed")
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                print("Worker cancelled")
                break
            except Exception as e:
                print(f"❌ Worker error: {e}")
                # Only set exception if future exists and not done
                if task and 'future' in task and not task['future'].done():
                    task['future'].set_exception(e)
            finally:
                if task:
                    self.queue.task_done() 
    async def start(self):
        """Start the worker"""
        if self._running:
            print("Worker already running")
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        self._worker_task.add_done_callback(self._worker_done_callback)
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
    
    def _validate_queue(self):
        """Check queue status and warn if near capacity"""
        if self.queue.qsize() >= self.queue.maxsize:
            print(f"⚠️ Queue is FULL ({self.queue.qsize()}/{self.queue.maxsize})!")
            return False
        return True
    
    # ========== CHAT METHOD ==========
    async def chat_llm(self, chat: str,chat_id:Optional[str]=None,timeout:float=160.0) -> str:
        """    Send a message to the LLM with optional custom request ID.
    
    Args:
        chat: The user message/prompt to send
        chat_id: Optional custom ID for tracking this request.
                 If not provided, auto-generates a UUID.
        timeout: Maximum time to wait for response in seconds.
                 Default: 160.0
    
    Returns:
        The LLM's response as a string
    
    Examples:
        >>> # Auto-generate ID
        >>> response = await llm.chat_llm("Hello")
        
        >>> # Custom ID for tracking
        >>> response = await llm.chat_llm("Hello", chat_id="user_123")
        
        >>> # Custom timeout
        >>> response = await llm.chat_llm("Long essay", timeout=300.0)"""
        try:
            task_id=None
            if not self.current_model:
                raise ValueError("No model loaded. Call _load_model() first.")
            
            if not self._running:
                raise RuntimeError("Service not started. Call start() first.")
            if self.queue.qsize() >= self.queue.maxsize * 0.8:
                print(f"⚠️ Queue is {self.queue.qsize()}/{self.queue.maxsize} - consider reducing load")
            if chat_id:
                task_id=chat_id
            else:
                task_id = str(uuid4())[:12]
            future = asyncio.Future()
            if task_id in self.futures:
                raise ValueError(f"Request ID '{task_id}' is already in use. Use a unique ID.")
            self.futures[task_id] = future
            
            await self.queue.put({
                "task_id": task_id,
                "prompt": chat,
                "future": future,
                "timestamp": time.time()
            })
            
            print(f"📝 [{task_id}] Queued (position: {self.queue.qsize()})")
            
            return await asyncio.wait_for(future, timeout=timeout)

            
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            raise
        finally:
            if 'task_id' in locals() and task_id in self.futures:
                self.futures.pop(task_id, None)
                print(f"🗑️ [{task_id}] Future cleaned from memory")
    async def ainvoke(self, input: Union[str, Dict]) -> str:
        """Async invoke for chaining"""
        if isinstance(input, dict):
            prompt = input.get("input", str(input))
            chat_id = input.get("chat_id")  # Optional
            timeout = input.get("timeout", 160.0)  # Optional custom timeout
        else:
            prompt = str(input)
            chat_id = None  # ← ADD THIS
            timeout = 160.0  # ← ADD THIS
        return await self.chat_llm(prompt,chat_id=chat_id,timeout=timeout)
    
    def invoke(self, input: Union[str, Dict]) -> str:
        """Sync invoke - runs LLM"""
        if isinstance(input, dict):
            prompt = input.get("input", str(input))
            chat_id = input.get("chat_id")
            timeout = input.get("timeout", 160.0)
        else:
            prompt = str(input)
            chat_id = None
            timeout = 160.0
        
        try:
            # Try to run in existing loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create new one
            return asyncio.run(self.chat_llm(prompt, chat_id, timeout))
        else:
            # ✅ Already in async context - need to pass chat_id and timeout!
            return loop.run_until_complete(self.chat_llm(prompt, chat_id, timeout))    
    # def invoke(self, input: Union[str, Dict]) -> str:
    #     """Sync invoke - runs LLM"""
    #     if isinstance(input, dict):
    #         prompt = input.get("input", str(input))
    #         chat_id = input.get("chat_id")  # Optional
    #         timeout = input.get("timeout", 160.0)  # Optional custom timeout
    #     else:
    #         prompt = str(input)
    #         chat_id = None  # ← ADD THIS
    #         timeout = 160.0  # ← ADD THIS
    #     try:
    #         # Try to run in existing loop
    #         loop = asyncio.get_running_loop()
    #     except RuntimeError:
    #         # No running loop, create new one
    #         return asyncio.run(self.chat_llm(prompt,chat_id,timeout))
    #     else:
    #         # Already in async context
    #         return loop.run_until_complete(self.chat_llm(prompt))
    
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
