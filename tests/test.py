import os
import sys
# Add parent directory to path so Python can find main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from main import PromptTemplate, AsyncLLM, PydanticOutputParser
import asyncio

async def main():

    class Data(BaseModel):
        name: str = Field(description="name of user")
        age: int = Field(description="age of person")

    # Create parser
    parser = PydanticOutputParser(pydantic_object=Data)
    # Decoupled: No parser passed to AsyncLLM
    llm = AsyncLLM()
    
    await llm._load_model(   
        model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        temperature=0.7,
        n_predict=512,  # Reduced for faster response
        streaming=False,
        n_ctx=4096,
        stop=["<|im_end|>"]
    )
    await llm.start()


    
    # Get format instructions
    format_instructions = parser.get_format_instructions()
    
    # ✅ Fixed: Use proper ChatML format and include format_instructions
    pt = PromptTemplate(
        template="""<|im_start|>system
                    You are a helpful assistant. Generate ONLY valid JSON matching the schema below.
                    {format_instructions}<|im_end|>
                    <|im_start|>user
                    {input}<|im_end|>
                    <|im_start|>assistant
                    """,
        input_variables=["input", "format_instructions"]
    )
    
    # 1. Testing Raw AIMessage (Returns object with metadata)
    res = await llm.ainvoke({
        "input": "What you know about zionism",
    })
    
    print("\n--- [1] AIMessage Discovery ---")
    print(f"Result Type: {type(res)}")
    print(f"Content Preview: {str(res)[:100]}...")
    
    # This is the "Raw Response with all params" you're looking for:
    print("\n--- Full Raw Response Metadata ---")
    import json
    print(json.dumps(res.response_metadata, indent=2))
    # 2. Testing Piped Chain (Decoupled Parsing)    
    print("\n--- [2] Piped Chain (Prompt | LLM | Parser) ---")
    # Link the components: Template -> LLM -> Parser
    chain = pt | llm | parser
    
    parsed_res = await chain.ainvoke({
        "input": "My name is Talha and I am 30 years old",
        "format_instructions": format_instructions
    })
    
    print(f"Parsed Type: {type(parsed_res)}")
    print(f"Parsed Data: {parsed_res}")
 
    
    await llm.stop()
    await llm.unload_all_models()

if __name__ == "__main__":
    asyncio.run(main())