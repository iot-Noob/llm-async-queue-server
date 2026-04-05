import os
import sys
# Add parent directory to path so Python can find main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from main import PromptTemplate, AsyncLLM, PydanticOutputParser
import asyncio

async def main():
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

    class Data(BaseModel):
        name: str = Field(description="name of user")
        age: int = Field(description="age of person")

    # Create parser
    parser = PydanticOutputParser(pydantic_object=Data)
    
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
    
    # ✅ Fixed: Add parser to chain
    chain = pt | llm | parser
    
    res = await chain.ainvoke({
        "input": "Create a random user with a name and age",
        "chat_id": "talha_id_69",
        "format_instructions": format_instructions
    })
    
    # ✅ Now res is a Data object (Pydantic model)
    print("\n" + "="*60)
    print("📊 PARSED RESULT")
    print("="*60)
    print(f"Name: {res.name}")
    print(f"Age: {res.age}")
    print(f"Type: {type(res)}")
    
    await llm.stop()
    llm.unload_all_models()

if __name__ == "__main__":
    asyncio.run(main())