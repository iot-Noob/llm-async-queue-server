import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import PromptTemplate, AsyncLLM, PydanticOutputParser,JsonOutputParser
import asyncio
from pydantic import BaseModel, Field

class Talha(BaseModel):
    name: str = Field(..., description="Name of the person")
    age: int = Field(..., description="Age of the person")

async def main():
    llm = AsyncLLM()
    
    await llm._load_model(   
        model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        temperature=0.7,
        n_predict=512,
        n_ctx=4096,
        repeat_penalty=1.1,
        stop=["<|im_end|>"]
    )
    await llm.start()
    
    parser = PydanticOutputParser(pydantic_object=Talha)
    
    # ✅ Get format instructions FIRST
    format_instructions = parser.get_format_instructions()
    
    # ✅ Use the correct key name matching the template
    pt = PromptTemplate(
        template="""<|im_start|>system
You are a helpful assistant. Return ONLY valid JSON matching the schema.
{format_instructions}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
""",
        input_variables=["input"],
        partial_variables={"format_instructions": format_instructions}  # ✅ Correct key
    )
    
    chain = pt | llm | parser
    
    print("🤖 Generating JSON response...")
    
    try:
        # ✅ No need to pass format_instructions in ainvoke - it's already in partial
        result = await chain.ainvoke({
            "input": "Create a person named John who is 25 years old"
        })
        
        print("\n" + "="*60)
        print("📊 PARSED RESULT")
        print("="*60)
        print("raw_result:::",result)
        print(f"Type: {type(result)}")
        print(f"Name: {result.name}")
        print(f"Age: {result.age}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    await llm.stop()
    llm.unload_all_models()

if __name__ == "__main__":
    asyncio.run(main())