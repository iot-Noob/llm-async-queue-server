import os
import sys
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List

# Add parent directory to path so Python can find main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    AsyncLLM, 
    PromptTemplate, 
    StrOutputParser, 
    JsonOutputParser, 
    PydanticOutputParser,
    AIMessage
)

# Configuration from test.py
MODEL_NAME = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"

class UserProfile(BaseModel):
    name: str = Field(description="The user's name")
    interests: List[str] = Field(description="List of their interests")
    age: int = Field(description="The user's age")

async def test_all_parsers():
    # Initialize LLM
    llm = AsyncLLM()
    
    print(f"--- Loading Model: {MODEL_NAME} ---")
    await llm._load_model(
        model_name=MODEL_NAME,
        temperature=0.1,  # Low temperature for parsing consistency
        n_predict=512,
        n_ctx=4096,
        stop=["<|im_end|>"]
    )
    await llm.start()

    try:
        # [1] Raw AIMessage (No Parser)
        print("\n[1] TEST: Raw AIMessage")
        res_raw = await llm.ainvoke("Write a short sentence about coding.")
        print(f"Type: {type(res_raw)}")
        print(f"Metadata Keys: {list(res_raw.response_metadata.keys())}")
        print(f"Content: {res_raw.content}")

        # [2] StrOutputParser
        print("\n[2] TEST: StrOutputParser")
        str_chain = llm | StrOutputParser()
        res_str = await str_chain.ainvoke("Write a one-word greeting.")
        print(f"Type: {type(res_str)}")
        print(f"Result: {res_str}")

        # [3] JsonOutputParser (Dict)
        print("\n[3] TEST: JsonOutputParser (to Dict)")
        json_parser = JsonOutputParser()
        json_prompt = PromptTemplate.from_template(
            "Return a JSON object with keys 'topic' and 'sentiment' for this text: {input}. "
            "Respond ONLY with JSON."
        )
        json_chain = json_prompt | llm | json_parser
        res_json = await json_chain.ainvoke({"input": "I love the new async engine!"})
        print(f"Type: {type(res_json)}")
        print(f"Result: {json.dumps(res_json, indent=2)}")

        # [4] PydanticOutputParser (Validated Model)
        print("\n[4] TEST: PydanticOutputParser")
        pydantic_parser = PydanticOutputParser(pydantic_object=UserProfile)
        format_instructions = pydantic_parser.get_format_instructions()
        
        pydantic_prompt = PromptTemplate(
            template="""<|im_start|>system
            Extract user info into JSON.
            {format_instructions}<|im_end|>
            <|im_start|>user
            My name is Alice, I'm 28, and I like Python and AI.<|im_end|>
            <|im_start|>assistant
            """,
            input_variables=["format_instructions"]
        )
        
        pydantic_chain = pydantic_prompt | llm | pydantic_parser
        res_pydantic = await pydantic_chain.ainvoke({"format_instructions": format_instructions})
        
        print(f"Type: {type(res_pydantic)}")
        print(f"Validated Model Data: {res_pydantic}")
        print(f"Name: {res_pydantic.name}")
        print(f"Interests: {res_pydantic.interests}")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n--- Cleaning Up ---")
        await llm.stop()
        await llm.unload_all_models()

if __name__ == "__main__":
    asyncio.run(test_all_parsers())
