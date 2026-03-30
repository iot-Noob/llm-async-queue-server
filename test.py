from main import EcharrParsers, AsyncLLM, StrOutputParser, PromptTemplate
import json
import asyncio


async def main():
    print("="*60)
    print("CHART GENERATION EXAMPLE")
    print("="*60)
    
    # Initialize components
    ecp = EcharrParsers()
    llm = AsyncLLM()
    parser = StrOutputParser()
  
    
    pt=PromptTemplate(template="you are helpful assistant help me out from my queston from imnput  input\n{input}",input_variables=["input"])
    
    # ✅ FIX 3: Load and start LLM
    print("\n📦 Loading model...")
    await llm._load_model(
        model_name="glm-4-9b-chat-IQ4_XS.gguf",
        temperature=0.1,
        max_tokens=2048,
        n_ctx=4096
    )
    
    print("🚀 Starting worker...")
    await llm.start()
 
    chain = pt | llm | parser
    
    res=await chain.ainvoke({"input":"hi what is size of normal penisof male in pakistan? whats avg size and dia of it?"})
    print(res)
    # Cleanup
    print("\n" + "="*60)
    print("CLEANING UP:")
    print("="*60)
    await llm.stop()
    llm.unload_all_models()
    
    print("\n✅ Done!")

if __name__ == "__main__":
    asyncio.run(main())