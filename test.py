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
  
    
    pt = PromptTemplate(
        template="""<|im_start|>system
    You are a helpful assistant that provides accurate, informative answers. Answer questions clearly and completely.
    <|im_end|>
    <|im_start|>user
    {input}
    <|im_end|>
    <|im_start|>assistant
    """,
        input_variables=["input"]
    )
    # ✅ FIX 3: Load and start LLM
    print("\n📦 Loading model...")
    await llm._load_model(
        model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        temperature=0.6,
        n_predict=4096,
        n_ctx=4096,
    )
    
    print("🚀 Starting worker...")
    await llm.start()
 
    chain = pt | llm | parser
    
    res=await chain.ainvoke({"input":"What boys suffer sexual abuse get shiftet eyes??"})
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