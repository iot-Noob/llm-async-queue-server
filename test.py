from main import ChatPromptTemplate,PromptTemplate,AsyncLLM,StrOutputParser
import asyncio
async def main():
 
    llm=AsyncLLM()
    model=await llm._load_model(   
    model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    temperature=0.7,
    n_predict=2046,
    n_ctx=4024,
    stop=["<|im_end|>"])
    await llm.start()
    parser=StrOutputParser()
    pt=PromptTemplate(template="you are a helpful assistant help me in any way with input {input}",input_variables=["input"])
    chain=pt | llm | parser
    res=await chain.ainvoke({"input":"hi is it true you cant make TNT at home"})
    print(res)
    llm.stop()
    llm.unload_all_models()
 

if __name__=="__main__":
    asyncio.run(main())