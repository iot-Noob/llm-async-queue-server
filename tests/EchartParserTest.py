from main import ChatPromptTemplate,PromptTemplate,AsyncLLM,StrOutputParser,EcharrParsers
import asyncio
async def main():
 
    llm=AsyncLLM()
    model=await llm._load_model(   
    model_name="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    temperature=0.1,
    n_predict=2046,
    n_ctx=4024,
    repeat_penalty=1.1,
    stop=["<|im_end|>"])
    await llm.start()
    
    parser=StrOutputParser()
    ecp=EcharrParsers()
    Dec=ecp.DEC
    pl=ecp.get_enum_list()
 
    Dec = ecp.DEC # This is your Dynamic Enum
    
    # 1. Define the PromptTemplate with a placeholder for the instructions {fi}
    pt = PromptTemplate(
        template="""<|im_start|>system
    You are an Echarts specialist. 
    Based on the user requirement: {input}
    Use the following FORMAT INSTRUCTIONS to generate the JSON:
    {fi}<|im_end|>
    <|im_start|>assistant
    """,
        input_variables=["input", "fi"]
    )
    
    chain = pt | llm | parser

    # 2. Get the specific template for BASIC_BAR_CHART
    # This retrieves ONLY the bar chart section of your JSON
    instructions = ecp.get_full_injection_prompt(
        user_input="make a temp bar chart for oil prices in China from 1920 to 2020", 
        selected_chart=Dec.BASIC_BAR_CHART
    )

    # 3. Pass both variables into the chain
    res = await chain.ainvoke({
        "input": "make a temp bar chart",
        "fi":instructions
    })
    print(res)
    await llm.stop()
    llm.unload_all_models()
 

if __name__=="__main__":
    asyncio.run(main())