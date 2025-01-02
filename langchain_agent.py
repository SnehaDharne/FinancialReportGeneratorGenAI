from langchain.agents import AgentExecutor, create_openai_tools_agent,Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from tools.extract_tool import Extract_Tool
from tools.growth_tool import Growth_Tool
from tools import GrossmmarginpercTool, ExpensespercTool, TaxrateTool
import openai


def create_agent(model_name="gpt-4o-mini", key="AAA"):
    openai.api_key = key
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # rag_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # rag_prompt = hub.pull("rlm/rag-prompt")
    # rag_db = Chroma(persist_directory="../../chroma_db", 
    #                    embedding_function=OpenAIEmbeddings())
    # rag_retriever = rag_db.as_retriever()
    
    tools = [Extract_Tool, Growth_Tool, GrossmmarginpercTool, ExpensespercTool, TaxrateTool]
    
    llm = ChatOpenAI(model=model_name, temperature=0)

    system_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Extract_Tool, Growth_Tool, GrossmmarginpercTool, ExpensespercTool, TaxrateTool"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_exe = AgentExecutor(agent=agent, tools=tools,memory=memory,verbose=True)
    return agent_exe

async def run_agent(agent,user_query):
    #print(agent.memory.chat_memory.messages[-2:] if len(agent.memory.chat_memory.messages) > 1 else "")
    #set_verbose(True)
    print(agent.memory.chat_memory)
    print('********************')
    print()
    return await agent.ainvoke(input={"input":user_query},verbose=True)