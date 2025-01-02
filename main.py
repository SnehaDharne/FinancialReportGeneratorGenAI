import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools import __all__
import streamlit as st

# Step 1: Environment Configuration
def load_api_key(env_file='key.env'):
    load_dotenv(env_file)
    return os.getenv('open_ai_API_Key')

# Step 2: Initialize Tools
def initialize_tools():
    tools = __all__
    return tools
    

# Step 3: Main Streamlit App
def main():
    # Load API key and tools
    api_key = load_api_key()
    tools = initialize_tools()
    
    # Initialize LLM and agent
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key, max_tokens=5000)
    prompt = hub.pull("hwchase17/structured-chat-agent")
    
    # Initialize memory in Streamlit
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",  # Explicit key for memory storage
            return_messages=True
        )
        initial_message = "You are an AI assistant that can provide helpful answers using available tools. \nIf you are unable to answer, you can use the following tools:'Extract_Tool', 'Growth_Tool', 'GrossmmarginpercTool','ExpensespercTool','TaxrateTool','NetMargin_Tool','EBTTool','EPS_Tool','RNDGrowth_Tool','RNDTool','SNGTool','GenerateGraph_Tool','GenerateCSV_Tool'. if a tool returns a dict like this: <curly bracket> link : <link to a file> <close curly bracked> return the response link [a response link looks like this /tmp/..] so it can displayed on the streamlit frontend. do not have any text other than the link itself in the output.\n"

        # Add the system message to memory only once
        st.session_state.memory.chat_memory.add_message(
            SystemMessage(content=initial_message)
        )

    # Create the structured chat agent with memory
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state.memory,  # Pass memory explicitly
        handle_parsing_errors=True,
        max_iterations=50
    )

    # Streamlit UI
    st.title("AI Financial Assistant")

    # Display chat history excluding the system message
    for msg in st.session_state.memory.chat_memory.messages:
        if isinstance(msg, SystemMessage):
            continue  # Skip displaying the system message
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        if msg.content.startswith("sandbox:/tmp/") or  msg.content.startswith("/tmp/") and os.path.exists(msg.content) and msg.content.endswith(".png"):
            # Display the image
                st.chat_message("assistant").write("Here's the generated file:")
                st.image(msg.content)  # This will display the image in the chat
        else:
            st.chat_message(role).write(msg.content)

        

    # User input
    user_input = st.chat_input("Enter your query...")
    if user_input:
        st.chat_message("user").write(user_input)
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": st.session_state.memory.chat_memory.messages  # Pass memory explicitly
            })
            response_output = response.get("output", "Sorry, I couldn't generate a response.")
            if response_output.startswith("sandbox:/tmp/") or  response_output.startswith("/tmp/") and os.path.exists(response_output):
                if response_output.endswith(".png"):
                    st.chat_message("assistant").write("Here's the generated file:")
                    st.image(response_output)  # This will display the image in the chat
                elif response_output.endswith(".csv"):
                    st.chat_message("assistant").write("Here's the generated CSV file:")
                    with open(response_output, "r") as file:
                        st.download_button(
                            label="Download CSV",
                            data=file,
                            file_name=os.path.basename(response_output),
                            mime="text/csv"
                        )
            else:
                st.chat_message("assistant").write(response_output)
        except Exception as e:
            response_output = f"Error: {str(e)}"
            st.chat_message("assistant").write(response_output)


if __name__ == "__main__":
    main()
