import streamlit as st
import os, json
import openai
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from chain_scoo import get_data_from_docs
from financial_model import financial_model
import pandas as pd
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

st.title("FinAI Co-Pilot")
def flatten_json(json_obj, parent_key='', sep='_'):
    """Flattens a nested JSON object into a single dictionary."""
    items = []
    for k, v in json_obj.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_data_to_json(data, filename="financial_data.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    return filename

def load_and_process_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    # Convert JSON data to text format that the LLM can understand
    text_data = json.dumps(data, indent=2)
    return text_data

def open_modal():
    st.session_state['show_modal'] = True


    
def get_conversation_chain():
    llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    return llm, memory

def main():
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "data" not in st.session_state:
        st.session_state.data = None
    if 'show_modal' not in st.session_state:
        st.session_state['show_modal'] = False
    st.sidebar.title("10-K Reports")
    pdf_files = st.sidebar.file_uploader(
        "Upload your 10-K PDF files", type=['pdf'], accept_multiple_files=True)
    
    if pdf_files:
        data = get_data_from_docs(pdf_files)
        processed_data = {}
        for item in data:
            report_date = item['report_date']
            item.pop("metadata")
            item.pop("section")
            flattened_item = flatten_json(item)
            # Remove the 'report_date' key from the flattened dictionary
            if 'report_date' in flattened_item:
                del flattened_item['report_date']
            processed_data[report_date] = flattened_item
            
        kpis = financial_model(data)
        table_data = []
        for key, value_dict in kpis.items():
            row = {"Metric": key}
            row.update(value_dict)
            table_data.append(row)

        # Create the DataFrame
        df_kpis = pd.DataFrame(table_data)

        # Set the 'Metric' column as the index
        df_kpis = df_kpis.set_index("Metric")
        # Create the DataFrame
        df = pd.DataFrame.from_dict(processed_data, orient='index').T

        


    st.write("Ask questions about the 10-K reports")
    st.button("Open Details", on_click=open_modal)
    if st.session_state['show_modal']:
            st.write("Extracted Financial Data")
            st.dataframe(df)
            st.write("Extracted Financial KPIs")
            st.dataframe(df_kpis)
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_question = st.chat_input("Ask a question about the 10-K reports...")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        
        if st.session_state.data:
            with st.spinner("Analyzing..."):
                # Create a prompt that includes the JSON data
                prompt = f"""
                Here is the financial data in JSON format:
                {json.dumps(st.session_state.data, indent=2)}
                
                Question: {user_question}
                
                Please analyze the data and provide an answer.
                """
                
                # Get the LLM and memory
                llm, memory = get_conversation_chain()
                
                # Get the response
                response = llm.invoke(prompt)
                
                with st.chat_message("assistant"):
                    st.write(response.content)
                
                # Update chat history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": response.content})
        else:
            st.warning("Please upload 10-K PDF files first.")

if __name__ == "__main__":
    main()

