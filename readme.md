# FinAi- AI Agent

This project is an AI-driven tool designed to analyze financial metrics, generate reports, and create graphs based on user preferences. It leverages LangChain for AI agent workflows and provides a suite of tools for various financial computations.

## Code Structure
The project consists of the following components:
chaining.ipynb : This files extracts the data from the 10-k reports and saves it to data_soo.json

data_soo.json : consists of data extracted in json format from the 10k reports

langchain_agent.py : Implements the AI agent workflow using LangChain.

tools: It contains the custom tools for the agent to use

main.py: The primary entry point for the project.

## Preparing the code to run:
Set Up Environment Variables:
set up a key.env and store openai key.

## How to run:
in the terminal locate to the main.py
streamlit run main.py

The chat agent will be ready to answer.

If you want to add more data or 10-ks, just add your document path in the loader of chaining.ipynb and run the whole file.



