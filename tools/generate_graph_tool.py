from pydantic import BaseModel, Field
import traceback
import matplotlib.pyplot as plt
from langchain.tools import tool
import io
import os
from dotenv import load_dotenv
import openai
load_dotenv('key.env')  
key_string = os.getenv('open_ai_API_Key')

# Set your OpenAI API key
openai.api_key = key_string

class GenerateGraphToolArgs(BaseModel):
    data: list = Field(description="List of numerical values or metrics to plot.")
    labels: list = Field(description="List of labels (e.g., years, categories) corresponding to the data.")
    graph_type: str = Field(description="Type of graph, such as 'bar chart', 'line graph', or 'scatter plot'.")
    title: str = Field(description="Title of the graph.")

@tool(args_schema=GenerateGraphToolArgs)
def GenerateGraph_Tool(data: list, labels: list, graph_type: str, title: str) -> str:
    """
    Generates Python code to create a graph (e.g., bar, line, scatter) using matplotlib based on user input.
    Outputs code that gets executed in the frontend.
    """
    prompt = (
        f"Generate Python code to create a '{graph_type}' using matplotlib. The graph should have the title '{title}', "
        f"the X-axis labels as {labels}, and the Y-axis values as {data}. Return only the code. Always replace plt.show() with plt.savefig(buffer, format='png')"
    )
    
    try:
        # Use OpenAI to generate code
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant that generates Python graph code using matplotlib."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        code = response.choices[0].message.content
        return str({'code': str(code.strip())})
    except Exception as e:
        return f"Error generating graph code: {str(e)}"

### Step 2: Tool Schema for Executing Graph Code ###
class ExecuteGraphToolArgs(BaseModel):
    code: str = Field(description="The Python code to execute and display the graph.")
    filename: str = Field(description="A relevant name for the vizualization file")

class executedGraphToolArgs(BaseModel):
    file_path: str = Field(description="A dictionary with key 'link' and value 'file_path' which is the url of the file")


@tool(args_schema=ExecuteGraphToolArgs)
def ExecuteGraph_Tool(code: str, filename: str) -> executedGraphToolArgs:
    """
    Executes the Python code to  display the graph.
    """
    code = code.strip('```python')
    namespace = {"plt": plt, "io": io}
    try:
        exec(code, namespace)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        print(f"Buffer type: {type(buffer)}")
        file_path = f"/tmp/{filename}.png"
        plt.savefig(file_path)
        plt.close() 
        return str({'link':file_path})
    except Exception as e:
        return ("An error occurred while executing the graph code."+traceback.format_exc())


