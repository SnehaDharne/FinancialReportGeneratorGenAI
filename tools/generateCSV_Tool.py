import pandas as pd
import os
import io
from langchain.tools import tool
from pydantic import BaseModel, Field

class GenerateCSVToolArgs(BaseModel):
    data: list = Field(description="List of data (list of dictionaries) to save as a CSV.")
    filename: str = Field(description="Desired filename for the CSV (without extension).")

class executedGenerateCSVToolArgs(BaseModel):
    file_path: str = Field(description="A dictionary with key 'link' and value 'file_path' which is the url of the file")

@tool(args_schema=GenerateCSVToolArgs)
def GenerateCSV_Tool(data: list, filename: str) -> executedGenerateCSVToolArgs:
    """
    Generates a CSV file from the provided data and returns the path to download the file.
    """
    try:
        df = pd.DataFrame(data)
        file_path = f"/tmp/{filename}.csv"
        df.to_csv(file_path, index=False)
        return str({'link': file_path})
    
    except Exception as e:
        return f"Error generating CSV: {str(e)}"