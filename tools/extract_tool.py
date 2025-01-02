import json
from pydantic import BaseModel, Field
from langchain.tools import tool
from typing import Optional, List, Tuple
class Extract_toolArgs(BaseModel):
    a: str = Field(description="Company name")
    b: str = Field(description="item")
    c: str = Field(description="Year_1")
    d: str = Field(description="Year_2")
    e :Optional[str] = Field(default=None,description="category")

class Extracted_toolArgs(BaseModel):
    year : list = Field(description="List of years")
    value : list = Field(discription = "List of Values")



@tool(args_schema=Extract_toolArgs)
def Extract_Tool(a: str, b: str, c: str, d: str, e: str = None) -> Extract_toolArgs:
    """Extracts specific data for a company from Year_1 to Year_2 based on item and optional category."""
    with open("data_soo.json", "r") as json_file:
        data = json.load(json_file)

    a = a.upper()
    b = b.lower()

    filtered_data = [
        item for item in data.get(a, [])
        if item["Item"].lower() == b and (not e or item.get("Category", "").lower() == e.lower())
    ]

    if not filtered_data:
        raise ValueError(f"No data found for company '{a}', item '{b}', and category '{e}'.")

    json_data = filtered_data[0]

    year_data = []
    cash_data = []

    if int(c) > int(d):
        c, d = d, c

    for year in range(int(c), int(d) + 1):
        value = json_data.get(str(year))
        if value is not None:
            cash_data.append(value)
            year_data.append(year)

    if len(cash_data) == 0:
        raise ValueError(f"Data for the years {c} to {d} is not available.")

    return cash_data, year_data