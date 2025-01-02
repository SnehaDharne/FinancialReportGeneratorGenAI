import json
from pydantic import BaseModel, Field
from langchain.tools import tool

class RNDgrowthToolArgs(BaseModel):
    RNDgrowth_data : list = Field(description="list of research and development percentage sales")
    year_data:list = Field(description="list of years of research and development percentage sales")

@tool(args_schema=RNDgrowthToolArgs)
def RNDGrowth_Tool(RNDgrowth_data: list, year_data: list) -> list[float]:
    """
    Calculates year-on-year (YoY) growth for research and development percentage sales for a specific company.
    This function receives year data and research and development percentage sales.

    Arguments:
    - RNDgrowth_data: List of research and development percentage sales.
    - year_data: List of corresponding years.
    
    Returns:
    - A list of floats representing the YoY growth percentage for each year compared to the previous year.
    """
    if len(RNDgrowth_data) < 2:
        return ["Not enough data to calculate year-on-year sales growth."]
    
    yoy_growth = []
    growth_data_sorted = sorted(zip(year_data, RNDgrowth_data), key=lambda x: x[0])
    sorted_years, sorted_sales = zip(*growth_data_sorted)

    for i in range(1, len(sorted_years)):
        current_year = sorted_years[i]
        previous_year = sorted_years[i - 1]

        current_year_sales = sorted_sales[i]
        previous_year_sales = sorted_sales[i - 1]

        if previous_year_sales is not None and current_year_sales is not None:
            growth = ((current_year_sales - previous_year_sales) / previous_year_sales) * 100
            yoy_growth.append(growth)
        else:
            yoy_growth.append(None)
    
    return yoy_growth