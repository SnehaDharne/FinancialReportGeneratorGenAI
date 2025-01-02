import json
from pydantic import BaseModel, Field
from langchain.tools import tool

class GrowthToolArgs(BaseModel):
    sales_data : list = Field(description="list of sales data")
    year_data:list = Field(description="list of years of sales data")

@tool(args_schema=GrowthToolArgs)
def Growth_Tool(sales_data: list, year_data: list) -> list[float]:
    """
    Calculates year-on-year (YoY) growth for 'Net Sales' for a specific company.
    This function receives year data and sales data list.

    Arguments:
    - sales_data: List of corresponding sales.
    - year_data: List of years corresponding to the sales data.
    
    Returns:
    - A list of floats representing the YoY growth percentage for each year compared to the previous year.
    """
    if len(sales_data) < 2:
        return ["Not enough data to calculate year-on-year sales growth."]
    
    yoy_growth = []
    sales_data_sorted = sorted(zip(year_data, sales_data), key=lambda x: x[0])
    sorted_years, sorted_sales = zip(*sales_data_sorted)

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