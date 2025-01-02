from pydantic import BaseModel, Field
from langchain.tools import tool

class EBTArgs(BaseModel):
    IT_data : list = Field(description="list of income before provision for income taxes")
    sales_data : list = Field(description="list of sales data")
    year_data:list = Field(description="list of years of total operating expenses data")

import json
@tool(args_schema=EBTArgs)
def EBTTool(IT_data: list, year_data: list, sales_data: list) -> list[float]:
    """
    Calculates EBT percentage.
    This function receives year data and sales data list and income before provision for income taxes.

    Arguments:
    - sales_data: List of corresponding sales data.
    - year_data: List of years corresponding to the sales data.
    - IT_data: List of corresponding income before provision for income taxes.
    
    Returns:
    - A list of floats representing the tax rate percentage for each year.
    """

    if len(IT_data) != len(sales_data) or len(sales_data) != len(year_data):
        raise ValueError("All input lists must have the same length.")
    if len(year_data) < 2:
        raise ValueError("At least two years of data are required for YoY calculation.")

    data_sorted = sorted(zip(year_data, IT_data, sales_data), key=lambda x: x[0])

    sorted_years, sorted_IT_data, sorted_sales = zip(*data_sorted)

    yoy_EBT_percentage = []

    for i in range(len(sorted_years)):
        if sorted_sales[i] is not None and sorted_sales[i] > 0:  
            EBT_percentage = (sorted_IT_data[i] / sorted_sales[i]) * 100
            yoy_EBT_percentage.append(EBT_percentage)
        else:
            yoy_EBT_percentage.append(None) 

    return yoy_EBT_percentage