import json
from pydantic import BaseModel, Field
from langchain.tools import tool

class SNGPercArgs(BaseModel):
    SNG_data : list = Field(description="list of item:{operating expenses} category:{selling, general and administrative}")
    sales_data : list = Field(description="list of sales data")
    year_data:list = Field(description="list of years")
    
@tool(args_schema=SNGPercArgs)
def SNGTool(SNG_data: list, year_data: list, sales_data: list) -> list[float]:
    """
    Calculates selling and general percentage sales growth yoy for  item:'operating expenses' category:'selling, general and administrative'
    This function receives year data and sales data list and  item:'operating expenses' category:'selling, general and administrative'.

    Arguments:
    - sales_data: List of corresponding sales data.
    - year_data: List of years corresponding to the sales data.
    - SNG_data: list of item:{operating expenses} category:{selling, general and administrative}
    
    Returns:
    - A list of floats representing the selling and general percentage sales for each year.
    """

    if len(SNG_data) != len(sales_data) or len(sales_data) != len(year_data):
        raise ValueError("All input lists must have the same length.")

    if len(year_data) < 2:
        raise ValueError("At least two years of data are required for YoY calculation.")

    data_sorted = sorted(zip(year_data, SNG_data, sales_data), key=lambda x: x[0])

    sorted_years, sorted_SNG_data, sorted_sales = zip(*data_sorted)

    yoy_SNG_percentage = []

    for i in range(len(sorted_years)):
        if sorted_sales[i] is not None and sorted_sales[i] > 0: 
            net_margin_percentage = (sorted_SNG_data[i] / sorted_sales[i]) * 100
            yoy_SNG_percentage.append(net_margin_percentage)
        else:
            yoy_SNG_percentage.append(None) 

    return yoy_SNG_percentage


