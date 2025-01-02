import json
from pydantic import BaseModel, Field
from langchain.tools import tool

class RnDPercArgs(BaseModel):
    RND_data : list = Field(description="list of item:{operating expenses} category:{research and development}")
    sales_data : list = Field(description="list of sales data")
    year_data:list = Field(description="list of years")


@tool(args_schema=RnDPercArgs)
def RNDTool(RND_data: list, year_data: list, sales_data: list) -> list[float]:
    """
    Calculates research and development percentage sales for  item:'operating expenses' category:'research and development'
    This function receives year data and sales data list and  item:'operating expenses' category:'research and development'.

    Arguments:
    - sales_data: List of corresponding sales data.
    - year_data: List of years corresponding to the sales data.
    - RND_data: list of item:{operating expenses} category:{research and development}
    
    Returns:
    - A list of floats representing the research and development percentage sales for each year.
    """

    if len(RND_data) != len(sales_data) or len(sales_data) != len(year_data):
        raise ValueError("All input lists must have the same length.")

    if len(year_data) < 2:
        raise ValueError("At least two years of data are required for YoY calculation.")

    data_sorted = sorted(zip(year_data, RND_data, sales_data), key=lambda x: x[0])

    sorted_years, sorted_RND_data, sorted_sales = zip(*data_sorted)

    yoy_RND_percentage = []

    for i in range(len(sorted_years)):
        if sorted_sales[i] is not None and sorted_sales[i] > 0:  
            net_margin_percentage = (sorted_RND_data[i] / sorted_sales[i]) * 100
            yoy_RND_percentage.append(net_margin_percentage)
        else:
            yoy_RND_percentage.append(None) 

    return yoy_RND_percentage