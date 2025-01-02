from pydantic import BaseModel, Field
from langchain.tools import tool

class NETMarginArgs(BaseModel):
    net_income_data : list = Field(description="list of net income")
    sales_data : list = Field(description="list of sales data")
    year_data:list = Field(description="list of years")

@tool(args_schema=NETMarginArgs)
def NetMarginTool(net_income_data: list, year_data: list, sales_data: list) -> list[float]:
    """
    Calculates Net margin.
    This function receives year data and sales data list and net income.

    Arguments:
    - sales_data: List of corresponding sales data.
    - year_data: List of years corresponding to the sales data.
    - IT_data: List of corresponding net income.
    
    Returns:
    - A list of floats representing net margin percentage for each year.
    """

    if len(net_income_data) != len(sales_data) or len(sales_data) != len(year_data):
        raise ValueError("All input lists must have the same length.")
    if len(year_data) < 2:
        raise ValueError("At least two years of data are required for YoY calculation.")

    data_sorted = sorted(zip(year_data, net_income_data, sales_data), key=lambda x: x[0])

    sorted_years, sorted_net_income_data, sorted_sales = zip(*data_sorted)

    yoy_net_margin_percentage = []

    for i in range(len(sorted_years)):
        if sorted_sales[i] is not None and sorted_sales[i] > 0:  
            net_margin_percentage = (sorted_net_income_data[i] / sorted_sales[i]) * 100
            yoy_net_margin_percentage.append(net_margin_percentage)
        else:
            yoy_net_margin_percentage.append(None) 

    return yoy_net_margin_percentage