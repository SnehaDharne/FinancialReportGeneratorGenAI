from pydantic import BaseModel, Field
from langchain.tools import tool

class GrossMarginToolArgs(BaseModel):
    gross_margin_data : list = Field(description="list of Gross margin data")
    sales_data : list = Field(description="list of sales data")
    year_data:list = Field(description="list of years of Gross margin data")

@tool(args_schema=GrossMarginToolArgs)
def GrossmmarginpercTool(gross_margin_data: list, year_data: list, sales_data: list) -> list[float]:
    """
    Calculates percentage of Gross Margin.
    This function receives year data and sales data list and Gross margin.

    Arguments:
    - sales_data: List of corresponding sales data.
    - year_data: List of years corresponding to the sales data.
    -gross_margin_data: List of corresponding gross margin data.
    
    Returns:
    - A list of floats representing the YoY gross margin percentage for each year.
    """

    if len(gross_margin_data) != len(sales_data) or len(sales_data) != len(year_data):
        raise ValueError("All input lists must have the same length.")
    if len(year_data) < 2:
        raise ValueError("At least two years of data are required for YoY calculation.")
    data_sorted = sorted(zip(year_data, gross_margin_data, sales_data), key=lambda x: x[0])
    sorted_years, sorted_gross_margins, sorted_sales = zip(*data_sorted)

    yoy_gross_margin_percentage = []

    for i in range(len(sorted_years)):
        if sorted_sales[i] is not None and sorted_sales[i] > 0: 
            gross_margin_percentage = (sorted_gross_margins[i] / sorted_sales[i]) * 100
            yoy_gross_margin_percentage.append(gross_margin_percentage)
        else:
            yoy_gross_margin_percentage.append(None) 

    return yoy_gross_margin_percentage