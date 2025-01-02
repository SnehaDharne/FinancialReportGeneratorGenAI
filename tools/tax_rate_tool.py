import json
from pydantic import BaseModel, Field
from langchain.tools import tool


class TaxratesArgs(BaseModel):
    Provision_of_IT_data : list = Field(description="list of provision for income taxes data")
    sales_data : list = Field(description="list of sales data")
    year_data:list = Field(description="list of years of total operating expenses data")

@tool(args_schema=TaxratesArgs)
def TaxrateTool(Provision_of_IT_data: list, year_data: list, sales_data: list) -> list[float]:
    """
    Calculates tax rate percentage.
    This function receives year data and sales data list and provision for income tax.

    Arguments:
    - sales_data: List of corresponding sales data.
    - year_data: List of years corresponding to the sales data.
    - Provision_of_IT_data: List of corresponding provision for income tax.
    
    Returns:
    - A list of floats representing the tax rate percentage for each year.
    """

    if len(Provision_of_IT_data) != len(sales_data) or len(sales_data) != len(year_data):
        raise ValueError("All input lists must have the same length.")

    if len(year_data) < 2:
        raise ValueError("At least two years of data are required for YoY calculation.")

    data_sorted = sorted(zip(year_data, Provision_of_IT_data, sales_data), key=lambda x: x[0])

    sorted_years, sorted_IT_data, sorted_sales = zip(*data_sorted)

    yoy_Tax_Rate_percentage = []

    for i in range(len(sorted_years)):
        if sorted_sales[i] is not None and sorted_sales[i] > 0: 
            tax_percentage = (sorted_IT_data[i] / sorted_sales[i]) * 100
            yoy_Tax_Rate_percentage.append(tax_percentage)
        else:
            yoy_Tax_Rate_percentage.append(None)

    return yoy_Tax_Rate_percentage