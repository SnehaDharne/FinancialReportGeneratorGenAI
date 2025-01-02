from pydantic import BaseModel, Field
from langchain.tools import tool

class expensespercArgs(BaseModel):
    t_o_e_data : list = Field(description="list of total operating expenses data")
    sales_data : list = Field(description="list of sales data")
    year_data:list = Field(description="list of years of total operating expenses data")


@tool(args_schema=expensespercArgs)
def ExpensespercTool(t_o_e_data: list, year_data: list, sales_data: list) -> list[float]:
    """
    Calculates expenses percentage sales.
    This function receives year data and sales data list and total operating expenses.

    Arguments:
    - sales_data: List of corresponding sales data.
    - year_data: List of years corresponding to the sales data.
    - t_o_e_data: List of corresponding total operating expenses data.
    
    Returns:
    - A list of floats representing the  expenses percentage sales for each year.
    """

    if len(t_o_e_data) != len(sales_data) or len(sales_data) != len(year_data):
        raise ValueError("All input lists must have the same length.")
    if len(year_data) < 2:
        raise ValueError("At least two years of data are required for YoY calculation.")

    data_sorted = sorted(zip(year_data, t_o_e_data, sales_data), key=lambda x: x[0])
    sorted_years, sorted_t_o_e_data, sorted_sales = zip(*data_sorted)

    yoy_expenses_percentage = []

    for i in range(len(sorted_years)):
        if sorted_sales[i] is not None and sorted_sales[i] > 0:  
            expenses_percentage = (sorted_t_o_e_data[i] / sorted_sales[i]) * 100
            yoy_expenses_percentage.append(expenses_percentage)
        else:
            yoy_expenses_percentage.append(None) 
    return yoy_expenses_percentage