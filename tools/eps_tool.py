from pydantic import BaseModel, Field
from langchain.tools import tool

class EPSToolArgs(BaseModel):
    EPS_data : list = Field(description="list of item:{earnings per share} category:{basic}")
    year_data:list = Field(description="list of years of earnings per share")

@tool(args_schema=EPSToolArgs)
def EPS_Tool(EPS_data: list, year_data: list) -> list[float]:
    """
    Calculates year-on-year (YoY) growth for item:'earnings per share' category:'basic' for a specific company.
    This function receives year data and 'earnings per share''basic'.

    Arguments:
    - EPS_data: List of 'earnings per share' 'basic'.
    - year_data: List of corresponding years.
    
    Returns:
    - A list of floats representing the YoY growth percentage for each year compared to the previous year.
    """
    if len(EPS_data) < 2:
        return ["Not enough data to calculate year-on-year sales growth."]
    
    yoy_growth = []
    sales_data_sorted = sorted(zip(year_data, EPS_data), key=lambda x: x[0])
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