import json
from typing import Dict

def financial_model(structured_data: Dict)->Dict:
    """compute the 10 kpis (growth and margins) from the structured statement of operations data"""

    def get_year(entry):
        return entry["report_date"][:4]
    
    def compute_growth(data_dict: Dict[str, float])->Dict[str, float]:
        """compute YoY growth %"""
        years = sorted(data_dict.keys())
        growth={}
        for i in range(len(years)):
            year = years[i]
            if i==0:
                growth[year] = None
            else:
                prev = years[i-1]
                if data_dict[prev]:
                    
                    growth[year] = ((data_dict[year]-data_dict[prev])/data_dict[prev])*100
                    # formula = str(growth[year]) +' = '+'(('+str(data_dict[year])+'-'+str(data_dict[prev])+')/'+str(data_dict[prev])+')*100'
                else:
                    growth[year] = None
        return growth
    
    def compute_margin(numerator_dict: Dict[str, float], denominator_dict: Dict[str, float]) -> Dict[str, float]:
        """compute margin % (numerator/denomenator)*100"""
        all_years = sorted(set(numerator_dict.keys()) | set(denominator_dict.keys()))
    
        margin = {}
        for year in all_years:
            num = numerator_dict.get(year)
            denom = denominator_dict.get(year)
            if num is not None and denom:
                margin[year] = (num / denom) * 100
            else:
                margin[year] = None  # Can't compute if any piece is missing
        return margin

    def format_output(data: Dict[str, float]) -> Dict[str, str]:
        return {
            year: f"{round(val, 2)}%" if val is not None else "N/A"
            for year, val in sorted(data.items())
        }
    
    net_sales = {get_year(e): e["net_sales"]["total"] for e in structured_data}
    cost_sales = {get_year(e): e["cost_of_sales"]["total"] for e in structured_data}
    gross_margin = {get_year(e): e["gross_margin"] for e in structured_data}
    rnd = {get_year(e): e["operating_expenses"]["research_and_development"] for e in structured_data}
    sng = {get_year(e): e["operating_expenses"]["selling_general_and_administrative"] for e in structured_data}
    operating_income = {get_year(e): e["operating_income"] for e in structured_data}
    income_before_income_tax = {get_year(e): e["income_before_tax"] for e in structured_data}
    provision_of_income_tax = {get_year(e): e["tax_provision"] for e in structured_data}
    net_income = {get_year(e): e["net_income"] for e in structured_data}
    eps = {get_year(e): e["earnings_per_share"]["basic"] for e in structured_data}

    rnd_sales = compute_margin(rnd,net_sales)
    net_margin = compute_margin(net_income,net_sales)
    sales_growth = compute_growth(net_sales)
    rnd_growth = compute_growth(rnd_sales)
    margin_growth = compute_growth(net_margin)
    eps_growth = compute_growth(eps)
    kpis = {
        "sales_growth": sales_growth,
        "gross_margin": compute_margin(cost_sales,net_sales),
        "COGS_perc_sales": compute_margin(gross_margin,net_sales),
        "rnd_perc_sales": rnd_sales,
        "rnd_growth": rnd_growth,
        "sng_perc_sales": compute_margin(sng,net_sales),
        "income_margin": compute_margin(operating_income,net_sales),
        "ebt": compute_margin(income_before_income_tax,net_sales),
        "tax_margin": compute_margin(provision_of_income_tax,net_sales),
        "net_margins": net_margin,
        "margin_growth": margin_growth,
        "eps_growth": eps_growth

    }

    return {kpi: format_output(values) for kpi, values in kpis.items()}
