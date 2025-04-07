from typing import Dict, List, Any

def organize_extracted_documents(extracted_docs: Dict[str, List[Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organize extracted documents into a structured format by statement type and year.
    
    Args:
        extracted_docs: Dictionary containing documents by year
        
    Returns:
        Dictionary containing organized statements by type
    """
    statements = {
        'operations': [],
        'balance_sheet': [],
        'cash_flow': []
    }
    
    for year, docs in extracted_docs.items():
        for doc in docs:
            content = doc.page_content
            if "CONSOLIDATED STATEMENTS OF OPERATIONS" in content:
                statements['operations'].append({
                    'year': year,
                    'content': content
                })
            elif "CONSOLIDATED BALANCE SHEETS" in content:
                statements['balance_sheet'].append({
                    'year': year,
                    'content': content
                })
            elif "CONSOLIDATED STATEMENTS OF CASH FLOWS" in content:
                statements['cash_flow'].append({
                    'year': year,
                    'content': content
                })
    
    return statements

def get_statements_by_year(statements: Dict[str, List[Dict[str, Any]]], year: str) -> Dict[str, str]:
    """
    Get all statements for a specific year.
    
    Args:
        statements: Organized statements dictionary
        year: Year to filter by
        
    Returns:
        Dictionary containing statements for the specified year
    """
    year_statements = {}
    for statement_type, docs in statements.items():
        for doc in docs:
            if doc['year'] == year:
                year_statements[statement_type] = doc['content']
    return year_statements 