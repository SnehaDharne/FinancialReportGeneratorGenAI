import os
from dotenv import load_dotenv
import re, json
load_dotenv()  
from langchain.document_loaders import PyPDFLoader
key_string = os.getenv('GEMINI_API_KEY')
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain

def pdf_loader(pdf_file):
    temp_file_path = f"temp_{pdf_file.name}"
    
    # Save the uploaded file temporarily
    with open(temp_file_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    # Load the PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    # Clean up the temporary file
    os.remove(temp_file_path)
    return documents

def page_finder(document):
    start_page = None
    end_page = None

    for doc in document:
        if doc.page_content.startswith("Item 8"):
            start_page = doc.metadata['page']
            break 

    for doc in document:
        if doc.page_content.startswith("Item 9"):
            end_page = doc.metadata['page']
            break 

    if start_page is None or end_page is None:
        raise ValueError("Could not find 'Item 8' or 'Item 9' in the document")

    return start_page, end_page


def page_executor(document):
    start_page, end_page = page_finder(document)
    extracted_documents = [
        doc for doc in document
        if start_page <= doc.metadata['page'] < end_page
    ]

    return extracted_documents
def executor_main(pdf_files):
    final_document = []
    for pdf_file in pdf_files:
        document = pdf_loader(pdf_file)
        extracted_docs = page_executor(document)
        for r in extracted_docs:
            if "CONSOLIDATED STATEMENTS OF OPERATIONS" in r.page_content:
                final_document.append(r)
    return final_document

def get_data_from_docs(pdf_files):
    doc = executor_main(pdf_files)
    extract_prompt = PromptTemplate(
        input_variables=["document"],
        template="""
    You are a financial data extraction assistant.

    Your job is to extract the STATEMENTS OF OPERATIONS (also known as Income Statement or Profit & Loss) from the document below into a consistent JSON format, matching standard 10-K structure.
    There might be multiple statements of operations in the document.
    Document:
    {document}

    Return a JSON array where each object represents a **single fiscal year** and contains:

    [
    {{
        "report_date": "YYYY-MM-DD",
        "section": "profit_and_loss",
        "net_sales": {{
        "products": ...,
        "services": ...,
        "total": ...
        }},
        "cost_of_sales": {{
        "products": ...,
        "services": ...,
        "total": ...
        }},
        "gross_margin": ...,
        "operating_expenses": {{
        "research_and_development": ...,
        "selling_general_and_administrative": ...,
        "total": ...
        }},
        "operating_income": ...,
        "other_income_or_expense_net": ...,
        "income_before_tax": ...,
        "tax_provision": ...,
        "net_income": ...,
        "earnings_per_share": {{
        "basic": ...,
        "diluted": ...
        }},
        "shares_used_in_computing_eps": {{
        "basic": ...,
        "diluted": ...
        }}
        "metadata" : {{
            "page_number" : ...,
            "source" : ...
        }}
    }},
    ...
    ]

    Instructions:
    - Parse values from tables and text if needed.
    - Normalize field names (e.g., "SG&A" → "selling_general_and_administrative").
    - Use `null` for missing values.
    - Use `"report_date"` in ISO format: YYYY-MM-DD (e.g., “Sep 30, 2024” → “2024-09-30”).
    - Return only the valid JSON array with all numeric values as numbers (not strings).
    - Include the metadata for each entry in the json array
    - please make sure the list of dictionaries returned is in the order of the report_date
    - please make sure all the report_dates are covered and we dont miss any report dates
    - please make sure we have only one entry for each report date, pritoritize the most complete and recent data for each repeated report date
    """
    )


    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0, google_api_key=key_string)

    extract_chain = LLMChain(
        llm=llm,
        prompt=extract_prompt,
        output_key="json_output"  # customize output key
    )


    response = extract_chain.run(document=doc)

    response = response.strip()
    response = response.replace('```json\n', '')
    response = response.replace('```', '')
    response = response.strip()

    structured_data = json.loads(response)
    return structured_data



