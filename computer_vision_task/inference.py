#!/usr/bin/env python3
import os
import docx2txt
from scanner import process_pdf # Make sure processing.py exports this function
import requests

def load_sop_text(sop_path):
    """
    Load the SOP document and return a set of words (lowercase, no punctuation).
    """
    return docx2txt.process(sop_path)


def call_claude_model(prompt: str) -> str:
    """
    Calls the Anthropic Claude API to evaluate component specifications against the SOP.
    
    This implementation uses Anthropic's current API structure for messages endpoint.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        data = response.json()
        return data.get("content", [{}])[0].get("text", "No response generated")
    except Exception as e:
        print(f"API call error details: {str(e)}")
        if hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        return f"Error calling Claude API: {str(e)}"

def evaluate_component_against_sop(component_specs: str, sop: str) -> str:
    """
    Evaluate a component's specifications against a given SOP using an LLM (Claude model).
    
    The prompt template has been refined to be more clear:
    
    You are an expert in technical documentation and product specifications.
    You are provided with a component's specifications and a partial SOP guideline.
    Identify which areas break SOP guidelines.
    
    Component Specifications:
    {component_specs}
    
    SOP Guideline:
    {sop}
    
    Provide a clear, concise evaluation and suggestions for corrections if necessary.
    """
    prompt = (
        "You are provided with a component's specifications and a partial SOP guideline.\n"
        "Analyze the specifications to determine whether they conform to the SOP range provided.\n"
        f"Component Specifications:\n{component_specs}\n\n"
        f"SOP Guideline:\n{sop}\n\n"
        "Provide a clear, concise evaluation and suggestions for corrections if necessary."
    )
    
    # Call the Claude model with the prompt.
    response = call_claude_model(prompt)
    return response

def generate_report(sop_path="data/sop/sop.docx", pdf_path="data/p&id/diagram.pdf"):
    # Load SOP words
    sop_words = load_sop_text(sop_path)
    
    # Find matching components on each page
    component_by_page = process_pdf(pdf_path)
    
    # Store results in a string
    full_report = ""
    
    for page, component in component_by_page.items():
        component_specs = component['specs']
        response = evaluate_component_against_sop(component_specs, sop_words)
        
        page_report = f"Page {page} Component Specs: {component_specs}\n"
        page_report += f"Page {page} Evaluation: {response}\n"
        page_report += "---------------------------------------------------\n"
        
        # Print to console
        print(page_report)
        
        # Add to full report string
        full_report += page_report
    
    return full_report


def main():
    print(load_sop_text("data/sop/sop.docx"))

if __name__ == "__main__":
    main()
