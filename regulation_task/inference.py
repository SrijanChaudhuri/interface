import os
import sys
import json
import anthropic
from datetime import datetime
from comparison import load_text, process_specific_reg_file, process_regulatory_docs, extract_clauses

# Initialize the Anthropic client with your API key
client = anthropic.Anthropic(
    # Remove this line in production and use environment variables instead
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

def create_prompt(sop_text, relevant_clauses, reg_name):
    """
    Creates a prompt for Claude to analyze the SOP against regulatory clauses.
    
    Args:
        sop_text (str): The text of the SOP document
        relevant_clauses (list): List of (clause, score) tuples from the regulatory document
        reg_name (str): Name of the regulatory document
        
    Returns:
        str: A formatted prompt for Claude
    """
    clauses_text = "\n\n".join([f"Clause (score: {score:.3f}): {clause}" for clause, score in relevant_clauses])
    
    prompt = f"""
You are an expert in regulatory compliance analysis. Your task is to analyze a Standard Operating Procedure (SOP) document and determine its compliance with relevant regulatory clauses.

### SOP DOCUMENT:
{sop_text}

### RELEVANT REGULATORY CLAUSES FROM {reg_name}:
{clauses_text}

Please provide a detailed compliance analysis report that includes:
1. Overview of the SOP document's purpose and scope
2. Analysis of each relevant regulatory clause and how it applies to the SOP
3. Identification of any compliance issues or gaps in the SOP
4. Specific recommendations for adjustments to bring the SOP into full compliance
5. Summary of compliance status

Format your analysis as a professional compliance report with clear sections and actionable insights.
"""
    return prompt

def generate_report_with_claude(prompt, max_tokens=4000):
    """
    Generate a compliance report using Claude.
    
    Args:
        prompt (str): The prompt for Claude
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The generated report
    """
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error generating report with Claude: {e}")
        return f"Error: {str(e)}"

def save_report(report, reg_name, output_dir="reports"):
    """
    Saves the generated report to a file.
    
    Args:
        report (str): The generated compliance report
        reg_name (str): Name of the regulatory document
        output_dir (str): Directory to save the report
        
    Returns:
        str: Path to the saved report file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a filename based on the regulation name and timestamp
    base_name = os.path.splitext(os.path.basename(reg_name))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"compliance_report_{base_name}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Save the report
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    
    return filepath

def analyze_specific_regulation(sop_file, reg_file_path):
    """
    Analyzes a specific regulatory file against the SOP.
    
    Args:
        sop_file (str): Path to the SOP file
        reg_file_path (str): Path to the regulatory file
        
    Returns:
        str: Path to the saved report
    """
    # Get the regulatory file name
    reg_name = os.path.basename(reg_file_path)
    
    # Process the regulatory file to find relevant clauses
    relevant_clauses = process_specific_reg_file(sop_file, reg_file_path)
    
    if not relevant_clauses:
        print(f"No relevant clauses found in {reg_name}.")
        return None
    
    # Load the SOP text
    sop_text = load_text(sop_file)
    
    # Create the prompt
    prompt = create_prompt(sop_text, relevant_clauses, reg_name)
    
    # Generate the report
    print(f"Generating compliance report for {reg_name}...")
    report = generate_report_with_claude(prompt)
    
    # Save the report
    report_path = save_report(report, reg_name)
    print(f"Report saved to {report_path}")
    
    return report_path

def analyze_all_regulations(sop_file, regulatory_folder):
    """
    Analyzes all regulatory files in the specified folder against the SOP.
    
    Args:
        sop_file (str): Path to the SOP file
        regulatory_folder (str): Path to the folder containing regulatory files
        
    Returns:
        list: Paths to all saved reports
    """
    # Process all regulatory documents
    results = process_regulatory_docs(sop_file, regulatory_folder)
    
    if not results:
        print("No relevant clauses found in any regulatory document.")
        return []
    
    # Load the SOP text
    sop_text = load_text(sop_file)
    
    report_paths = []
    for doc_name, clauses in results:
        # Create the prompt
        prompt = create_prompt(sop_text, clauses, doc_name)
        
        # Generate the report
        print(f"Generating compliance report for {doc_name}...")
        report = generate_report_with_claude(prompt)
        
        # Save the report
        report_path = save_report(report, doc_name)
        print(f"Report saved to {report_path}")
        report_paths.append(report_path)
    
    return report_paths