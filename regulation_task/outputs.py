import os
import gradio as gr
import anthropic
from datetime import datetime
from comparison import load_text, process_specific_reg_file
from inference import create_prompt, generate_report_with_claude, save_report

# Initialize the Anthropic client with your API key
client = anthropic.Anthropic(
    # Remove this line in production and use environment variables instead
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

def get_available_regulations():
    """Get list of available regulation files"""
    reg_folder = os.path.join("parsed", "regulations")
    if os.path.isdir(reg_folder):
        return [f for f in os.listdir(reg_folder) if f.endswith("_extracted.txt")]
    return []

def analyze_regulation(reg_file):
    """Analyze default SOP against a selected regulation file"""
    if not reg_file:
        return "Please select a regulation document."
    
    # Use the default SOP file
    sop_folder = os.path.join("parsed", "sop")
    sop_file = os.path.join(sop_folder, "original_extracted.txt")
    
    if not os.path.exists(sop_file):
        return f"Default SOP file '{sop_file}' not found!"
    
    reg_path = os.path.join("parsed", "regulations", reg_file)
    
    # Process the files to find relevant clauses
    relevant_clauses = process_specific_reg_file(sop_file, reg_path)
    
    if not relevant_clauses:
        return "No relevant clauses found between the SOP and the regulation."
    
    # Load the SOP text
    sop_text = load_text(sop_file)
    
    # Create the prompt
    prompt = create_prompt(sop_text, relevant_clauses, reg_file)
    
    # Generate the report
    report = generate_report_with_claude(prompt)
    
    # Save the report
    report_path = save_report(report, reg_file)
    
    return f"Analysis complete! Report saved to {report_path}\n\n{report}"

# Create the Gradio interface
with gr.Blocks(title="Regulatory Compliance Analysis System") as demo:
    gr.Markdown("# Regulatory Compliance Analysis System")
    
    gr.Markdown("## Select a Regulation to Analyze Against Standard SOP")
    reg_dropdown = gr.Dropdown(
        choices=get_available_regulations(),
        label="Select Regulatory Document"
    )
    
    analyze_btn = gr.Button("Analyze Regulation")
    result = gr.Textbox(label="Analysis Result", lines=20)
    
    analyze_btn.click(
        fn=analyze_regulation,
        inputs=reg_dropdown,
        outputs=result
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()