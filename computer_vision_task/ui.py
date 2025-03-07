#!/usr/bin/env python3
import json
import gradio as gr
from inference import generate_report  # Import generate_report from inference.py

def generate_report_ui():
    """
    Gradio UI function with no inputs.
    Uses default file paths for SOP and PDF.
    """
    try:
        # Define default file paths.
        sop_path = "data/sop/sop.docx"
        pdf_path = "data/p&id/diagram.pdf"
        
        # Generate the report using the default paths.
        report = generate_report(sop_path=sop_path, pdf_path=pdf_path)
        
        # Return the report as a formatted JSON string.
        return json.dumps(report, indent=2)
    except Exception as e:
        return f"Error generating report: {e}"

iface = gr.Interface(
    fn=generate_report_ui,
    inputs=[],  # No inputs are provided
    outputs=gr.Textbox(label="Report"),
    title="SOP & Diagram Report Generator",
    description="Click 'Submit' to generate a report using default file paths for the SOP and diagram."
)

if __name__ == "__main__":
    print("Starting Gradio interface... at port 7860")
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

