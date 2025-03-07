import os
import sys
import fitz  # PyMuPDF for PDF processing
import pytesseract
from PIL import Image
import io
from docx import Document

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file, including text within images using OCR.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        str: The combined text extracted from the PDF.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF '{pdf_path}': {e}")
        return ""

    full_text = ""
    for page_number in range(len(doc)):
        page = doc[page_number]
        full_text += f"\n--- Page {page_number + 1} Text ---\n"
        page_text = page.get_text()
        full_text += page_text
    
    return full_text

def extract_text_from_docx(docx_path: str) -> str:
    """
    Extracts text from a DOCX file.

    Args:
        docx_path (str): The file path to the DOCX.

    Returns:
        str: The combined text extracted from the DOCX.
    """
    try:
        doc = Document(docx_path)
    except Exception as e:
        print(f"Error opening DOCX '{docx_path}': {e}")
        return ""

    full_text = ""
    for para in doc.paragraphs:
        full_text += para.text + "\n"
    
    return full_text

def process_all_files(root_dir: str = ".") -> None:
    """
    Recursively processes all PDF and DOCX files under the root directory.
    Extracted text from each file is stored in a 'parsed/' directory,
    preserving the relative directory structure.

    Args:
        root_dir (str): The root directory to search for files.
                        Defaults to the current directory.
    """
    output_root = "parsed"
    # Ensure the output root directory exists
    os.makedirs(output_root, exist_ok=True)

    # Walk through the root directory recursively
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                input_file_path = os.path.join(dirpath, filename)
                print(f"Processing PDF: {input_file_path} ...")
                extracted_text = extract_text_from_pdf(input_file_path)
            elif filename.lower().endswith(".docx"):
                input_file_path = os.path.join(dirpath, filename)
                print(f"Processing DOCX: {input_file_path} ...")
                extracted_text = extract_text_from_docx(input_file_path)
            else:
                continue

            # Compute the relative path to preserve directory structure
            relative_dir = os.path.relpath(dirpath, root_dir)
            output_dir = os.path.join(output_root, relative_dir)
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, f"{base_name}_extracted.txt")
            try:
                with open(output_file, "w", encoding="utf-8") as out_file:
                    out_file.write(extracted_text)
                print(f"Saved extracted text to: {output_file}\n")
            except Exception as e:
                print(f"Error saving output for '{input_file_path}': {e}")

if __name__ == '__main__':
    # Optionally, allow specifying a root directory as a command-line argument.
    # Otherwise, default to the current directory.
    root_directory = "./data"
    process_all_files(root_directory)