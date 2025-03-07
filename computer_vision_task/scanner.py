import os
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path

def extract_components_from_image(image):
    """
    Extracts a component from the top section of the image defined as:
      - Vertical range: 0 to height/5
      - Horizontal range: 0 to width
    The component is a dictionary with:
      - id: a unique identifier.
      - bounding_box: (x_min, y_min, x_max, y_max) for the text region.
      - label: the first non-empty line of OCR output.
      - specs: the full OCR extracted text.
    
    Args:
        image (np.array): The OpenCV BGR image.
    
    Returns:
        dict: A component dictionary.
    """
    # Get dimensions of the image.
    h, w = image.shape[:2]
    # Define the ROI as the top section.
    roi = image[0: h // 8, 0: w]

    # Perform OCR on the ROI.
    ocr_text = pytesseract.image_to_string(roi)
    specs = ocr_text.strip()

    # The bounding box for the ROI.
    bounding_box = (0, 0, w, h // 8)
    
    component = {
        "id": 1,
        "bounding_box": bounding_box,
        "specs": specs
    }
            
    return component


def process_pdf(pdf_path, output_folder="output"):
    """
    Process a PDF file where each page is an image:
      - Convert pages to images.
      - Extract components (by iterating over the entire image with OCR).
      - Annotate the image with bounding boxes.
    
    Returns:
        dict: Mapping from page numbers to detected components.
    """
    pages = convert_from_path(pdf_path, dpi=200)
    all_components = {}
    os.makedirs(output_folder, exist_ok=True)
    
    for page_num, page_image in enumerate(pages, start=1):
        # Convert PIL image to OpenCV BGR image.
        open_cv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        
        # Extract components from the entire image.
        component = extract_components_from_image(open_cv_image)
        
        # Annotate the image: draw bounding boxes and IDs for each component.
        annotated_image = open_cv_image.copy()
        x_min, y_min, x_max, y_max = component["bounding_box"]
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(annotated_image, str(component["id"]), (x_min, max(y_min - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the annotated image.
        annotated_image_path = os.path.join(output_folder, f"page_{page_num}.jpg")
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"Page {page_num}: component detected. Saved annotated image to {annotated_image_path}.")
        
        # Save the components for this page.
        all_components[page_num] = component
        
        
    return all_components

def main():
    pdf_path = "data/p&id/diagram.pdf"  # Update this path as needed.
    components_by_page = process_pdf(pdf_path)
    
    # Print summary details.
    for page, comps in components_by_page.items():
        print(f"\nPage {page}: {len(comps)} component(s) detected.")
        for comp in comps:
            print(f"ID {comp['id']}:")
            print(f"  Bounding Box: {comp['bounding_box']}")
            print(f"  Specs: {comp['specs']}")

if __name__ == "__main__":
    main()
