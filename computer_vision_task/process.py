import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import networkx as nx
import easyocr  # Using EasyOCR instead of Tesseract

# Initialize EasyOCR reader (specify languages and GPU usage as needed)
reader = easyocr.Reader(['en'], gpu=False)

def zhang_suen_thinning(binary_image):
    """
    Apply the Zhang–Suen thinning algorithm to a binary image.
    The input image should be a binary image (0 and 255 values). The algorithm
    returns a thinned image where the lines are a single pixel wide.
    """
    # Normalize image to 0 and 1
    image = binary_image.copy() // 255
    prev = np.zeros(image.shape, np.uint8)
    while True:
        padded = np.pad(image, pad_width=1, mode='constant', constant_values=0)
        # Define neighbors using slicing
        p2 = padded[0:-2, 1:-1]   # north
        p3 = padded[0:-2, 2:]     # northeast
        p4 = padded[1:-1, 2:]     # east
        p5 = padded[2:, 2:]       # southeast
        p6 = padded[2:, 1:-1]     # south
        p7 = padded[2:, 0:-2]     # southwest
        p8 = padded[1:-1, 0:-2]   # west
        p9 = padded[0:-2, 0:-2]   # northwest
        p1 = padded[1:-1, 1:-1]   # center

        # Count neighbors and transitions
        B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        A = ((p2 == 0) & (p3 == 1)).astype(np.uint8) + \
            ((p3 == 0) & (p4 == 1)).astype(np.uint8) + \
            ((p4 == 0) & (p5 == 1)).astype(np.uint8) + \
            ((p5 == 0) & (p6 == 1)).astype(np.uint8) + \
            ((p6 == 0) & (p7 == 1)).astype(np.uint8) + \
            ((p7 == 0) & (p8 == 1)).astype(np.uint8) + \
            ((p8 == 0) & (p9 == 1)).astype(np.uint8) + \
            ((p9 == 0) & (p2 == 1)).astype(np.uint8)
        
        # First sub-iteration
        cond1 = (p1 == 1)
        cond2 = (B >= 2) & (B <= 6)
        cond3 = (A == 1)
        cond4 = (p2 * p4 * p6 == 0)
        cond5 = (p4 * p6 * p8 == 0)
        marker = cond1 & cond2 & cond3 & cond4 & cond5
        image[marker] = 0
        
        # Second sub-iteration
        padded = np.pad(image, pad_width=1, mode='constant', constant_values=0)
        p2 = padded[0:-2, 1:-1]
        p3 = padded[0:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, 0:-2]
        p8 = padded[1:-1, 0:-2]
        p9 = padded[0:-2, 0:-2]
        p1 = padded[1:-1, 1:-1]
        
        cond1 = (p1 == 1)
        cond2 = (B >= 2) & (B <= 6)
        cond3 = (A == 1)
        cond4 = (p2 * p4 * p8 == 0)
        cond5 = (p2 * p6 * p8 == 0)
        marker = cond1 & cond2 & cond3 & cond4 & cond5
        image[marker] = 0

        if np.array_equal(image, prev):
            break
        prev = image.copy()
    
    return (image * 255).astype(np.uint8)

def extract_components_from_image(image):
    """
    Extract component candidates from a blueprint image by:
      1. Converting to grayscale and applying Gaussian blur.
      2. Using adaptive thresholding to get a binary mask.
      3. Detecting contours.
      4. Filtering out very small contours (to ignore nested or noisy details),
         including filtering by both contour area and bounding box area.
      5. Performing OCR on an expanded region around each contour.
         (Uses EasyOCR for improved accuracy over Tesseract)
      6. Only adding components with non-empty OCR results.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    components = []
    comp_id = 1
    height, width = image.shape[:2]
    
    # Define thresholds.
    min_contour_area = 100        # minimum contour area (in pixels)
    min_bounding_box_area = 600   # minimum bounding box area
    
    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_box_area = w * h
        
        if w > 0.9 * width or h > 0.9 * height:
            continue
        
        if bounding_box_area < min_bounding_box_area:
            continue
        
        # Expand ROI by a margin to capture nearby text.
        margin = 30
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, width)
        y2 = min(y + h + margin, height)
        
        roi = image[y1:y2, x1:x2]
        # Use EasyOCR to extract text from the ROI.
        ocr_result = reader.readtext(roi, detail=0, paragraph=True)
        ocr_text = " ".join(ocr_result).strip()
        
        # Only add component if OCR detects some text.
        if not ocr_text:
            continue
        
        component = {
            "id": comp_id,
            "bounding_box": (x, y, x + w, y + h),
            "specs": ocr_text
        }
        components.append(component)
        comp_id += 1
    
    return components

def detect_connections_from_image(image, components, min_line_length=10):
    """
    Detect connections between components using edge detection and the Hough Line Transform.
    This version first skeletonizes the image using Zhang–Suen thinning and applies Canny edge detection.
    Then, for each line detected by HoughLinesP, it checks if the line
    actually intersects two different component bounding boxes. Only such lines are
    considered valid connections.
    """
    def ccw(A, B, C):
        # Helper to check counter-clockwise order.
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def segments_intersect(A, B, C, D):
        # Returns True if line segment AB intersects line segment CD.
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def line_intersects_rect(x1, y1, x2, y2, rect, tolerance=5):
        """
        Check if the line segment from (x1,y1) to (x2,y2) intersects the given rectangle.
        The rectangle is given as (xmin, ymin, xmax, ymax). A tolerance expands the rectangle slightly.
        """
        xmin, ymin, xmax, ymax = rect
        # Expand the rectangle by tolerance.
        xmin_t, ymin_t, xmax_t, ymax_t = xmin - tolerance, ymin - tolerance, xmax + tolerance, ymax + tolerance
        
        # If either endpoint lies within the expanded rectangle, we consider it intersecting.
        if (xmin_t <= x1 <= xmax_t and ymin_t <= y1 <= ymax_t) or (xmin_t <= x2 <= xmax_t and ymin_t <= y2 <= ymax_t):
            return True
        
        # Otherwise, check if the segment crosses any of the rectangle's edges.
        A = (x1, y1)
        B = (x2, y2)
        edges = [
            ((xmin_t, ymin_t), (xmax_t, ymin_t)),
            ((xmax_t, ymin_t), (xmax_t, ymax_t)),
            ((xmax_t, ymax_t), (xmin_t, ymax_t)),
            ((xmin_t, ymax_t), (xmin_t, ymin_t))
        ]
        for C, D in edges:
            if segments_intersect(A, B, C, D):
                return True
        return False

    # Convert the image to grayscale and binarize.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Apply Zhang–Suen thinning to get a skeletonized image.
    thinned = zhang_suen_thinning(binary)
    
    # Run Canny edge detection on the thinned image.
    edges = cv2.Canny(thinned, 50, 150, apertureSize=3)
    
    # Use HoughLinesP to extract line segments.
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=50,
        minLineLength=min_line_length, maxLineGap=10
    )
    
    connections = set()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            connected_ids = set()
            # Check each component to see if the line segment touches its bounding box.
            for comp in components:
                rect = comp["bounding_box"]
                if line_intersects_rect(x1, y1, x2, y2, rect):
                    connected_ids.add(comp["id"])
            # Only consider lines that connect exactly two distinct components.
            if len(connected_ids) == 2:
                connection = tuple(sorted(connected_ids))
                connections.add(connection)
    return list(connections)


def build_graph(components, connections):
    """
    Build a NetworkX graph with nodes as components and edges as detected connections.
    The nodes will contain only:
      - component_id
      - bounding_box
      - specifications
    """
    G = nx.Graph()
    for comp in components:
        G.add_node(
            comp["id"],
            component_id=comp["id"],
            bounding_box=str(comp["bounding_box"]),
            specifications=comp["specs"]
        )
    for conn in connections:
        G.add_edge(conn[0], conn[1])
    return G

def process_pdf(pdf_path, output_folder="output"):
    """
    Process a PDF file where each page is a blueprint image:
      - Convert pages to images.
      - Extract components (with OCR) and detect connections.
      - Build and return a graph for each page.
      - Save annotated images with bounding boxes and connection lines.
    
    Returns:
        Tuple[dict, dict]: Mappings from page numbers to detected components and graphs.
    """
    pages = convert_from_path(pdf_path, dpi=200)
    all_components = {}
    all_graphs = {}
    os.makedirs(output_folder, exist_ok=True)
    
    for page_num, page_image in enumerate(pages, start=1):
        # Convert PIL image to OpenCV format (BGR)
        open_cv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        components = extract_components_from_image(open_cv_image)
        connections = detect_connections_from_image(open_cv_image, components, min_line_length=10)
        graph = build_graph(components, connections)
        all_components[page_num] = components
        all_graphs[page_num] = graph
        
        # Annotate image with bounding boxes and connection lines.
        annotated_image = open_cv_image.copy()
        for comp in components:
            x_min, y_min, x_max, y_max = comp["bounding_box"]
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Annotate using the component id.
            cv2.putText(annotated_image, str(comp["id"]), (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for conn in connections:
            comp1 = next(c for c in components if c["id"] == conn[0])
            comp2 = next(c for c in components if c["id"] == conn[1])
            x1, y1, x2, y2 = comp1["bounding_box"]
            center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
            x1, y1, x2, y2 = comp2["bounding_box"]
            center2 = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.line(annotated_image, center1, center2, (255, 0, 0), 2)
        
        annotated_image_path = os.path.join(output_folder, f"page_{page_num}.jpg")
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"Page {page_num}: {len(components)} components and {len(connections)} connections detected. Saved annotated image to {annotated_image_path}.")
    
    return all_components, all_graphs

def main():
    pdf_path = "data/p&id/diagram.pdf"  # Update this path as needed.
    components_by_page, graphs_by_page = process_pdf(pdf_path)
    
    # Print summary details.
    for page, comps in components_by_page.items():
        print(f"\nPage {page}: {len(comps)} components detected.")
        for comp in comps:
            print(f"ID {comp['id']}:")
            print(f"  Specs: {comp['specs']}")
    for page, graph in graphs_by_page.items():
        print(f"\nGraph for Page {page}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

if __name__ == "__main__":
    main()
