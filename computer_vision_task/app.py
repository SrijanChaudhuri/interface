import os
from flask import Flask, render_template, request
# Use the correct import for process_pdf (choose one, comment out the other)
from process import process_pdf  
# from blueprint_processor import process_pdf  

import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    graph_images = []  # List to hold tuples of (page_number, base64 image)
    if request.method == 'POST':
        pdf_path = request.form.get('pdf_path')
        if pdf_path and os.path.exists(pdf_path):
            try:
                # Process the PDF (this may take some time)
                components_by_page, graphs_by_page = process_pdf(pdf_path)
                if graphs_by_page:
                    # For each page, generate a graph image.
                    for page_num in sorted(graphs_by_page.keys()):
                        G = graphs_by_page[page_num]
                        plt.figure(figsize=(8, 6))
                        pos = nx.spring_layout(G)
                        nx.draw_networkx(
                            G, pos, with_labels=True,
                            node_color='lightblue', edge_color='black',
                            font_size=10
                        )
                        plt.axis('off')
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        plt.close()
                        buf.seek(0)
                        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        graph_images.append((page_num, img_base64))
                else:
                    error = "No graphs were found in the PDF."
            except Exception as e:
                error = f"An error occurred while processing the PDF: {e}"
        else:
            error = "Invalid file path. Please ensure the file exists."
    return render_template('index.html', graph_images=graph_images, error=error)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
