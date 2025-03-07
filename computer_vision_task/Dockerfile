# Use a lightweight Python image.
FROM python:3.9-slim

# Set the working directory.
WORKDIR /app

# Install system dependencies and clean up apt cache.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Accept Anthropic API key as a build argument and set it as an environment variable.
ARG ANTHROPIC_API_KEY
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

# Copy the requirements file.
COPY requirements_reduced.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements_reduced.txt

# Copy only the necessary application files into the container.
COPY ui.py .
COPY scanner.py .

# Expose the default Gradio port.
EXPOSE 7860

# Run the Gradio application.
CMD ["python", "ui.py"]
