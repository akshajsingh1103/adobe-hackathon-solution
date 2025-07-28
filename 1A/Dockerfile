# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system packages
RUN apt-get update && \
    apt-get install -y build-essential poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run the script
CMD ["python", "extractor.py"]
