FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends libmupdf-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt . 
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src
COPY models/ ./models/

ENTRYPOINT ["python", "src/main.py", "--input_dir", "/app/input", "--output_dir", "/app/output"]
