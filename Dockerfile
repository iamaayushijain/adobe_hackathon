# Dockerfile.main2
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements2.txt ./
RUN pip install --no-cache-dir -r requirements2.txt

COPY . .

RUN mkdir -p /app/input /app/output /app/output2

CMD ["python", "main2.py"] 