# PDF Intelligence Dockerfile – CPU-only, offline

FROM python:3.11-slim

# --- OS deps -------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev libssl-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# set workdir
WORKDIR /app

# copy requirements & install (cached layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy project source
COPY app/           ./app/
COPY parser.py pipeline.py output_writer.py utils.py ./
COPY main.py main2.py ./

# create runtime dirs
RUN mkdir -p /app/input /app/output /app/output2

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_DATASETS_OFFLINE=1

# default entrypoint parses Challenge-1B & /app/input
CMD ["python", "main.py"] 