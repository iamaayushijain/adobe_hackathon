# PDF Intelligence – Quick Guide

A lightweight, **offline** toolkit that:

1. Parses PDFs with multiple libraries to capture _all_ text, headings & tables.
2. Builds a clean JSON outline (Title + H1-H3 + full raw text).
3. (Round-1B) Ranks the most relevant sections for a given _persona & task_ using a tiny 80 MB MiniLM model.

## 📽️ Demo Video

[![Watch the demo](https://img.youtube.com/vi/0_dlGD0t930/0.jpg)](https://youtu.be/0_dlGD0t930)

---

## 🔧 Setup

```bash
# clone repo then …
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # all wheels – no Internet needed at runtime
```

Docker version:

```bash
docker build -t pdf-intel .
```

---

## 🚀 Run – Outline-only (Round-1A)

Put PDFs in `input/` (or mount `/app/input`).

## Run by Docker

Build image:

```bash
docker build --platform=linux/amd64 -t pdf-intel:1 .
```

Run image:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-intel:1
```

## Run by Python

Execute **main2.py**:

```bash
python main2.py            # writes JSON to output/<file>.json
```

Internally uses `DocumentPipeline` to generate:

```json
{
  "title": "…",
  "outline": [{ "level": "H1", "text": "…", "page": 1 }],
  "raw_text": [{ "page": 1, "text": "…" }],
  "tables": [{ "page": 1, "data": [["A", "B"]] }]
}
```

---

## 🚀 Run – Persona ranking (Round-1B)

Round-1B uses the **main.py** orchestrator and the sample collections inside `Challenge_1b/`.
Simply run:

```bash
python main.py
```

main.py will:

1. Iterate over every `Challenge_1b/Collection*/` folder.
2. Parse PDFs → `challenge1b_outline_only.json`.
3. Rank headings vs persona/task and write `challenge1b_refined_output.json` with top sections + refined text.

### 🐳 Docker – Round-1B

The sample input PDFs for Round-1B live in the repo at `Challenge_1b/` – each _Collection_ folder already contains the required documents and metadata JSON.

Build the container (BuildKit disabled for deterministic output):

```bash
DOCKER_BUILDKIT=0 docker build -f Dockerfile.main -t pdf-intel-main .
```

Run it while mounting the models, input/output and Challenge-1B collections:

```bash
docker run --rm \
  -e MODEL_PATH=/app/models/all-MiniLM-L6-v2 \
  -e TOKENIZERS_PARALLELISM=false \
  -e OCR_ENABLED=0 \
  -v $PWD/models:/app/models \
  -v $PWD/input:/app/input \
  -v $PWD/output:/app/output \
  -v $PWD/Challenge_1b:/app/Challenge_1b \
  pdf-intel-main
```

After the container finishes, each collection folder will contain a freshly generated
`challenge1b_refined_output.json`, e.g.:

```
Challenge_1b/Collection 1/challenge1b_refined_output.json
Challenge_1b/Collection 2/challenge1b_refined_output.json
Challenge_1b/Collection 3/challenge1b_refined_output.json
```

These files include the ranked sections and refined subsection analyses for every PDF in the collection.

---

## ✨ Why this works so well

- **Four parsers, no blind spots** – pdfplumber, PyMuPDF, pdfminer.six, Camelot.
- **Font-aware heading detection** with fallback heuristics – always returns a hierarchy.
- **Tiny offline model** for semantic ranking (MiniLM 80 MB) – no web calls.
- **CPU-only & fast** – ≈8 s for a 50-page doc; 15 s for the 7-PDF challenge on a laptop.
- **Single command usage** – nothing to configure, logs explain every step.

---

## 📄 Minimal Example

```python
from parser import PDFOutlineParser
outline = PDFOutlineParser().extract_outline("my.pdf")
print(outline["title"], len(outline["outline"]))
```

That’s it – clone, install, run, and get structured JSON ready for search, RAG or reporting.
