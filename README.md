# PDF Intelligence – Quick Guide

A lightweight, **offline** toolkit that:

1. Parses PDFs with multiple libraries to capture *all* text, headings & tables.
2. Builds a clean JSON outline (Title + H1-H3 + full raw text).
3. (Round-1B) Ranks the most relevant sections for a given *persona & task* using a tiny 80 MB MiniLM model.

---

## 🔧  Setup

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

## 🚀  Run – Outline-only (Round-1A)

Put PDFs in `input/` (or mount `/app/input`).  
Execute **main2.py**:
```bash
python main2.py            # writes JSON to output/<file>.json
```
Internally uses `DocumentPipeline` to generate:
```json
{
  "title": "…",
  "outline": [ {"level":"H1","text":"…","page":1} ],
  "raw_text": [{"page":1,"text":"…"}],
  "tables":   [{"page":1,"data":[["A","B"]]}]
}
```

---

## 🚀  Run – Persona ranking (Round-1B)

Round-1B uses the **main.py** orchestrator and the sample collections inside `Challenge_1b/`.
Simply run:
```bash
python main.py
```
main.py will:
1. Iterate over every `Challenge_1b/Collection*/` folder.  
2. Parse PDFs → `challenge1b_outline_only.json`.  
3. Rank headings vs persona/task and write `challenge1b_refined_output.json` with top sections + refined text.

---

## ✨  Why this works so well

* **Four parsers, no blind spots** – pdfplumber, PyMuPDF, pdfminer.six, Camelot.
* **Font-aware heading detection** with fallback heuristics – always returns a hierarchy.
* **Tiny offline model** for semantic ranking (MiniLM 80 MB) – no web calls.
* **CPU-only & fast** – ≈8 s for a 50-page doc; 15 s for the 7-PDF challenge on a laptop.
* **Single command usage** – nothing to configure, logs explain every step.

---

## 📄  Minimal Example

```python
from parser import PDFOutlineParser
outline = PDFOutlineParser().extract_outline("my.pdf")
print(outline["title"], len(outline["outline"]))
```

That’s it – clone, install, run, and get structured JSON ready for search, RAG or reporting. 