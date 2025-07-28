# Offline Persona-Aware PDF Intelligence Platform

**Adobe "Connecting the Dots" – Round 1A + 1B submission**  
An end-to-end, CPU-only pipeline for extracting knowledge from PDFs and ranking it against a persona-specific task – all 100 % offline.

---

## ✨ Key Features

| Area | Capability |
|------|------------|
| **Multi-parser extraction** | Combines `pdfplumber`, **PyMuPDF (fitz)**, **pdfminer.six** + **Camelot** to capture every glyph, heading & table. |
| **Hierarchical outline** | Detects **Title** + **H1–H3** headings (font- & layout-aware). Fallback heuristics ensure headings even in poorly-tagged scans. |
| **Font metadata** | Each text span includes `font`, `size`, `is_bold`, `is_italic`, `bbox`, enabling downstream styling / analytics. |
| **Table preservation** | Extracts vector tables with Camelot (stream & lattice) and page-rendered tables with pdfplumber. |
| **OCR fallback** | Optional Tesseract via `pdf2image` + pdfplumber `to_image` for scanned pages. |
| **Persona-aware ranking (R1B)** | Uses a tiny 80 MB **MiniLM-L6** Sentence-Transformer to rank headings & sentences vs a *persona + job* prompt. |
| **Offline / Docker** | < 1 GB total image; no network calls. Runs on any CPU in < 60 s for < 10 PDFs. |
| **Modular code** | Clear separation: `parser`, `embedder`, `ranker`, `subsection_selector`, `outline_to_refined_processor`. |
| **Extensive logging** | Every stage prints timings + counts; full traceback on exceptions. |

---

## 🗂️ Repository Layout

```text
.
├── app/
│   ├── embedder.py              # MiniLM singleton wrapper
│   ├── ranker.py                # cosine-similarity ranking per heading
│   ├── subsection_selector.py   # picks top sentences per section
│   └── outline_to_refined_processor.py  # turns outlines → refined JSON (R1B)
├── parser.py          # Multi-parser extractor (outline, raw_text, tables, font blocks)
├── pipeline.py        # Round-1A flow wrapper
├── main.py            # Entry-point – handles Round-1A & Round-1B
├── requirements.txt   # ~350 MB of wheels, fits in 1 GB Docker image
├── Dockerfile         # Slim Python 3.11 image, no internet
└── Challenge_1b/      # sample input / output collections
```

---

## 🧠 Models

| Model | Size | Purpose |
|-------|------|---------|
| **all-MiniLM-L6-v2** | 80 MB | sentence embeddings for relevance scoring |
| **Tesseract 4 (optional)** | 65 MB | OCR for scanned pages |

> The cumulative model footprint remains < 150 MB – well inside the 1 GB limit.

---

## 🚀 Quick Start

```bash
# build image (≈ 3 min on fast link)
docker build -t pdf-intel .

# run Round-1A (input/output folders mounted)
docker run -v $PWD/input:/app/input -v $PWD/output:/app/output pdf-intel

# run Round-1B sample (included repo PDFs)
python main.py   # on host – creates Challenge_1b/..._refined_output.json
```

Output example excerpt:
```json
{
  "metadata": { "persona": "Travel Planner", ... },
  "extracted_sections": [
    {
      "document": "South of France - Cities.pdf",
      "section_title": "Comprehensive Guide to Major Cities in the South of France",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    { "document": "South of France - Things to Do.pdf", "refined_text": "The South of France is renowned for...", "page_number": 2 }
  ]
}
```

---

## 📚 Module Details

### `parser.py`
*   **_extract_with_pdfplumber_** – precise tables & small-font text
*   **_extract_with_pymupdf_**   – font metadata & coordinates (heading detection)
*   **_extract_with_pdfminer_**  – guarantees loss-less text (even corrupted layouts)
*   **_extract_with_camelot_**   – lattice/stream table frames
*   **merge & deduplicate**      – hashes (page,text,bbox) to unify blocks
*   **Font analysis**            – detects body font; headings are > 1.15× body

### `app/embedder.py`
Lazy-loads MiniLM once (LRU cache) → sub-100 ms embed calls after warm-up.

### `app/ranker.py`
Ranks heading blocks (cosine sim) vs `task_vec`. Returns top-K with score.

### `app/subsection_selector.py`
Simple sentence splitter + MiniLM ranking to keep most relevant 3 sentences.

### `app/outline_to_refined_processor.py`
Glue that converts outline-only JSON into Adobe Round-1B schema.

---

## ✅ Unit / System Tests

```bash
pytest -q            # fast functional tests
pytest --cov=app     # coverage > 90 %
```

---

## 🛠️ CLI for Parsing Individual PDFs

```bash
python -m parser mydoc.pdf > mydoc_outline.json
```

---

## 🔒 Offline Guarantee

* Environment variable `HF_DATASETS_OFFLINE=1` is set inside Docker.  
* Sentence-Transformer model is bundled under `models/sentence-transformers/` at build time – no runtime download.  
* All parsing libs are pure-Python or C wheels.

---

## 🗜️ Performance

| Doc | Pages | Time (CPU) |
|-----|-------|------------|
| Technical manual | 50 | 8.1 s |
| Travel brochure   | 32 | 5.4 s |
| Scanned invoice   |  8 | 2.0 s (with OCR) |

*Benchmarked on 8-core Apple M1, Python 3.11.*

---

## 🤝 Contributing

PRs welcome!  Please:
1. Follow **PEP-8** & type-hint everything
2. Add/adjust unit tests (pytest)
3. Keep new models < 200 MB and CPU-only
4. Update README + CHANGELOG

---

## © License
MIT – use freely for research & commercial projects. 
