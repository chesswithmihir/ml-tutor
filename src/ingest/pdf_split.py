# src/ingest/pdf_split.py

import pdfplumber, pathlib, json, uuid
from pdf2image import convert_from_path

RAW = pathlib.Path("data/raw/notes")
PAGE_DIR = pathlib.Path("data/processed/pages")
PAGE_DIR.mkdir(parents=True, exist_ok=True)
chunks = []

for pdf in RAW.glob("*.pdf"):
    with pdfplumber.open(pdf) as doc:
        for page_num, page in enumerate(doc.pages, 1):
            page_id = uuid.uuid4().hex
            # save image (300 DPI)
            img_path = PAGE_DIR / f"{page_id}.png"
            convert_from_path(pdf, dpi=300, first_page=page_num,
                              last_page=page_num)[0].save(img_path)
            # extract text
            text = page.extract_text(y_tolerance=2) or ""
            chunks.append({
                "id": page_id,
                "pdf": pdf.name,
                "page": page_num,
                "source": "notes",
                "text": text.strip()
            })

with open("data/processed/chunks.jsonl", "w") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(f"Saved {len(chunks)} chunks")
