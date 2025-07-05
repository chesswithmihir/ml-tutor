# src/ingest/exam_split.py

import pdfplumber
from pdf2image import convert_from_path
import pathlib
import uuid
import json
import re

# Where your raw PDFs live:
BLANK_DIR = pathlib.Path("data/raw/exams_blank")
SOL_DIR   = pathlib.Path("data/raw/exams_sol")

# Where we'll dump page images + JSONL records:
PAGE_IMG_DIR = pathlib.Path("data/processed/exam_pages")
PAGE_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = pathlib.Path("data/processed/exam_chunks.jsonl")

def extract_questions_from_text(text: str, page_num: int) -> list:
    """
    Split OCR text into individual questions based on question markers.
    Returns list of (question_id, question_text) tuples.
    """
    if not text.strip():
        return []
    
    # Common question patterns
    patterns = [
        r'^\s*Q\s*(\d+)\.?\s*',  # Q1. or Q 1 or Q1
        r'^\s*Question\s+(\d+)\.?\s*',  # Question 1. or Question 1
        r'^\s*Problem\s+(\d+)\.?\s*',   # Problem 1. or Problem 1
        r'^\s*(\d+)\.?\s*\[.*?\]',      # 1. [20 pts] or 1 [20 pts]
        r'^\s*(\d+)\.?\s*\(',           # 1. (20 points) or 1 (20 points)
    ]
    
    lines = text.split('\n')
    questions = []
    current_question = None
    current_text = []
    
    for line in lines:
        # Check if this line starts a new question
        found_question = False
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE | re.MULTILINE)
            if match:
                # Save previous question if exists
                if current_question and current_text:
                    questions.append((current_question, '\n'.join(current_text).strip()))
                
                # Start new question
                current_question = f"Q{match.group(1)}"
                current_text = [line]
                found_question = True
                break
        
        if not found_question:
            # Add to current question
            if current_question:
                current_text.append(line)
            else:
                # No question started yet, might be header text
                # Try to detect if this could be a question without clear numbering
                if any(keyword in line.lower() for keyword in ['question', 'problem', 'solve', 'find', 'prove']):
                    current_question = f"Q{len(questions) + 1}"
                    current_text = [line]
    
    # Don't forget the last question
    if current_question and current_text:
        questions.append((current_question, '\n'.join(current_text).strip()))
    
    # If no questions found, treat entire text as one question
    if not questions and text.strip():
        questions.append((f"Q{page_num}", text.strip()))
    
    return questions

def ingest_pdf(pdf_path: pathlib.Path, source: str):
    """
    Split the given PDF into per-question records.
    source is either "exam_blank" or "exam_solution".
    """
    # Derive a canonical exam name, e.g. "midf15blank.pdf" → "midf15"
    stem = pdf_path.stem
    if source == "exam_blank":
        exam = stem.replace("blank", "")
    else:
        exam = stem  # mids13.pdf → mids13

    with pdfplumber.open(pdf_path) as doc:
        for page_num, page in enumerate(doc.pages, start=1):
            # 1) Save a PNG snapshot at 300 DPI
            images = convert_from_path(str(pdf_path), dpi=300,
                                     first_page=page_num,
                                     last_page=page_num)
            if not images:
                continue
                
            img = images[0]
            page_id = uuid.uuid4().hex
            img_path = PAGE_IMG_DIR / f"{page_id}.png"
            img.save(img_path)

            # 2) Extract text
            text = page.extract_text(y_tolerance=2) or ""
            
            # 3) Split into questions
            questions = extract_questions_from_text(text, page_num)
            
            # 4) Yield one record per question
            for question_id, question_text in questions:
                yield {
                    "id": f"{page_id}_{question_id}",
                    "exam": exam,
                    "question_id": question_id,
                    "page": page_num,
                    "source": source,
                    "pdf": pdf_path.name,
                    "text": question_text.strip(),
                    "page_image": f"/exam_pages/{page_id}.png",
                    # Link to the counterpart PDF:
                    "linked_pdf": f"{exam}.pdf" if source=="exam_blank" else f"{exam}blank.pdf"
                }

def main():
    records = []

    # Ingest all the blank exams
    for pdf in sorted(BLANK_DIR.glob("*.pdf")):
        print(f"Processing blank exam: {pdf.name}")
        records.extend(ingest_pdf(pdf, source="exam_blank"))

    # Ingest all the solution exams
    for pdf in sorted(SOL_DIR.glob("*.pdf")):
        print(f"Processing solution exam: {pdf.name}")
        records.extend(ingest_pdf(pdf, source="exam_solution"))

    # Write out to JSONL for downstream chunking/embedding
    with OUT_FILE.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} exam question chunks to {OUT_FILE}")

if __name__ == "__main__":
    main()