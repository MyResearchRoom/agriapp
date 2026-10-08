import json
import os
import sys
from pathlib import Path
import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Knowledge base lives beside the backend by default, with an override for deployments.
KNOWLEDGE_BASE_DIR = Path(
    os.getenv(
        "AGRIAI_KNOWLEDGE_BASE_DIR",
        str(Path(__file__).resolve().parent.parent / "knowledge_base"),
    )
)
OUTPUT_DIR = Path(__file__).parent / "vector_store"
FAISS_INDEX_PATH = OUTPUT_DIR / "faiss_index.bin"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def discover_pdf_files(base_dir: Path):
    """Recursively find all PDF files in the specified directory."""
    if not base_dir.exists():
        print(f"ERROR: Knowledge base directory '{base_dir}' does not exist.")
        sys.exit(1)
        
    pdf_files = sorted(list(base_dir.rglob("*.pdf")))
    return pdf_files


def extract_and_chunk_pdfs(pdf_files, base_dir: Path):
    """Extract text from PDFs page by page and split into metadata-enriched chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    all_chunks = []
    processed_files_count = 0
    skipped_files_count = 0

    print(f"\n[Step 2/7] Extracting text and chunking {len(pdf_files)} PDF document(s)...")
    
    for pdf_path in pdf_files:
        rel_path = pdf_path.relative_to(base_dir)
        source_folder = rel_path.parent.name if str(rel_path.parent) != "." else base_dir.name
        source_file = pdf_path.name

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            print(f"  WARNING: Could not open '{pdf_path}': {e}. Skipping file.")
            skipped_files_count += 1
            continue

        file_has_text = False
        file_chunk_index = 0

        for page_idx, page in enumerate(doc):
            page_text = page.get_text("text").strip()
            if not page_text:
                continue

            file_has_text = True
            page_num = page_idx + 1  # 1-indexed page number
            chunks = text_splitter.split_text(page_text)

            for chunk_text in chunks:
                if chunk_text.strip():
                    all_chunks.append({
                        "chunk_text": chunk_text,
                        "source_file": source_file,
                        "source_folder": source_folder,
                        "page_number": page_num,
                        "chunk_index": file_chunk_index
                    })
                    file_chunk_index += 1

        doc.close()

        if not file_has_text:
            print(f"  WARNING: File '{pdf_path.name}' has no extractable text (possibly scanned/image-only). Skipping.")
            skipped_files_count += 1
        else:
            processed_files_count += 1

    return all_chunks, processed_files_count, skipped_files_count


def main():
    print("==================================================")
    print(" AgriAI RAG Stage 1: Knowledge Base Ingestion ")
    print("==================================================")
    
    # 1. Discover files
    print(f"\n[Step 1/7] Discovering PDFs in '{KNOWLEDGE_BASE_DIR}'...")
    pdf_files = discover_pdf_files(KNOWLEDGE_BASE_DIR)
    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        rel_path = pdf.relative_to(KNOWLEDGE_BASE_DIR)
        print(f"  - {rel_path}")

    if not pdf_files:
        print("No PDF files found to ingest. Exiting.")
        return

    # 2 & 3. Extract text & Chunk
    chunks, processed_count, skipped_count = extract_and_chunk_pdfs(pdf_files, KNOWLEDGE_BASE_DIR)
    print(f"Chunking complete. Total chunks generated: {len(chunks)}")
    
    if not chunks:
        print("ERROR: No extractable text chunks found in any PDF. Exiting.")
        sys.exit(1)

    # 4. Embed locally
    print(f"\n[Step 4/7] Loading local embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Generating dense embeddings for {len(chunks)} chunk(s)...")
    chunk_texts = [c["chunk_text"] for c in chunks]
    raw_embeddings = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)
    
    # L2 Normalization for Cosine Similarity via Inner Product
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = (raw_embeddings / norms).astype(np.float32)

    dimension = normalized_embeddings.shape[1]
    print(f"Embeddings generated successfully. Dimension: {dimension}")

    # 5. Build FAISS index
    print("\n[Step 5/7] Building FAISS Index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # 6. Persist to disk
    print(f"\n[Step 6/7] Persisting FAISS index and metadata to '{OUTPUT_DIR}'...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    index_size_bytes = FAISS_INDEX_PATH.stat().st_size
    index_size_kb = index_size_bytes / 1024.0

    # 7. Print final summary
    print("\n==================================================")
    print(" INGESTION SUMMARY ")
    print("==================================================")
    print(f"Total PDFs Discovered : {len(pdf_files)}")
    print(f"PDFs Processed        : {processed_count}")
    print(f"PDFs Skipped/Warned   : {skipped_count}")
    print(f"Total Chunks Generated: {len(chunks)}")
    print(f"Embedding Model       : {MODEL_NAME}")
    print(f"Embedding Dimension   : {dimension}")
    print(f"FAISS Vector Count    : {index.ntotal}")
    print(f"FAISS Index Size      : {index_size_kb:.2f} KB ({index_size_bytes} bytes)")
    print(f"FAISS Index Path      : {FAISS_INDEX_PATH}")
    print(f"Metadata JSON Path    : {METADATA_PATH}")
    print("==================================================")
    

if __name__ == "__main__":
    main()


# ==================================================
#  INGESTION SUMMARY
# ==================================================
# Total PDFs Discovered : 8
# PDFs Processed        : 4
# PDFs Skipped/Warned   : 4
# Total Chunks Generated: 512
# Embedding Model       : all-MiniLM-L6-v2
# Embedding Dimension   : 384
# FAISS Vector Count    : 512
# FAISS Index Size      : 768.04 KB (786477 bytes)
# FAISS Index Path      : C:\AgriAi\rag_backend\vector_store\faiss_index.bin
# Metadata JSON Path    : C:\AgriAi\rag_backend\vector_store\metadata.json
# ==================================================