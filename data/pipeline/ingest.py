"""
Data Ingestion Module.

Loads PDFs, splits them into chunks, generates embeddings,
and bulk-indexes them into OpenSearch.
"""
import os
import glob
import argparse
import sys
import re
from typing import List, Dict

from opensearchpy import OpenSearch, helpers
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import config

client = OpenSearch(
    hosts=[{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
    http_auth=(config.OPENSEARCH_USER, config.OPENSEARCH_PASSWORD),
    use_ssl=False, verify_certs=False, ssl_show_warn=False, ssl_assert_hostname=False,
)

embedder = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True, "batch_size": config.EMBED_BATCH_SIZE},
)

def load_pages(folder: str) -> List[Dict]:
    """
    Loads all PDF files from a folder into a list of page dictionaries.
    Applies regex cleaning to fix broken lines and excessive whitespace.
    """
    pdfs = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not pdfs:
        print(f"Error: No PDFs found in: {folder}", file=sys.stderr)
        return []
        
    pages = []
    print(f"Loading {len(pdfs)} PDF files from {folder}...")
    
    for f in pdfs:
        try:
            # PyPDFLoader load each page as a document
            for i, d in enumerate(PyPDFLoader(f).load(), start=1):
                raw_text = (d.page_content or "").strip()
                
                if raw_text:
                    text_clean = re.sub(r'[ \t]+', ' ', raw_text)
                    
                    text_clean = re.sub(r'\n{2,}', '___PARAGRAPH___', text_clean)
                    
                    text_clean = re.sub(r'\n', ' ', text_clean)
                    
                    # Restore the actual paragraphs
                    text_clean = text_clean.replace('___PARAGRAPH___', '\n')
                    text_clean = re.sub(r' +', ' ', text_clean).strip()
                    
                    pages.append({
                        "file_name": os.path.basename(f), 
                        "page": i, 
                        "text": text_clean
                    })
                    
        except Exception as e:
            print(f"Error processing file {f}: {e}. Skipping.", file=sys.stderr)
            
    if not pages:
        print("Error: No text could be extracted from any PDF.", file=sys.stderr)
        
    return pages

def make_chunks(pages: List[Dict], chunk_size: int, chunk_overlap: int) -> (List[str], List[Dict]):
    """
    Splits document pages into text chunks based on specified size and overlap.
    """
    print(f"Creating chunks... (chunk_size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    texts, metas = [], []
    for p in pages:
        for j, doc in enumerate(splitter.create_documents([p["text"]])):
            texts.append(doc.page_content)
            metas.append({"file_name": p["file_name"], "page": p["page"], "chunk_id": j})
    print(f"Generated {len(texts)} chunks from {len(pages)} pages.")
    return texts, metas

def bulk_index(texts: List[str], metas: List[Dict]) -> int:
    """
    Embeds and bulk indexes documents into OpenSearch.
    """
    if not texts:
        print("No texts to index.")
        return 0
        
    print(f"Generating embeddings and indexing {len(texts)} chunks in OpenSearch...")
    vecs = embedder.embed_documents(texts)
    
    actions = [{
        "_op_type": "index",
        "_index": config.OPENSEARCH_INDEX,
        "_source": {"text": t, "metadata": m, "embedding": v},
    } for t, m, v in zip(texts, metas, vecs)]
    
    if actions:
        helpers.bulk(client, actions, chunk_size=500, request_timeout=120)
        client.indices.refresh(index=config.OPENSEARCH_INDEX)
        
    print("Indexing complete.")
    return len(actions)

def knn_test(q: str, k: int = 5) -> None:
    """
    Runs a sample k-NN search to test the index.
    """
    qv = embedder.embed_query(q)
    body = {"size": k, "query": {"knn": {"embedding": {"vector": qv, "k": k}}}, "_source": ["text", "metadata"]}
    res = client.search(index=config.OPENSEARCH_INDEX, body=body)
    print("\n--- k-NN Test Results ---")
    for i, h in enumerate(res.get("hits", {}).get("hits", []), 1):
        s = h["_source"]; m = s["metadata"]; prev = s["text"].replace("\n", " ")
        print(f"[{i}] {m['file_name']} p.{m['page']}  score={h['_score']:.3f}")
        print(f"    {prev[:160]}{'...' if len(prev) > 160 else ''}")
    print("---------------------------")

if __name__ == "__main__":
    """
    Main function for standalone script execution.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="Recreate the index")
    ap.add_argument("--test_query", type=str, help="k-NN test query")
    ap.add_argument("--pdf_dir", default=config.PDF_DIR, help="Directory to read PDFs from")
    args = ap.parse_args()

    if args.recreate:
        from setup_opensearch import get_client, recreate_index
        print("Recreating index from standalone ingest script...")
        recreate_index(get_client(), config.OPENSEARCH_INDEX)
    
    pages = load_pages(args.pdf_dir)
    if not pages:
        sys.exit(1)
    
    chunk_size_local = int(config.CHUNK_SIZE_ENV or 600)
    chunk_overlap_local = int(config.CHUNK_OVERLAP_ENV or 120)

    texts, metas = make_chunks(pages, chunk_size_local, chunk_overlap_local)
    n = bulk_index(texts, metas)

    num_pdfs = len({p["file_name"] for p in pages})
    print(f"Done. PDFs={num_pdfs} | pages={len(pages)} | chunks={n} | index={config.OPENSEARCH_INDEX} | chunk={chunk_size_local}/{chunk_overlap_local} | model={config.EMBEDDING_MODEL}")

    if args.test_query:
        knn_test(args.test_query)