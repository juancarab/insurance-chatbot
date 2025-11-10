"""
Main Data Ingestion Pipeline Orchestrator.

This script runs the full data pipeline in sequence:
1. Download PDFs from S3.
2. Set up (recreate) the OpenSearch index.
3. Run EDA on PDFs to determine optimal chunking.
4. Run ingestion (load, chunk, embed, index) using EDA results.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import config
from download_from_s3 import download_pdfs_from_s3
from setup_opensearch import get_client as get_os_client, recreate_index, MAPPING as OS_MAPPING
from eda_policies import analyze_pdf, propose_chunking
from ingest import load_pages, make_chunks, bulk_index

def download(skip: bool) -> str:
    """
    Pipeline Step 1: Download PDFs from S3.
    """
    print("--- [STEP 1 of 4] PDF Download ---")
    local_dir = config.PDF_DIR
    
    if skip:
        print("Skipping S3 download as per --skip-download flag.")
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        return local_dir

    if not config.S3_AWS_ACCESS_KEY_ID or not config.S3_AWS_SECRET_ACCESS_KEY:
        print("Warning: S3 credentials not found. Skipping download.", file=sys.stderr)
        print("Assuming files already exist in:", local_dir)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        return local_dir

    count = download_pdfs_from_s3(
        bucket=config.S3_BUCKET,
        prefix=config.S3_PREFIX,
        key=config.S3_AWS_ACCESS_KEY_ID,
        secret=config.S3_AWS_SECRET_ACCESS_KEY,
        local_dir=local_dir
    )
    if count == 0:
        raise RuntimeError(f"No files downloaded or found. Check S3 settings or local directory {local_dir}")
    
    print(f"Download complete. {count} files ready in {local_dir}")
    return local_dir


def setup_opensearch():
    """
    Pipeline Step 2: Recreate the OpenSearch index.
    """
    print(f"--- [STEP 2 of 4] Setup OpenSearch Index ---")
    client = get_os_client()
    recreate_index(client, config.OPENSEARCH_INDEX, OS_MAPPING)
    print(f"Index '{config.OPENSEARCH_INDEX}' is clean and ready.")


def run_eda(pdf_dir: str, skip: bool) -> Dict[str, Any]:
    """
    Pipeline Step 3: Run EDA to get chunking parameters.
    """
    print(f"--- [STEP 3 of 4] Run EDA ---")
    
    if skip:
        print("Skipping EDA as per --skip-eda flag.")
        default_recs = {
            "chunk_size": int(config.CHUNK_SIZE_ENV or 600),
            "chunk_overlap": int(config.CHUNK_OVERLAP_ENV or 120),
        }
        print(f"Using default chunk params: {default_recs}")
        return default_recs
        
    pdf_paths = list(Path(pdf_dir).glob("**/*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir} to run EDA.")

    print(f"Analyzing {len(pdf_paths)} PDF files for EDA...")
    stats: List[Any] = [analyze_pdf(p) for p in pdf_paths]
    recs = propose_chunking(stats)

    out_path = Path(config.EDA_OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    recs_file_path = Path(config.EDA_RECS_FILE)
    
    with open(recs_file_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)

    print(f"EDA complete. Recommendations saved to {recs_file_path}")
    print(f"  -> Recommended chunk_size: {recs.get('chunk_size')}")
    print(f"  -> Recommended chunk_overlap: {recs.get('chunk_overlap')}")
    
    return recs


def run_ingest(pdf_dir: str, chunk_params: Dict[str, Any]):
    """
    Pipeline Step 4: Run the data ingestion.
    """
    chunk_size = chunk_params.get('chunk_size')
    chunk_overlap = chunk_params.get('chunk_overlap')

    print(f"--- [STEP 4 of 4] Run Ingestion ---")
    print(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    pages = load_pages(pdf_dir)
    if not pages:
        raise RuntimeError("Load PDFs returned no pages. Ingestion halted.")

    texts, metas = make_chunks(pages, chunk_size, chunk_overlap)
    
    n = bulk_index(texts, metas)

    print("\n--- Ingestion Summary ---")
    print(f"  PDFs processed: {len(set(p['file_name'] for p in pages))}")
    print(f"  Pages loaded:   {len(pages)}")
    print(f"  Chunks created: {n}")
    print(f"  Index:          {config.OPENSEARCH_INDEX}")
    print("---------------------------")


def main():
    """
    Main orchestrator for the data pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the full data ingestion pipeline.")
    parser.add_argument("--skip-download", action="store_true", help="Skip the S3 download step")
    parser.add_argument("--skip-eda", action="store_true", help="Skip the EDA step and use default chunking")
    args = parser.parse_args()

    print("======= STARTING DATA INGESTION PIPELINE =======")
    try:
        pdf_dir = download(skip=args.skip_download)
        
        setup_opensearch()
        
        chunk_recs = run_eda(pdf_dir, skip=args.skip_eda)
        
        run_ingest(pdf_dir, chunk_recs)
        
        print("\n======= PIPELINE COMPLETED SUCCESSFULLY =======")
        
    except Exception as e:
        print(f"\n======= PIPELINE FAILED =======", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()