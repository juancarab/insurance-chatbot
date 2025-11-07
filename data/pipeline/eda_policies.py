# source/eda_policies.py
#!/usr/bin/env python3
"""
EDA mínimo para PDFs de pólizas.

Genera:
- eda_out/eda_summary_by_file.csv      (resumen por archivo)
- eda_out/eda_recommendations.json     (chunk_size y chunk_overlap sugeridos)
"""
import os, re, json, argparse
from pathlib import Path
from statistics import mean
from dataclasses import dataclass, asdict
from typing import List
import pandas as pd
from pypdf import PdfReader

# --------- Config vía entorno (.env si está disponible) ---------
try:
    from dotenv import load_dotenv  # opcional
    load_dotenv()
except Exception:
    pass

PDF_DIR_ENV = os.getenv("PDF_DIR", "./data/raw_policies")
CHUNK_SIZE_ENV = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP_ENV = int(os.getenv("CHUNK_OVERLAP", "120"))

# ---------- utilidades ----------
def percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    # interpolación lineal
    return float(xs[f] * (c - k) + xs[c] * (k - f))

def split_paragraphs(text: str) -> List[str]:
    text = text.replace("\r", "\n")
    if "\n\n" in text:
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    else:
        # fallback por oraciones
        parts = [p.strip() for p in re.split(r"(?<=[\.\!\?])\s+(?=[A-ZÁÉÍÓÚÑ0-9])", text) if p.strip()]
    return parts

@dataclass
class FileStats:
    file: str
    pages: int
    total_chars: int
    avg_chars_per_page: float
    paragraphs: int
    avg_paragraph_chars: float
    p90_paragraph_chars: float

def analyze_pdf(path: Path) -> FileStats:
    try:
        reader = PdfReader(str(path))
    except Exception:
        # si no se puede abrir, devolvemos ceros
        return FileStats(path.name, 0, 0, 0.0, 0, 0.0, 0.0)

    para_lengths: List[int] = []
    total_chars, total_paras = 0, 0

    for p in reader.pages:
        try:
            txt = (p.extract_text() or "").strip()
        except Exception:
            txt = ""
        paras = split_paragraphs(txt)
        para_lengths.extend(len(x) for x in paras)
        total_paras += len(paras)
        total_chars += len(txt)

    avg_para = mean(para_lengths) if para_lengths else 0.0
    p90_para = percentile(para_lengths, 90) if para_lengths else 0.0
    pages = len(reader.pages) if hasattr(reader, "pages") else 0

    return FileStats(
        file=path.name,
        pages=pages,
        total_chars=total_chars,
        avg_chars_per_page=(total_chars / pages) if pages else 0.0,
        paragraphs=total_paras,
        avg_paragraph_chars=avg_para,
        p90_paragraph_chars=p90_para,
    )

def propose_chunking(stats: List[FileStats]) -> dict:
    """
    Sugerencia basada en el p90 promedio del tamaño de párrafo.
    Permite sobreescribir con CHUNK_SIZE/CHUNK_OVERLAP del entorno.
    """
    p90s = [s.p90_paragraph_chars for s in stats if s.p90_paragraph_chars > 0]
    # base sugerida por EDA
    base = int(mean(p90s) * 1.25) if p90s else 1200
    chunk_size_auto = max(400, min(1600, base))
    chunk_overlap_auto = min(250, max(120, int(chunk_size_auto * 0.18)))

    # si el entorno define valores, prevalecen
    chunk_size = CHUNK_SIZE_ENV if CHUNK_SIZE_ENV else chunk_size_auto
    chunk_overlap = CHUNK_OVERLAP_ENV if CHUNK_OVERLAP_ENV else chunk_overlap_auto

    return {
        "files_analyzed": len(stats),
        "p90_paragraph_mean": round(mean(p90s), 1) if p90s else 0.0,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "source": {
            "chunk_size": "env" if CHUNK_SIZE_ENV else "eda",
            "chunk_overlap": "env" if CHUNK_OVERLAP_ENV else "eda",
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=PDF_DIR_ENV, help="Carpeta con PDFs (defecto: PDF_DIR del entorno)")
    ap.add_argument("--out_dir", default="./eda_out", help="Carpeta de salida (defecto: ./eda_out)")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(data_dir.glob("**/*.pdf"))
    if not pdfs:
        raise SystemExit(f"No se encontraron PDFs en {data_dir}")

    stats: List[FileStats] = [analyze_pdf(p) for p in pdfs]

    # CSV: resumen por archivo
    df = pd.DataFrame([asdict(s) for s in stats]).sort_values("file")
    df.to_csv(out_dir / "eda_summary_by_file.csv", index=False)

    # JSON: recomendación de chunking
    recs = propose_chunking(stats)
    (out_dir / "eda_recommendations.json").write_text(
        json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Mini resumen por consola
    print(f"PDFs: {len(stats)} | p90_paragraph_mean: {recs['p90_paragraph_mean']}")
    print(f"chunk_size: {recs['chunk_size']} | chunk_overlap: {recs['chunk_overlap']} (fuente: {recs['source']})")
    print(f"Archivos: {out_dir/'eda_summary_by_file.csv'} ; {out_dir/'eda_recommendations.json'}")

if __name__ == "__main__":
    main()
