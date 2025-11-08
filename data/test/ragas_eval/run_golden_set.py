from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from cli import parse_args
from golden_io import load_golden_set, ensure_output_path
from backend_runner import run_entry
from ragas_llm_gemini import build_ragas_llm
from ragas_metrics import run_ragas_evaluation
from baseline import load_baseline_metrics, build_comparison

def main() -> None:
    args = parse_args()
    golden_path = Path(args.golden_set)
    entries = load_golden_set(golden_path)

    if args.output:
        output_path = ensure_output_path(Path(args.output))
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = ensure_output_path(Path("results") / f"golden_{ts}.jsonl")

    print(f"Golden set: {golden_path} ({len(entries)} escenarios)")
    print(f"Endpoint:   {args.base_url}")
    print(f"Resultados: {output_path}\n")

    successes = 0
    failures = 0
    all_results: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as fh:
        for idx, entry in enumerate(entries, start=1):
            result = run_entry(entry, args.base_url, args.timeout)
            all_results.append(result)
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

            if result["error"]:
                failures += 1
                print(f"[{idx}/{len(entries)}] {entry.get('id')} ✗ {result['error']}")
            else:
                successes += 1
                sources_count = len(result.get("sources") or [])
                print(
                    f"[{idx}/{len(entries)}] {entry.get('id')} ✓ {result['elapsed_ms']} ms, "
                    f"fuentes={sources_count}"
                )

    api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Debes proporcionar GEMINI_API_KEY o --gemini-api-key")

    ragas_llm = build_ragas_llm(args.ragas_model, api_key, args.debug_llm)

    metrics_summary = run_ragas_evaluation(entries, all_results, ragas_llm)
    summary_path = output_path.with_suffix(".metrics.json")
    summary_payload: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "base_url": args.base_url,
        "results_file": str(output_path),
        "summary": metrics_summary,
    }

    if args.baseline_metrics:
        baseline_path = Path(args.baseline_metrics)
        baseline_data = load_baseline_metrics(baseline_path)
        if baseline_data:
            comparison = build_comparison(metrics_summary, baseline_data)
            summary_payload["comparison"] = comparison

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)

    print("\nResumen")
    print("-------")
    print(f"Total:     {len(entries)}")
    print(f"Éxitos:    {successes}")
    print(f"Fallidos:  {failures}")
    print(f"Archivo:   {output_path}")

    if metrics_summary.get("metrics"):
        print("\nMétricas RAGAS:")
        for name, value in metrics_summary["metrics"].items():
            print(f"  - {name}: {round(value, 4)}")

    if metrics_summary.get("errors"):
        print("\nErrores de métricas:")
        for name, err in metrics_summary["errors"].items():
            print(f"  - {name}: {err}")

    print(f"Resumen de métricas: {summary_path}")


if __name__ == "__main__":
    main()