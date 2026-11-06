"""Compare base and compressed dev results for the benchmark pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-banking", type=Path, required=True)
    parser.add_argument("--base-qa", type=Path, required=True)
    parser.add_argument("--comp-banking", type=Path, required=True)
    parser.add_argument("--comp-qa", type=Path, required=True)
    parser.add_argument("--compression-report", type=Path, default=None)
    args = parser.parse_args()

    b_base = load(args.base_banking)
    b_comp = load(args.comp_banking)
    q_base = load(args.base_qa)
    q_comp = load(args.comp_qa)

    print("Benchmark pilot (dev only; locked exams untouched)")
    print("=" * 72)
    if b_base and b_comp:
        print(f"Banking77 accuracy: base={b_base['accuracy']:.4f} "
              f"compressed={b_comp['accuracy']:.4f} "
              f"delta={b_comp['accuracy'] - b_base['accuracy']:+.4f} "
              f"ratio={b_comp['accuracy'] / b_base['accuracy']:.4f}")
    else:
        print("Banking77: missing result")
    if q_base and q_comp:
        print(f"SQuAD v2 EM:        base={q_base['em']:.4f} "
              f"compressed={q_comp['em']:.4f} "
              f"delta={q_comp['em'] - q_base['em']:+.4f} "
              f"ratio={q_comp['em'] / q_base['em']:.4f}")
        print(f"SQuAD v2 F1:        base={q_base['f1']:.4f} "
              f"compressed={q_comp['f1']:.4f} "
              f"delta={q_comp['f1'] - q_base['f1']:+.4f} "
              f"ratio={q_comp['f1'] / q_base['f1']:.4f}")
        print(f"SQuAD retrieval@k:  base={q_base['retrieval_hit_at_k']:.4f} "
              f"compressed={q_comp['retrieval_hit_at_k']:.4f}")
    else:
        print("SQuAD v2: missing result")

    if args.compression_report and args.compression_report.exists():
        report = json.loads(args.compression_report.read_text(encoding="utf-8"))
        print("-" * 72)
        print(f"Compression: params={report['compressed_params']} / {report['base_params']} "
              f"= {report['param_ratio']:.4f}")
        print(f"Compression: bytes={report['compressed_weight_bytes']} / {report['base_weight_bytes']} "
              f"= {report['byte_ratio']:.4f}")


if __name__ == "__main__":
    main()
