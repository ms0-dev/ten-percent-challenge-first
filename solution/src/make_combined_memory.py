"""Combine the old-intent memory with the new-intent user feedback."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banking-dir", type=Path, default=Path("solution/data/banking77"))
    args = parser.parse_args()

    old = list(read_jsonl(args.banking_dir / "old_memory.jsonl"))
    new = list(read_jsonl(args.banking_dir / "new_feedback.jsonl"))
    out = args.banking_dir / "combined_memory.jsonl"
    n = write_jsonl(out, old + new)
    print(f"Wrote {out} with {n} rows ({len(old)} old + {len(new)} feedback)")


if __name__ == "__main__":
    main()
