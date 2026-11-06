"""Modal jobs for data prep, base evaluation, compression, and compressed eval.

Run from the repo root, e.g.:

    modal run solution/modal_app.py::prepare_banking77_remote
    modal run solution/modal_app.py::inspect_techqa_remote
    modal run solution/modal_app.py::download_base_remote
    modal run solution/modal_app.py::base_dev_eval_remote
    modal run solution/modal_app.py::collect_router_remote
    modal run solution/modal_app.py::build_compressed_remote --k 9
    modal run solution/modal_app.py::compressed_dev_eval_remote --k 9
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "ten-percent-challenge"
VOLUME_NAME = "ten-percent-challenge"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==5.22.1",
        "accelerate==1.15.0",
        "datasets==5.1.0",
        "rank_bm25==0.2.2",
        "huggingface_hub[hf_transfer]==2.4.0",
        "safetensors==0.8.0",
        "kernels",
        "jinja2",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("solution/src", remote_path="/src")
)


def _run(args: list[str]) -> None:
    print("RUN", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd="/src")


@app.function(image=image, volumes={"/vol": volume}, timeout=1800, cpu=4, memory=8192)
def prepare_banking77_remote(force: bool = True) -> None:
    args = [sys.executable, "/src/prepare_banking77.py", "--out", "/vol/data/banking77"]
    if force:
        args.append("--force")
    _run(args)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=1800, cpu=4, memory=8192)
def prepare_squad_remote(force: bool = True) -> None:
    args = [sys.executable, "/src/prepare_squad.py", "--out", "/vol/data/squad_v2"]
    if force:
        args.append("--force")
    _run(args)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=3600, cpu=8, memory=16384)
def inspect_techqa_remote() -> None:
    import os
    import tarfile

    from huggingface_hub import hf_hub_download

    raw_dir = Path("/vol/raw/techqa")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="PrimeQA/TechQA",
        filename="TechQA.tar.gz",
        repo_type="dataset",
        local_dir=str(raw_dir),
    )
    print("downloaded", path, os.path.getsize(path), flush=True)

    with tarfile.open(path, "r:gz") as tar:
        names = tar.getnames()
        print("entries", len(names), flush=True)
        for name in names[:100]:
            print(" ", name, flush=True)
        extract_dir = raw_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            tar.extractall(extract_dir, filter="data")
        except TypeError:
            tar.extractall(extract_dir)

    print("extracted tree:", flush=True)
    for p in sorted(extract_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(extract_dir)} {p.stat().st_size}", flush=True)

    print("sample JSON heads:", flush=True)
    for p in sorted(extract_dir.rglob("*.json"))[:6]:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            head = f.read(2500)
        print(f"--- {p.name} ---", flush=True)
        print(head, flush=True)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=10800, cpu=8, memory=16384)
def download_base_remote() -> None:
    from huggingface_hub import snapshot_download

    target = "/vol/models/gpt-oss-120b"
    path = snapshot_download(
        repo_id="openai/gpt-oss-120b",
        local_dir=target,
        max_workers=8,
        ignore_patterns=["original/*", "metal/*"],
    )
    print("snapshot", path, flush=True)
    total = 0
    for p in sorted(Path(target).glob("*")):
        if p.is_file():
            print(f"  {p.name} {p.stat().st_size}", flush=True)
            total += p.stat().st_size
    print("total bytes", total, flush=True)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=3600, cpu=4, memory=8192)
def inspect_base_remote() -> None:
    base = Path("/vol/models/gpt-oss-120b")
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    print(json.dumps({k: config.get(k) for k in (
        "num_local_experts",
        "num_experts_per_tok",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
    )}, indent=2), flush=True)
    total = 0
    for p in sorted(base.glob("*")):
        if p.is_file():
            print(f"  {p.name} {p.stat().st_size}", flush=True)
            total += p.stat().st_size
    print("total bytes", total, flush=True)


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A100-80GB",
    timeout=10800,
    cpu=8,
    memory=131072,
)
def base_dev_eval_remote(banking_limit: int = 300, qa_limit: int = 300) -> None:
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", "/vol/models/gpt-oss-120b",
        "--name", "base",
        "--split", "dev",
        "--data-dir", "/vol/data/banking77",
        "--out-dir", "/vol/results/base/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ])
    _run([
        sys.executable, "/src/eval_qa.py",
        "--model-path", "/vol/models/gpt-oss-120b",
        "--name", "base",
        "--split", "dev",
        "--data-dir", "/vol/data/squad_v2",
        "--out-dir", "/vol/results/base/qa",
        "--limit", str(qa_limit),
        "--top-k", "5",
        "--batch-size", "1",
        "--max-new-tokens", "32",
        "--max-length", "3072",
    ])
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A100-80GB",
    timeout=10800,
    cpu=8,
    memory=131072,
)
def collect_router_remote(banking_limit: int = 120, qa_limit: int = 120) -> None:
    _run([
        sys.executable, "/src/collect_router.py",
        "--model-path", "/vol/models/gpt-oss-120b",
        "--banking-dir", "/vol/data/banking77",
        "--qa-dir", "/vol/data/squad_v2",
        "--out-dir", "/vol/results/router",
        "--banking-limit", str(banking_limit),
        "--qa-limit", str(qa_limit),
        "--top-k-retrieval", "4",
    ])
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=7200, cpu=8, memory=32768)
def build_compressed_remote(k: int = 9, usage: str = "combined") -> None:
    _run([
        sys.executable, "/src/compress_experts.py",
        "--base-dir", "/vol/models/gpt-oss-120b",
        "--out-dir", f"/vol/models/compressed-k{k}",
        "--router-usage", f"/vol/results/router/{usage}_counts.json",
        "--k", str(k),
        "--force",
    ])
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=10800,
    cpu=8,
    memory=32768,
)
def compressed_dev_eval_remote(k: int = 9, banking_limit: int = 300, qa_limit: int = 300) -> None:
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", f"/vol/models/compressed-k{k}",
        "--name", f"compressed-k{k}",
        "--split", "dev",
        "--data-dir", "/vol/data/banking77",
        "--out-dir", f"/vol/results/compressed-k{k}/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ])
    _run([
        sys.executable, "/src/eval_qa.py",
        "--model-path", f"/vol/models/compressed-k{k}",
        "--name", f"compressed-k{k}",
        "--split", "dev",
        "--data-dir", "/vol/data/squad_v2",
        "--out-dir", f"/vol/results/compressed-k{k}/qa",
        "--limit", str(qa_limit),
        "--top-k", "5",
        "--batch-size", "1",
        "--max-new-tokens", "32",
        "--max-length", "3072",
    ])
    volume.commit()
