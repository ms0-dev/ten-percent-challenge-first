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
import os
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
        "kernels==0.17.2",
        "jinja2",
        "sentencepiece",
        "protobuf",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir("solution/src", remote_path="/src")
)


def _run(args: list[str], env: dict | None = None) -> None:
    print("RUN", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd="/src", env=env)


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
def download_qwen3_remote() -> None:
    from huggingface_hub import snapshot_download

    target = "/vol/models/qwen3-30b"
    path = snapshot_download(
        repo_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
        local_dir=target,
        max_workers=8,
    )
    print("snapshot", path, flush=True)
    total = 0
    for p in sorted(Path(target).glob("*")):
        if p.is_file():
            print(f"  {p.name} {p.stat().st_size}", flush=True)
            total += p.stat().st_size
    print("total bytes", total, flush=True)
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
    timeout=7200,
    cpu=8,
    memory=131072,
)
def base_banking_dev_remote(banking_limit: int = 300, candidate_k: int = 0, candidate_pool: int = 30) -> None:
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", "/vol/models/gpt-oss-120b",
        "--name", "base",
        "--split", "dev",
        "--data-dir", "/vol/data/banking77",
        "--out-dir", "/vol/results/base/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--candidate-k", str(candidate_k),
        "--candidate-pool", str(candidate_pool),
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ])
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A100-80GB",
    timeout=7200,
    cpu=8,
    memory=131072,
)
def base_qa_mc_dev_remote(qa_limit: int = 300) -> None:
    _run([
        sys.executable, "/src/eval_qa_mc.py",
        "--model-path", "/vol/models/gpt-oss-120b",
        "--name", "base",
        "--split", "dev",
        "--data-dir", "/vol/data/squad_v2",
        "--out-dir", "/vol/results/base/qa_mc",
        "--limit", str(qa_limit),
        "--top-k", "5",
        "--batch-size", "1",
        "--max-new-tokens", "8",
        "--max-length", "4096",
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
def build_compressed_remote(k: int = 9, usage: str = "banking") -> None:
    _run([
        sys.executable, "/src/compress_experts.py",
        "--base-dir", "/vol/models/gpt-oss-120b",
        "--out-dir", f"/vol/models/compressed-k{k}-{usage}",
        "--router-usage", f"/vol/results/router/{usage}_counts.json",
        "--k", str(k),
        "--force",
    ])
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=1200,
    cpu=4,
    memory=24576,
)
def inspect_mxfp4_roundtrip() -> None:
    import torch
    from safetensors import safe_open
    from transformers.integrations import mxfp4
    from transformers.integrations.hub_kernels import get_kernel

    index = json.loads(Path("/vol/models/gpt-oss-120b/model.safetensors.index.json").read_text(encoding="utf-8"))
    key = "model.layers.0.mlp.experts.gate_up_proj_blocks"
    scale_key = "model.layers.0.mlp.experts.gate_up_proj_scales"
    shard = index["weight_map"][key]
    with safe_open(str(Path("/vol/models/gpt-oss-120b") / shard), framework="pt", device="cpu") as f:
        blocks = f.get_tensor(key)[:2].clone()
        scales = f.get_tensor(scale_key)[:2].clone()
    print("blocks", blocks.shape, blocks.dtype, flush=True)
    print("scales", scales.shape, scales.dtype, flush=True)
    deq = mxfp4.convert_moe_packed_tensors(blocks, scales)
    print("dequantized", deq.shape, deq.dtype, flush=True)
    print("deq sample", deq[0, 0, :8].tolist(), flush=True)

    hub = get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)
    deq_gpu = deq.to("cuda")
    w, w_scale = mxfp4.quantize_to_mxfp4(deq_gpu.transpose(-1, -2), hub)
    print("requantized", w.shape, w.dtype, w_scale.shape, w_scale.dtype, flush=True)
    # Compare with the original checkpoint blocks/scales.
    print("blocks match", torch.equal(w.cpu().to(torch.uint8), blocks), flush=True)
    print("scales match", torch.equal(w_scale.cpu().to(torch.uint8), scales), flush=True)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=600, cpu=2, memory=4096)
def make_combined_memory_remote() -> None:
    _run([sys.executable, "/src/make_combined_memory.py", "--banking-dir", "/vol/data/banking77"])
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=14400,
    cpu=8,
    memory=32768,
)
def growth_eval_remote(model_tag: str = "k9-banking", old_dev_limit: int = 300) -> None:
    model_path = f"/vol/models/compressed-{model_tag}"
    out = f"/vol/results/growth/compressed-{model_tag}"
    common = [
        sys.executable, "/src/eval_classification.py",
        "--model-path", model_path,
        "--name", f"compressed-{model_tag}",
        "--meta", "/vol/data/banking77/meta.json",
    ]
    _run(common + [
        "--memory", "/vol/data/banking77/old_memory.jsonl",
        "--rows", "/vol/data/banking77/locked/new_test.jsonl",
        "--labels", "old",
        "--out-dir", f"{out}/new_before",
        "--top-k", "8",
        "--max-new-tokens", "24",
        "--max-length", "3072",
        "--final-eval",
    ])
    _run(common + [
        "--memory", "/vol/data/banking77/combined_memory.jsonl",
        "--rows", "/vol/data/banking77/locked/new_test.jsonl",
        "--labels", "all",
        "--out-dir", f"{out}/new_after",
        "--top-k", "8",
        "--max-new-tokens", "24",
        "--max-length", "3072",
        "--final-eval",
    ])
    _run(common + [
        "--memory", "/vol/data/banking77/old_memory.jsonl",
        "--rows", "/vol/data/banking77/old_dev.jsonl",
        "--labels", "old",
        "--out-dir", f"{out}/old_before",
        "--limit", str(old_dev_limit),
        "--top-k", "8",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ])
    _run(common + [
        "--memory", "/vol/data/banking77/combined_memory.jsonl",
        "--rows", "/vol/data/banking77/old_dev.jsonl",
        "--labels", "old",
        "--out-dir", f"{out}/old_after",
        "--limit", str(old_dev_limit),
        "--top-k", "8",
        "--max-new-tokens", "24",
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
def final_banking_base_remote() -> None:
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", "/vol/models/gpt-oss-120b",
        "--name", "base",
        "--split", "exam",
        "--final-eval",
        "--data-dir", "/vol/data/banking77",
        "--out-dir", "/vol/results/final/base/banking",
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ])
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=14400,
    cpu=8,
    memory=32768,
)
def final_banking_compressed_remote(model_tag: str = "k9-banking") -> None:
    _run([
        sys.executable, "/src/eval_classification.py",
        "--model-path", f"/vol/models/compressed-{model_tag}",
        "--name", f"compressed-{model_tag}",
        "--memory", "/vol/data/banking77/old_memory.jsonl",
        "--rows", "/vol/data/banking77/locked/old_test.jsonl",
        "--labels", "old",
        "--meta", "/vol/data/banking77/meta.json",
        "--out-dir", f"/vol/results/final/compressed-{model_tag}/banking",
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
        "--final-eval",
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
def collect_qwen3_router_remote(
    data_dir: str = "/vol/data/banking77_10class",
    out: str = "/vol/results/router_qwen3/banking10_counts.json",
    limit: int = 120,
) -> None:
    _run([
        sys.executable, "/src/collect_qwen3_router.py",
        "--model-path", "/vol/models/qwen3-30b",
        "--data-dir", data_dir,
        "--out", out,
        "--limit", str(limit),
        "--top-k-retrieval", "4",
        "--top-k-router", "8",
    ])
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=14400, cpu=8, memory=32768)
def compress_qwen3_remote(k: int = 7, usage: str = "banking10") -> None:
    _run([
        sys.executable, "/src/compress_qwen3.py",
        "--base-dir", "/vol/models/qwen3-30b",
        "--out-dir", f"/vol/models/compressed-qwen3-k{k}-{usage}",
        "--router-usage", f"/vol/results/router_qwen3/{usage}_counts.json",
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
def compressed_qwen3_banking_dev_remote(
    model_tag: str = "k7-banking10",
    data_dir: str = "/vol/data/banking77_10class",
    banking_limit: int = 300,
) -> None:
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", f"/vol/models/compressed-qwen3-{model_tag}",
        "--name", f"compressed-qwen3-{model_tag}",
        "--eager-experts",
        "--split", "dev",
        "--data-dir", data_dir,
        "--out-dir", f"/vol/results/qwen3/{model_tag}/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ])
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=14400, cpu=8, memory=65536)
def build_averaged_remote(k: int = 9, usage: str = "banking") -> None:
    _run([
        sys.executable, "/src/build_averaged_experts.py",
        "--base-dir", "/vol/models/gpt-oss-120b",
        "--out-dir", f"/vol/models/averaged-k{k}-{usage}-bf16",
        "--router-usage", f"/vol/results/router/{usage}_counts.json",
        "--k", str(k),
        "--force",
    ])
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="L40S",
    timeout=14400,
    cpu=8,
    memory=65536,
)
def averaged_banking_dev_remote(model_tag: str = "k9-banking", banking_limit: int = 300) -> None:
    env = {**os.environ, "USE_HUB_KERNELS": "0"}
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", f"/vol/models/averaged-{model_tag}-bf16",
        "--name", f"averaged-{model_tag}",
        "--eager-experts",
        "--split", "dev",
        "--data-dir", "/vol/data/banking77",
        "--out-dir", f"/vol/results/averaged/{model_tag}/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ], env=env)
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A100-80GB",
    timeout=10800,
    cpu=8,
    memory=131072,
)
def base_grouped_banking_dev_remote(
    base_model: str = "/vol/models/gpt-oss-120b",
    model_name: str = "base-gpt-oss",
    banking_limit: int = 300,
    data_dir: str = "/vol/data/banking77_grouped",
) -> None:
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", base_model,
        "--name", model_name,
        "--split", "dev",
        "--data-dir", data_dir,
        "--out-dir", f"/vol/results/grouped/{model_name}/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ])
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="L40S",
    timeout=14400,
    cpu=8,
    memory=65536,
)
def averaged_grouped_banking_dev_remote(model_tag: str = "k9-banking", banking_limit: int = 300) -> None:
    env = {**os.environ, "USE_HUB_KERNELS": "0"}
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", f"/vol/models/averaged-{model_tag}-bf16",
        "--name", f"averaged-{model_tag}",
        "--eager-experts",
        "--split", "dev",
        "--data-dir", "/vol/data/banking77_grouped",
        "--out-dir", f"/vol/results/grouped/averaged-{model_tag}/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ], env=env)
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="L40S",
    timeout=14400,
    cpu=8,
    memory=65536,
)
def compressed_python_grouped_banking_dev_remote(model_tag: str = "k9-banking", banking_limit: int = 300) -> None:
    env = {**os.environ, "USE_HUB_KERNELS": "0"}
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", f"/vol/models/compressed-{model_tag}",
        "--name", f"compressed-{model_tag}-python",
        "--python-experts",
        "--split", "dev",
        "--data-dir", "/vol/data/banking77_grouped",
        "--out-dir", f"/vol/results/grouped/compressed-{model_tag}/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ], env=env)
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="L40S",
    timeout=14400,
    cpu=8,
    memory=65536,
)
def averaged_qa_mc_dev_remote(model_tag: str = "k9-qa", qa_limit: int = 300) -> None:
    env = {**os.environ, "USE_HUB_KERNELS": "0"}
    _run([
        sys.executable, "/src/eval_qa_mc.py",
        "--model-path", f"/vol/models/averaged-{model_tag}-bf16",
        "--name", f"averaged-{model_tag}",
        "--eager-experts",
        "--split", "dev",
        "--data-dir", "/vol/data/squad_v2",
        "--out-dir", f"/vol/results/averaged/{model_tag}/qa_mc",
        "--limit", str(qa_limit),
        "--top-k", "5",
        "--batch-size", "1",
        "--max-new-tokens", "8",
        "--max-length", "4096",
    ], env=env)
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="L40S",
    timeout=7200,
    cpu=8,
    memory=65536,
)
def debug_down_transpose_remote(model_tag: str = "k9-banking", limit: int = 5) -> None:
    env = {
        **os.environ,
        "USE_HUB_KERNELS": "0",
        "PYTHON_EXPERTS_DOWN_TRANSPOSE": "1",
    }
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", f"/vol/models/compressed-{model_tag}",
        "--name", "debug-down-transpose",
        "--python-experts",
        "--split", "dev",
        "--data-dir", "/vol/data/banking77",
        "--out-dir", "/vol/results/debug/down_transpose",
        "--limit", str(limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ], env=env)
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="L40S",
    timeout=14400,
    cpu=8,
    memory=65536,
)
def compressed_python_banking_dev_remote(
    model_tag: str = "k9-banking",
    banking_limit: int = 300,
    candidate_k: int = 0,
    candidate_pool: int = 30,
) -> None:
    env = {**os.environ, "USE_HUB_KERNELS": "0"}
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", f"/vol/models/compressed-{model_tag}",
        "--name", f"compressed-{model_tag}-python",
        "--python-experts",
        "--split", "dev",
        "--data-dir", "/vol/data/banking77",
        "--out-dir", f"/vol/results/python/compressed-{model_tag}/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--candidate-k", str(candidate_k),
        "--candidate-pool", str(candidate_pool),
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ], env=env)
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="L40S",
    timeout=14400,
    cpu=8,
    memory=65536,
)
def compressed_python_qa_mc_dev_remote(model_tag: str = "k9-qa", qa_limit: int = 300) -> None:
    env = {**os.environ, "USE_HUB_KERNELS": "0"}
    _run([
        sys.executable, "/src/eval_qa_mc.py",
        "--model-path", f"/vol/models/compressed-{model_tag}",
        "--name", f"compressed-{model_tag}-python",
        "--python-experts",
        "--split", "dev",
        "--data-dir", "/vol/data/squad_v2",
        "--out-dir", f"/vol/results/python/compressed-{model_tag}/qa_mc",
        "--limit", str(qa_limit),
        "--top-k", "5",
        "--batch-size", "1",
        "--max-new-tokens", "8",
        "--max-length", "4096",
    ], env=env)
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=10800,
    cpu=8,
    memory=32768,
)
def compressed_banking_dev_remote(
    model_tag: str = "k9-banking",
    banking_limit: int = 300,
    candidate_k: int = 0,
    candidate_pool: int = 30,
) -> None:
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", f"/vol/models/compressed-{model_tag}",
        "--name", f"compressed-{model_tag}",
        "--split", "dev",
        "--data-dir", "/vol/data/banking77",
        "--out-dir", f"/vol/results/compressed-{model_tag}/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--candidate-k", str(candidate_k),
        "--candidate-pool", str(candidate_pool),
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
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
def compressed_qa_mc_dev_remote(model_tag: str = "k9-qa", qa_limit: int = 300) -> None:
    _run([
        sys.executable, "/src/eval_qa_mc.py",
        "--model-path", f"/vol/models/compressed-{model_tag}",
        "--name", f"compressed-{model_tag}",
        "--split", "dev",
        "--data-dir", "/vol/data/squad_v2",
        "--out-dir", f"/vol/results/compressed-{model_tag}/qa_mc",
        "--limit", str(qa_limit),
        "--top-k", "5",
        "--batch-size", "1",
        "--max-new-tokens", "8",
        "--max-length", "4096",
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
def compressed_dev_eval_remote(model_tag: str = "k9-banking", banking_limit: int = 300, qa_limit: int = 300) -> None:
    model_path = f"/vol/models/compressed-{model_tag}"
    name = f"compressed-{model_tag}"
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", model_path,
        "--name", name,
        "--split", "dev",
        "--data-dir", "/vol/data/banking77",
        "--out-dir", f"/vol/results/compressed-{model_tag}/banking",
        "--limit", str(banking_limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ])
    _run([
        sys.executable, "/src/eval_qa.py",
        "--model-path", model_path,
        "--name", name,
        "--split", "dev",
        "--data-dir", "/vol/data/squad_v2",
        "--out-dir", f"/vol/results/compressed-{model_tag}/qa",
        "--limit", str(qa_limit),
        "--top-k", "5",
        "--batch-size", "1",
        "--max-new-tokens", "32",
        "--max-length", "3072",
    ])
    volume.commit()
