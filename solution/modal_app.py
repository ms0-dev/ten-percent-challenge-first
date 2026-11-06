"""Modal jobs for the 10% challenge.

Final pipeline (all other functions are exploratory and are not used by the
final proof):

    prepare_banking77_remote
    download_qwen3_remote
    collect_qwen3_router_remote --usage banking10
    compress_qwen3_remote --k 96 --usage banking10
    clone_llamacpp_remote
    convert_hf_to_gguf_remote
    make_calibration_text_remote
    llamacpp_imatrix_remote
    download_gguf_remote --filename Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf
    dump_gguf_types_remote
    quantize_gguf_remote
    llamacpp_banking_remote
    final_10_base_remote
    growth_prototype_remote
    final_metadata_remote
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

llamacpp_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu24.04", add_python="3.11")
    .apt_install("libgomp1", "curl", "ca-certificates")
    .pip_install(
        "huggingface_hub==2.4.0",
        "transformers==5.22.1",
        "tokenizers",
        "rank_bm25==0.2.2",
        "jinja2",
        "sentencepiece",
        "gguf",
    )
    .add_local_dir("solution/src", remote_path="/src")
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
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
        "gguf",
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


@app.function(image=image, volumes={"/vol": volume}, timeout=3600, cpu=4, memory=8192)
def clone_llamacpp_remote() -> None:
    if not Path("/vol/llama.cpp/convert_hf_to_gguf.py").exists():
        _run([
            "git", "clone", "--depth", "1", "--branch", "b12110",
            "https://github.com/ggml-org/llama.cpp", "/vol/llama.cpp",
        ])
    print("llama.cpp ready", flush=True)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=14400, cpu=16, memory=65536)
def convert_hf_to_gguf_remote(
    src: str,
    dst: str,
    outtype: str = "bf16",
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/vol/llama.cpp/gguf-py:/vol/llama.cpp"
    _run([
        sys.executable, "/vol/llama.cpp/convert_hf_to_gguf.py",
        "--outtype", outtype,
        "--outfile", dst,
        src,
    ], env=env)
    print("WROTE", dst, Path(dst).stat().st_size, flush=True)
    volume.commit()


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
def make_combined_memory_remote(banking_dir: str = "/vol/data/banking77") -> None:
    _run([sys.executable, "/src/make_combined_memory.py", "--banking-dir", banking_dir])
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


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=14400,
    cpu=8,
    memory=32768,
)
def growth_qwen3_remote(
    model_tag: str = "k7-banking10",
    data_dir: str = "/vol/data/banking77_10class",
    old_dev_limit: int = 300,
) -> None:
    model_path = f"/vol/models/compressed-qwen3-{model_tag}"
    out = f"/vol/results/qwen3/{model_tag}/growth"
    common = [
        sys.executable, "/src/eval_classification.py",
        "--model-path", model_path,
        "--name", f"compressed-qwen3-{model_tag}",
        "--eager-experts",
        "--meta", f"{data_dir}/meta.json",
    ]
    _run(common + [
        "--memory", f"{data_dir}/old_memory.jsonl",
        "--rows", f"{data_dir}/locked/new_test.jsonl",
        "--labels", "old",
        "--out-dir", f"{out}/new_before",
        "--top-k", "8",
        "--max-new-tokens", "24",
        "--max-length", "3072",
        "--final-eval",
    ])
    _run(common + [
        "--memory", f"{data_dir}/combined_memory.jsonl",
        "--rows", f"{data_dir}/locked/new_test.jsonl",
        "--labels", "all",
        "--out-dir", f"{out}/new_after",
        "--top-k", "8",
        "--max-new-tokens", "24",
        "--max-length", "3072",
        "--final-eval",
    ])
    _run(common + [
        "--memory", f"{data_dir}/old_memory.jsonl",
        "--rows", f"{data_dir}/old_dev.jsonl",
        "--labels", "old",
        "--out-dir", f"{out}/old_before",
        "--limit", str(old_dev_limit),
        "--top-k", "8",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ])
    _run(common + [
        "--memory", f"{data_dir}/combined_memory.jsonl",
        "--rows", f"{data_dir}/old_dev.jsonl",
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
def final_10_base_remote(limit: int = 1000) -> None:
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", "/vol/models/qwen3-30b",
        "--name", "base-qwen3-10-final",
        "--split", "exam",
        "--final-eval",
        "--data-dir", "/vol/data/banking77_10class",
        "--out-dir", "/vol/results/final/base-qwen3-10/banking",
        "--limit", str(limit),
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
    timeout=10800,
    cpu=8,
    memory=32768,
)
def final_10_compressed_remote(model_tag: str = "k7-banking10", limit: int = 1000) -> None:
    _run([
        sys.executable, "/src/eval_banking.py",
        "--model-path", f"/vol/models/compressed-qwen3-{model_tag}",
        "--name", f"compressed-qwen3-{model_tag}",
        "--eager-experts",
        "--split", "exam",
        "--final-eval",
        "--data-dir", "/vol/data/banking77_10class",
        "--out-dir", f"/vol/results/final/compressed-qwen3-{model_tag}/banking",
        "--limit", str(limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ])
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=7200, cpu=8, memory=16384)
def download_llamacpp_remote() -> None:
    import tarfile
    import urllib.request

    out_dir = Path("/vol/bin")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "https://github.com/ggml-org/llama.cpp/releases/download/b12110"
    for name in (
        "llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "cudart-llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz",
    ):
        target = out_dir / name
        if not target.exists():
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(f"{base}/{name}", target)
        print("extracting", name, flush=True)
        with tarfile.open(target) as tar:
            tar.extractall(out_dir)
    for path in sorted(out_dir.iterdir()):
        print(path, flush=True)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=7200, cpu=8, memory=16384)
def download_gguf_remote(
    repo_id: str = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
    filename: str = "Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf",
) -> None:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="/vol/models/gguf",
    )
    print("downloaded", path, Path(path).stat().st_size, flush=True)
    volume.commit()


@app.function(
    image=llamacpp_image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=14400,
    cpu=8,
    memory=32768,
)
def growth_prototype_remote(
    gguf: str = "/vol/models/gguf/qwen3-k96-udmix.gguf",
    old_dev_limit: int = 300,
) -> None:
    import hashlib

    digest = hashlib.sha256()
    with open(gguf, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    model_sha = digest.hexdigest()
    print("MODEL SHA256", model_sha, flush=True)

    server_bin, env = _llamacpp_server()
    log_path = Path("/tmp/llama-growth.log")
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            str(server_bin), "-m", gguf,
            "--host", "127.0.0.1", "--port", "8080",
            "-ngl", "99", "-c", "2048", "-t", "8",
            "-b", "2048", "-ub", "2048",
            "--embeddings", "--pooling", "last",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_server(8080, proc, log_path)
        _run([
            sys.executable, "/src/growth_prototype.py",
            "--port", "8080",
            "--data-dir", "/vol/data/banking77_10class",
            "--out-dir", "/vol/results/growth/prototype",
            "--batch-size", "32",
            "--model-path", gguf,
        ])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        print("SERVER LOG TAIL", log_path.read_text(errors="replace")[-2000:], flush=True)
    volume.commit()


@app.function(
    image=llamacpp_image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=3600,
    cpu=8,
    memory=32768,
)
def llamacpp_embedding_remote(
    gguf: str = "/vol/models/gguf/qwen3-k96-udmix.gguf",
    text: str = "My card was charged twice for the same purchase.",
) -> None:
    import time
    import urllib.request

    server_bin, env = _llamacpp_server()
    log_path = Path("/tmp/llama-embedding.log")
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            str(server_bin), "-m", gguf,
            "--host", "127.0.0.1", "--port", "8080",
            "-ngl", "99", "-c", "2048", "-t", "8",
            "--embeddings", "--pooling", "last",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_server(8080, proc, log_path)
        payload = json.dumps({"input": text, "model": "local"}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8080/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
        vector = result["data"][0]["embedding"]
        print("DIM", len(vector), flush=True)
        print("FIRST", vector[:8], flush=True)
        print("NORM", sum(x * x for x in vector) ** 0.5, flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
        log.close()
        print("SERVER LOG TAIL", log_path.read_text(errors="replace")[-2000:], flush=True)


@app.function(
    image=llamacpp_image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=3600,
    cpu=8,
    memory=32768,
)
def llamacpp_smoke_remote(
    gguf: str = "/vol/models/gguf/Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf",
    prompt: str = "Reply with exactly one word: hello",
) -> None:
    import os
    import time

    import urllib.request

    bin_dir = Path("/vol/bin")
    server_bin = next(bin_dir.rglob("llama-server"))
    lib_dirs = sorted({str(p.parent) for p in bin_dir.rglob("*.so*")})
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
    print("server_bin", server_bin, flush=True)
    print("lib_dirs", lib_dirs, flush=True)
    print(subprocess.run(["ldd", str(server_bin)], capture_output=True, text=True, env=env).stdout, flush=True)
    log_path = Path("/tmp/llama-server.log")
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            str(server_bin), "-m", gguf,
            "--host", "127.0.0.1", "--port", "8080",
            "-ngl", "99", "-c", "4096", "-t", "8", "--jinja",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        ready = False
        for _ in range(240):
            if proc.poll() is not None:
                break
            try:
                urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
                ready = True
                break
            except Exception:
                time.sleep(2)
        if not ready:
            log.flush()
            print("SERVER LOG", log_path.read_text(errors="replace")[-6000:], flush=True)
            raise SystemExit("server did not start")
        payload = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 24,
        }).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8080/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.time()
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
        print("ELAPSED", round(time.time() - started, 2), flush=True)
        print(json.dumps(result, indent=2)[:4000], flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def _llamacpp_server():
    bin_dir = Path("/vol/bin")
    server_bin = next(bin_dir.rglob("llama-server"))
    lib_dirs = sorted({str(p.parent) for p in bin_dir.rglob("*.so*")})
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
    return server_bin, env


def _wait_for_server(port: int, proc: subprocess.Popen, log_path: Path, attempts: int = 300) -> None:
    import time
    import urllib.request

    for _ in range(attempts):
        if proc.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(2)
    print("SERVER LOG", log_path.read_text(errors="replace")[-6000:], flush=True)
    raise SystemExit("llama.cpp server did not start")


@app.function(image=image, volumes={"/vol": volume}, timeout=1800, cpu=8, memory=16384)
def make_calibration_text_remote(
    data_dir: str = "/vol/data/banking77_10class",
    out: str = "/vol/data/banking10_calibration.txt",
    limit: int = 250,
) -> None:
    _run([
        sys.executable, "/src/make_calibration_text.py",
        "--data-dir", data_dir,
        "--tokenizer-path", "/vol/models/qwen3-30b",
        "--out", out,
        "--limit", str(limit),
    ])
    volume.commit()


@app.function(
    image=llamacpp_image,
    volumes={"/vol": volume},
    gpu="A100-80GB",
    timeout=14400,
    cpu=8,
    memory=65536,
)
def llamacpp_imatrix_remote(
    gguf: str,
    corpus: str,
    out: str,
    n_ctx: int = 2048,
) -> None:
    bin_dir = Path("/vol/bin")
    binary = next(bin_dir.rglob("llama-imatrix"))
    lib_dirs = sorted({str(p.parent) for p in bin_dir.rglob("*.so*")})
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
    _run([
        str(binary), "-m", gguf, "-f", corpus, "-o", out,
        "-ngl", "99", "-c", str(n_ctx), "-b", "512",
    ], env=env)
    print("WROTE", out, Path(out).stat().st_size, flush=True)
    volume.commit()


@app.function(image=llamacpp_image, volumes={"/vol": volume}, timeout=1800, cpu=4, memory=16384)
def shell_remote(command: str) -> None:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print("STDOUT", result.stdout[-8000:], flush=True)
    print("STDERR", result.stderr[-8000:], flush=True)
    print("RETURNCODE", result.returncode, flush=True)


@app.function(image=llamacpp_image, volumes={"/vol": volume}, timeout=14400, cpu=16, memory=65536)
def quantize_gguf_remote(
    src: str,
    dst: str,
    qtype: str = "TQ1_0",
    imatrix: str = "",
    allow_requantize: bool = False,
    pure: bool = False,
    tensor_type_file: str = "",
) -> None:
    bin_dir = Path("/vol/bin")
    quant_bin = next(bin_dir.rglob("llama-quantize"))
    lib_dirs = sorted({str(p.parent) for p in bin_dir.rglob("*.so*")})
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [str(quant_bin)]
    if imatrix:
        command += ["--imatrix", imatrix]
    if allow_requantize:
        command += ["--allow-requantize"]
    if pure:
        command += ["--pure"]
    if tensor_type_file:
        command += ["--tensor-type-file", tensor_type_file]
    command += [src, dst, qtype]
    _run(command, env=env)
    print("WROTE", dst, Path(dst).stat().st_size, flush=True)
    volume.commit()


@app.function(image=llamacpp_image, volumes={"/vol": volume}, timeout=1800, cpu=4, memory=16384)
def dump_gguf_types_remote(
    path: str = "/vol/models/gguf/Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf",
    out: str = "/vol/models/gguf/ud_tq1_0_types.txt",
) -> None:
    from gguf import GGUFReader

    reader = GGUFReader(path)
    lines = [f"{tensor.name}={tensor.tensor_type.name.lower()}" for tensor in reader.tensors]
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WROTE", out, len(lines), flush=True)
    print("\n".join(lines[:12]), flush=True)
    volume.commit()


@app.function(image=llamacpp_image, volumes={"/vol": volume}, timeout=1800, cpu=4, memory=16384)
def inspect_gguf_remote(
    path: str = "/vol/models/gguf/Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf",
) -> None:
    from collections import Counter

    from gguf import GGUFReader

    reader = GGUFReader(path)
    sizes: Counter = Counter()
    counts: Counter = Counter()
    for tensor in reader.tensors:
        sizes[str(tensor.tensor_type)] += int(tensor.n_bytes)
        counts[str(tensor.tensor_type)] += 1
    total = sum(sizes.values())
    print("FILE", Path(path).stat().st_size, "TENSOR_BYTES", total, flush=True)
    for kind, size in sizes.most_common():
        print(f"TYPE {kind}: {size} bytes ({100 * size / total:.2f}%), tensors={counts[kind]}", flush=True)
    for tensor in sorted(reader.tensors, key=lambda t: int(t.n_bytes), reverse=True)[:40]:
        print(f"TENSOR {tensor.name} {tensor.tensor_type} {int(tensor.n_bytes)}", flush=True)
    volume.commit()


@app.function(
    image=llamacpp_image,
    volumes={"/vol": volume},
    gpu="A100-80GB",
    timeout=14400,
    cpu=8,
    memory=65536,
)
def llamacpp_banking_a100_remote(
    gguf: str,
    name: str,
    data_dir: str = "/vol/data/banking77_10class",
    limit: int = 300,
    split: str = "dev",
    final_eval: bool = False,
    top_k: int = 8,
    max_new_tokens: int = 24,
    n_ctx: int = 8192,
) -> None:
    server_bin, env = _llamacpp_server()
    log_path = Path(f"/tmp/llama-server-{name}.log")
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            str(server_bin), "-m", gguf,
            "--host", "127.0.0.1", "--port", "8080",
            "-ngl", "99", "-c", str(n_ctx), "-t", "8",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_server(8080, proc, log_path)
        command = [
            sys.executable, "/src/eval_banking_llamacpp.py",
            "--port", "8080",
            "--tokenizer-path", "/vol/models/qwen3-30b",
            "--name", name,
            "--model-path", gguf,
            "--split", split,
            "--data-dir", data_dir,
            "--out-dir", f"/vol/results/llamacpp/{name}/banking",
            "--limit", str(limit),
            "--top-k", str(top_k),
            "--max-new-tokens", str(max_new_tokens),
        ]
        if final_eval:
            command.append("--final-eval")
        _run(command)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        print("SERVER LOG TAIL", log_path.read_text(errors="replace")[-3000:], flush=True)
    volume.commit()


@app.function(
    image=llamacpp_image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=14400,
    cpu=8,
    memory=32768,
)
def llamacpp_banking_remote(
    gguf: str,
    name: str,
    data_dir: str = "/vol/data/banking77_10class",
    limit: int = 300,
    split: str = "dev",
    final_eval: bool = False,
    top_k: int = 8,
    max_new_tokens: int = 24,
    n_ctx: int = 8192,
) -> None:
    server_bin, env = _llamacpp_server()
    log_path = Path(f"/tmp/llama-server-{name}.log")
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            str(server_bin), "-m", gguf,
            "--host", "127.0.0.1", "--port", "8080",
            "-ngl", "99", "-c", str(n_ctx), "-t", "8",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_server(8080, proc, log_path)
        command = [
            sys.executable, "/src/eval_banking_llamacpp.py",
            "--port", "8080",
            "--tokenizer-path", "/vol/models/qwen3-30b",
            "--name", name,
            "--model-path", gguf,
            "--split", split,
            "--data-dir", data_dir,
            "--out-dir", f"/vol/results/llamacpp/{name}/banking",
            "--limit", str(limit),
            "--top-k", str(top_k),
            "--max-new-tokens", str(max_new_tokens),
        ]
        if final_eval:
            command.append("--final-eval")
        _run(command)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        print("SERVER LOG TAIL", log_path.read_text(errors="replace")[-3000:], flush=True)
    volume.commit()


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=14400,
    cpu=8,
    memory=32768,
)
def model_banking_dev_remote(
    model_path: str,
    name: str,
    data_dir: str = "/vol/data/banking77_10class",
    limit: int = 300,
    eager: bool = False,
) -> None:
    tag = name.replace("/", "_")
    command = [
        sys.executable, "/src/eval_banking.py",
        "--model-path", model_path,
        "--name", name,
        "--split", "dev",
        "--data-dir", data_dir,
        "--out-dir", f"/vol/results/custom/{tag}/banking",
        "--limit", str(limit),
        "--top-k", "8",
        "--batch-size", "1",
        "--max-new-tokens", "24",
        "--max-length", "3072",
    ]
    if eager:
        command.append("--eager-experts")
    _run(command)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=14400, cpu=16, memory=65536)
def merge_qwen3_remote(
    k: int = 7,
    usage: str = "banking10",
    variant: str = "weight",
    iterations: int = 12,
) -> None:
    tag = f"merged-qwen3-k{k}-{variant}"
    _run([
        sys.executable,
        "/src/merge_qwen3.py",
        "--base-dir",
        "/vol/models/qwen3-30b",
        "--out-dir",
        f"/vol/models/{tag}",
        "--router-usage",
        f"/vol/results/router_qwen3/{usage}_counts.json",
        "--k",
        str(k),
        "--variant",
        variant,
        "--iterations",
        str(iterations),
        "--force",
    ])
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=3600, cpu=8, memory=131072)
def inspect_qwen3_load_remote(
    model_path: str = "/vol/models/compressed-qwen3-k7-banking10",
) -> None:
    from transformers import AutoModelForCausalLM

    model, info = AutoModelForCausalLM.from_pretrained(
        model_path,
        output_loading_info=True,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print("CONFIG", model.config.num_experts, model.config.num_experts_per_tok, flush=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        value = info.get(key)
        print(key, len(value) if value else 0, (value[:10] if value else []), flush=True)
    for name, module in model.named_modules():
        if name.endswith(".mlp.experts"):
            print("experts", name, module.__class__.__name__, flush=True)
            for pname, p in module.named_parameters():
                print("  ", pname, tuple(p.shape), p.dtype, flush=True)
            break
    for name, module in model.named_modules():
        if name.endswith(".mlp.gate"):
            print("gate", name, module.__class__.__name__, flush=True)
            for pname, p in module.named_parameters():
                print("  ", pname, tuple(p.shape), p.dtype, flush=True)
            break
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=3600, cpu=8, memory=65536)
def compare_qwen3_tensors_remote(
    model_tag: str = "k7-banking10",
) -> None:
    import torch
    from safetensors import safe_open

    base_dir = Path("/vol/models/qwen3-30b")
    comp_dir = Path(f"/vol/models/compressed-qwen3-{model_tag}")
    index = json.loads((base_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    report = json.loads((comp_dir / "compression_report.json").read_text(encoding="utf-8"))
    keep = report["keep"]
    shard_cache: dict[str, object] = {}

    def base_tensor(key: str) -> torch.Tensor:
        shard = index[key]
        handle = shard_cache.get(shard)
        if handle is None:
            handle = safe_open(str(base_dir / shard), framework="pt", device="cpu")
            shard_cache[shard] = handle
        return handle.get_tensor(key)

    with safe_open(str(comp_dir / "model.safetensors"), framework="pt", device="cpu") as cf:
        for key in (
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ):
            a = base_tensor(key)
            b = cf.get_tensor(key)
            print(key, "shapes", tuple(a.shape), tuple(b.shape), "equal", torch.equal(a, b), "maxdiff", float((a - b).abs().max()), flush=True)
        gate_base = base_tensor("model.layers.0.mlp.gate.weight")
        gate_comp = cf.get_tensor("model.layers.0.mlp.gate.weight")
        expected_gate = gate_base[keep[0]]
        print("gate slice equal", torch.equal(expected_gate, gate_comp), "maxdiff", float((expected_gate - gate_comp).abs().max()), flush=True)
        gu = cf.get_tensor("model.layers.0.mlp.experts.gate_up_proj")
        dn = cf.get_tensor("model.layers.0.mlp.experts.down_proj")
        print("compressed expert tensors", tuple(gu.shape), tuple(dn.shape), flush=True)
        for j, e in enumerate(keep[0][:4]):
            g = base_tensor(f"model.layers.0.mlp.experts.{e}.gate_proj.weight")
            u = base_tensor(f"model.layers.0.mlp.experts.{e}.up_proj.weight")
            d = base_tensor(f"model.layers.0.mlp.experts.{e}.down_proj.weight")
            exp_gu = torch.cat([g, u], dim=0)
            print(
                f"expert slot {j} base expert {e}",
                "gate_up equal", bool(torch.equal(exp_gu, gu[j])),
                "down equal", bool(torch.equal(d, dn[j])),
                "gu maxdiff", float((exp_gu - gu[j]).abs().max()),
                "down maxdiff", float((d - dn[j]).abs().max()),
                flush=True,
            )


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A10",
    timeout=3600,
    cpu=8,
    memory=32768,
)
def debug_qwen3_generate_remote(
    model_path: str = "/vol/models/compressed-qwen3-k7-banking10",
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        torch_dtype="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    print("CONFIG num_experts", model.config.num_experts, "top_k", model.config.num_experts_per_tok, flush=True)
    print("EXPERTS IMPL", getattr(model.config, "_experts_implementation", None), flush=True)
    print("EXPERT MODULE", type(model.model.layers[0].mlp.experts).__name__, flush=True)
    print("EOS", tokenizer.eos_token_id, tokenizer.eos_token, "PAD", tokenizer.pad_token_id, flush=True)
    messages = [{"role": "user", "content": "Reply with exactly one word: hello"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("PROMPT", repr(text), flush=True)
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **enc,
            max_new_tokens=8,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    ids = generated.sequences[0].tolist()
    print("IDS", ids, flush=True)
    print("DECODED", repr(tokenizer.decode(generated.sequences[0], skip_special_tokens=False)), flush=True)
    if generated.scores:
        first = generated.scores[0][0]
        top = torch.topk(first, 8)
        print(
            "FIRST TOP8",
            [(int(i), tokenizer.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values, top.indices)],
            flush=True,
        )
    with torch.inference_mode():
        out = model(**enc)
        logits = out.logits[0, -1]
        top = torch.topk(logits, 8)
        print(
            "FORWARD LAST TOP8",
            [(int(i), tokenizer.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values, top.indices)],
            flush=True,
        )
        print("LOGITS finite", bool(torch.isfinite(logits).all()), "norm", float(logits.norm()), flush=True)


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A100-80GB",
    timeout=7200,
    cpu=8,
    memory=131072,
)
def debug_qwen3_inplace_prune_remote(
    model_path: str = "/vol/models/qwen3-30b",
    usage: str = "/vol/results/router_qwen3/banking10_counts.json",
    k: int = 7,
) -> None:
    import copy

    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts, Qwen3MoeTopKRouter

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        torch_dtype="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    print("BASE CONFIG", model.config.num_experts, model.config.num_experts_per_tok, flush=True)

    def generate_once(tag: str) -> None:
        messages = [{"role": "user", "content": "Reply with exactly one word: hello"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **enc,
                max_new_tokens=8,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        new = generated.sequences[0, enc["input_ids"].shape[1] :]
        print(tag, "IDS", new.tolist(), "TEXT", repr(tokenizer.decode(new, skip_special_tokens=False)), flush=True)

    # Verify runtime expert order vs checkpoint source order.
    base_dir = Path("/vol/models/qwen3-30b")
    index = json.loads((base_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    shard_cache: dict[str, object] = {}

    def base_tensor(key: str) -> torch.Tensor:
        shard = index[key]
        handle = shard_cache.get(shard)
        if handle is None:
            handle = safe_open(str(base_dir / shard), framework="pt", device="cpu")
            shard_cache[shard] = handle
        return handle.get_tensor(key)

    rt = model.model.layers[0].mlp.experts.gate_up_proj.detach().float().cpu()
    for j in (0, 1, 2, 10, 127):
        g = base_tensor(f"model.layers.0.mlp.experts.{j}.gate_proj.weight")
        u = base_tensor(f"model.layers.0.mlp.experts.{j}.up_proj.weight")
        expected = torch.cat([g, u], dim=0).float()
        print(f"runtime slot {j} vs source expert {j}: equal", torch.equal(rt[j], expected), flush=True)

    generate_once("BASE")

    counts = json.loads(Path(usage).read_text(encoding="utf-8"))["counts"]
    keep = [sorted(range(model.config.num_experts), key=lambda e: counts[layer][e], reverse=True)[:k] for layer in range(model.config.num_hidden_layers)]
    cfg7 = copy.deepcopy(model.config)
    cfg7.num_experts = k
    cfg7.num_experts_per_tok = k
    for layer_idx, layer in enumerate(model.model.layers):
        old_e = layer.mlp.experts
        old_g = layer.mlp.gate
        idx = torch.tensor(keep[layer_idx], device=old_e.gate_up_proj.device)
        new_e = Qwen3MoeExperts(cfg7).to(device=old_e.gate_up_proj.device, dtype=old_e.gate_up_proj.dtype)
        new_g = Qwen3MoeTopKRouter(cfg7).to(device=old_g.weight.device, dtype=old_g.weight.dtype)
        with torch.no_grad():
            new_e.gate_up_proj.copy_(old_e.gate_up_proj.index_select(0, idx))
            new_e.down_proj.copy_(old_e.down_proj.index_select(0, idx))
            new_g.weight.copy_(old_g.weight.index_select(0, idx))
        layer.mlp.experts = new_e
        layer.mlp.gate = new_g
        del old_e, old_g
        if layer_idx % 8 == 0:
            torch.cuda.empty_cache()
            print("pruned layer", layer_idx, flush=True)
    model.config.num_experts = k
    model.config.num_experts_per_tok = k
    torch.cuda.empty_cache()
    generate_once("PRUNED-INPLACE")


@app.function(image=image, volumes={"/vol": volume}, timeout=3600, cpu=8, memory=16384)
def final_metadata_remote(
    gguf: str = "/vol/models/gguf/qwen3-k96-udmix.gguf",
    base_dir: str = "/vol/models/qwen3-30b",
    report_path: str = "/vol/models/compressed-qwen3-k96-banking10/compression_report.json",
) -> None:
    import hashlib

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    base = Path(base_dir)
    shards = sorted(base.glob("*.safetensors"))
    shard_info = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in shards]
    base_bytes = sum(x["bytes"] for x in shard_info)
    base_hash = hashlib.sha256(json.dumps(shard_info, sort_keys=True).encode()).hexdigest()
    model = Path(gguf)
    model_bytes = model.stat().st_size
    model_hash = sha256_file(model)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    result = {
        "base": {
            "dir": base_dir,
            "shards": len(shards),
            "bytes": base_bytes,
            "sha256_of_shard_manifest": base_hash,
            "shard_info": shard_info,
        },
        "compressed": {
            "path": gguf,
            "bytes": model_bytes,
            "sha256": model_hash,
            "byte_ratio": model_bytes / base_bytes,
            "params": report["compressed_params"],
            "param_ratio": report["compressed_params"] / report["base_params_expected"],
            "k": report["k"],
            "layers": len(report["keep"]),
            "experts_kept_per_layer": report["k"],
            "method": "top-96-of-128 experts by router usage + llama.cpp UD IQ2_XXS dynamic quant",
        },
    }
    out = Path("/vol/results/final/metadata.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "base"}, indent=2), flush=True)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=1800, cpu=2, memory=4096)
def hash_qwen3_models_remote(model_tag: str = "k7-banking10") -> None:
    import hashlib

    out = Path(f"/vol/results/qwen3/{model_tag}/hashes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": Path(f"/vol/models/compressed-qwen3-{model_tag}/model.safetensors"),
        "old_memory": Path("/vol/data/banking77_10class/old_memory.jsonl"),
        "combined_memory": Path("/vol/data/banking77_10class/combined_memory.jsonl"),
    }
    result = {}
    for name, path in paths.items():
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        result[name] = {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
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
