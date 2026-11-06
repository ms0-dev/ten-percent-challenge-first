# The 10% challenge — working solution

> Status: pilot in progress. Nothing in this directory is a final proof yet.

## Goal

Start from one trained model, keep about **10% of its size**, match its score
on a commercial benchmark, pass a **locked held-out exam**, and then let it
grow from user feedback **without training the weights**.

We satisfy the strictest reading of "size" by reporting both:

- parameter count: compressed / base
- on-disk checkpoint bytes: compressed / base

The retrieval memory/index is counted separately and kept small.

## Approach

1. **MoE expert pruning.** The base is an MoE. We keep the top-K experts per
   layer, selected by router usage on task calibration data (forward passes
   only). Every kept expert tensor is an exact slice of the base tensor.
2. **Same active path.** The base activates 4 experts per token; the compressed
   model keeps top-4 routing over the kept experts, so the per-token expert
   compute is unchanged.
3. **Non-parametric growth.** User feedback is stored in a retrieval memory.
   The model weights never change. Old knowledge is retained by construction;
   new intents/documents are learned by retrieving their examples.
4. **Locked exam.** Exam files are created and hashed before model development.
   They are never read for tuning or model selection.

## Pilot benchmarks

- **Banking77** — 67 banking support intents for the main task; 10 intents are
  held out as "new" for the feedback demo. Metric: top-1 accuracy.
  - Memory: 6,945 labeled train examples.
  - Dev: 1,733 held-out train examples.
  - Locked exam: 2,680 test examples (67 old intents).
  - Locked new-intent test: 400 test examples (10 new intents).
- **SQuAD v2 open-book QA** — pilot proxy for commercial support/legal QA.
  The answer is in the retrieved context. Metric: EM / F1.
  - Dev and locked exam are created by `prepare_squad.py`.
  - If QA wins the pilot, the final proof will switch to a commercial QA
    dataset (TechQA/PolicyQA/CUAD) and re-lock the exam.

The benchmark is chosen on **dev only**. The chosen benchmark's locked exam is
evaluated once.

## Locked commitments

Banking77 locked-exam hashes are in
[`locked/banking77_manifest.json`](locked/banking77_manifest.json):

- `old_test.jsonl` SHA-256: `255db6baeb5adb567c9be5cb17b6182214aed0f539c1a05fd1555c407b4cd602`
- `new_test.jsonl` SHA-256: `8107c48c90c5aa75f5cd9978a0bf10f3c493524fecc330016fd94793e4bda3dd`

SQuAD v2 pilot locked-exam hash is in
[`locked/squad_v2_manifest.json`](locked/squad_v2_manifest.json):

- `exam.jsonl` SHA-256: `c35ceb7537e83165b9054bcedab49545c23829c2405fc1ef5a8119b374594457`

## Base model (pilot)

`openai/gpt-oss-120b` — 116.8B total parameters, 5.1B active, official MXFP4
checkpoint (~63 GB). It fits one 80 GB GPU.

Target compressed model: keep 9–10 of 128 experts per layer, same top-4
routing. Parameter ratio ~8.8–9.5%; with an 8-bit non-expert tail the byte
ratio is ~9.8%. The kept expert tensors are bit-identical MXFP4 slices.

## Reproduction (Modal)

```bash
modal run solution/modal_app.py::prepare_banking77_remote
modal run solution/modal_app.py::prepare_squad_remote
modal run solution/modal_app.py::download_base_remote
modal run solution/modal_app.py::base_dev_eval_remote
modal run solution/modal_app.py::collect_router_remote
modal run solution/modal_app.py::build_compressed_remote --k 9
modal run solution/modal_app.py::compressed_dev_eval_remote --k 9
```

The final locked-exam evaluation is a separate function and is only run once,
after the benchmark and compressed model are frozen.

## Evidence to include in the final proof

- base vs compressed dev scores on both benchmarks
- parameter and byte ratios
- exact-slice proof / hashes for kept expert tensors
- no-training check (no optimizer, no gradients, weights frozen)
- locked exam scores (base and compressed), evaluated once
- feedback demo: new-intent accuracy before/after, old exam unchanged, model
  weight hash unchanged
- reproduction commands and seeds
