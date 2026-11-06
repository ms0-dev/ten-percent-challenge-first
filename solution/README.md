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
  - Base pilot on 300 dev examples: **94.33%** (283/300), 0 parse failures.
    The errors are concentrated in genuinely ambiguous fine-grained pairs.
- **SQuAD v2 open-book QA** — pilot proxy for commercial support/legal QA.
  The answer is in the retrieved context. Metric: four-way answer-selection
  accuracy (one correct answer + three distractors).
  - Dev and locked exam are created by `prepare_squad.py`.
  - Base pilot on 300 dev examples: **98.33%** (295/300), 0 parse failures.
  - If QA wins the pilot, the final proof will switch to a commercial QA
    dataset (TechQA/PolicyQA/CUAD) and re-lock the exam.
  - TechQA has been inspected on Modal: the Q&A files are clean
    (`training_Q_A.json`, `dev_Q_A.json`, `validation_*`), but its full
    technote corpus is large (`full_technote_collection.sections.json` is
    6.4 GB). It is a viable commercial final dataset, not the fast pilot.

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

## Base model (current)

**`Qwen/Qwen3-30B-A3B-Instruct-2507`** — 30.5B total parameters, 3.3B active,
61.08 GB BF16. Its 10% target is the cleanest possible:

- 7 of 128 experts per layer
- 3.115B parameters (**10.2%** of the base)
- 6.23 GB BF16 checkpoint (**10.2%** of the base)
- 7 experts active per token (base: 8)

The first pilot used `openai/gpt-oss-120b`, but its MXFP4 expert path is
technically blocked: the fused Triton kernels only work at the original
128-expert count, and reconstructing reduced-expert checkpoints through the
Python path produced garbage. Qwen3's experts are plain BF16 3D tensors, so
the 7-expert checkpoint runs through the standard `Qwen3MoeExperts` eager
path without Special kernels.

## Benchmark variants

The fine-grained Banking77 intents are grouped into commercially meaningful
support queues. Both variants use the same locked test texts and differ only
in label granularity:

- **20-queue** grouped exam SHA-256:
  `53c66c72e322fc82165c6ebb544140cae30c66351e2f8f2a8fe85d574e3a174a`
- **10-family** commercial router exam SHA-256:
  `f9e90fc97ec1808297426381a1750e36193eae6417c99c129d421c2c4d13e872`

Base Qwen3 dev results so far:

- 20-queue, 300 dev examples: **95.67%**, 0 parse failures
- 10-family, 300 dev examples: running

## Reproduction (Modal)

```bash
modal run solution/modal_app.py::prepare_banking77_remote
modal run solution/modal_app.py::prepare_squad_remote
modal run solution/modal_app.py::download_base_remote
modal run solution/modal_app.py::base_dev_eval_remote
modal run solution/modal_app.py::collect_router_remote
modal run solution/modal_app.py::build_compressed_remote --k 9 --usage banking
modal run solution/modal_app.py::compressed_dev_eval_remote --model-tag k9-banking
modal run solution/modal_app.py::make_combined_memory_remote
modal run solution/modal_app.py::growth_eval_remote --model-tag k9-banking
modal run solution/modal_app.py::final_banking_base_remote
modal run solution/modal_app.py::final_banking_compressed_remote --model-tag k9-banking
```

The final locked-exam evaluation is a separate function and is only run once,
after the benchmark and compressed model are frozen.

## Growth demo design

The deployed memory is namespaced by intent family:

- **Old partition:** 67 old intents, 6,945 examples. The locked old exam is
  evaluated in this partition.
- **New partition:** 10 new intents, 5 user-feedback examples each. The
  new-intent test is evaluated in this partition.

The model weights are shared and frozen. The growth demo reports:

- new-intent accuracy before feedback (old partition only; the new labels are
  not available)
- new-intent accuracy after feedback (new partition added)
- old-intent dev accuracy before/after feedback (interference check)
- old locked exam after growth (namespaced old partition)
- SHA-256 of the model files before/after (must be identical)
- SHA-256 of the memory/index before/after (changes; this is the growth)

## Evidence to include in the final proof

- base vs compressed dev scores on both benchmarks
- parameter and byte ratios
- exact-slice proof / hashes for kept expert tensors
- no-training check (no optimizer, no gradients, weights frozen)
- locked exam scores (base and compressed), evaluated once
- feedback demo: new-intent accuracy before/after, old exam unchanged, model
  weight hash unchanged
- reproduction commands and seeds
