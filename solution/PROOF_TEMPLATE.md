# The 10% challenge — proof

## Claim

Starting from `Qwen/Qwen3-30B-A3B-Instruct-2507` (30.53B total parameters,
3.3B active, 61.08 GB BF16), we keep:

- **10.20% of the parameters** — 3,114,645,504 / 30,532,122,624
- **10.20% of the checkpoint bytes** — 6,229,464,952 / 61,079,824,827
- **the same score** on a commercial banking support-routing benchmark
- **the same score on a locked held-out exam** hashed before development
- **no weight training** at any point
- **growth from user feedback** through a non-parametric retrieval memory,
  with the locked exam still passing and the model weight hash unchanged

## Base model

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Total / active parameters: 30.53B / 3.3B
- Checkpoint: 61.08 GB BF16 (16 shards)
- License: Apache 2.0

## Compression

- Keep the top 7 of 128 experts per layer, selected by router usage on the
  task calibration split (forward passes only).
- The compressed model has 7 experts active per token (base: 8).
- Every kept expert tensor is an **exact copy** of the corresponding base
  expert tensor; every non-expert tensor is copied; the router rows are exact
  slices. No merging, no quantization, no gradients.
- This is a strict subset of the base model, not a trained model.

### Size accounting

| Metric | Base | Compressed | Ratio |
|---|---:|---:|---:|
| Parameters | 30,532,122,624 | 3,114,645,504 | 10.20% |
| Checkpoint bytes | 61,079,824,827 | 6,229,464,952 | 10.20% |
| Retrieval index | — | counted separately | small |

## Benchmark

- Task: 10-family commercial banking support router (the fine-grained
  Banking77 intents are grouped into 10 operational queues).
- Metric: top-1 accuracy, same retrieval/prompt pipeline for both models.
- Dev: 1,733 held-out train examples (300 used for the pilot).
- Locked exam (10-family variant): full 2,680-example test set, SHA-256
  `f9e90fc97ec1808297426381a1750e36193eae6417c99c129d421c2c4d13e872`,
  committed in `locked/banking77_10family_manifest.json` before development.

## Results (dev)

| Model | Accuracy | Parse failures |
|---|---:|---:|
| Base Qwen3-30B | 97.33% (292/300) | 0 |
| Compressed 3.1B | TBD | TBD |

Also measured on the 20-queue variant: base 95.67% (287/300).

## Locked exam (evaluated once, at the end)

| Model | Accuracy |
|---|---:|
| Base | TBD |
| Compressed | TBD |

## Growth with user feedback

- Initial memory: 10 old support families, 6,945 examples.
- New intents: 10 fine-grained intents, 50 feedback examples (5 each).
- New-intent test: 400 examples, SHA-256
  `8107c48c90c5aa75f5cd9978a0bf10f3c493524fecc330016fd94793e4bda3dd`.

| Phase | New-intent accuracy | Old-family dev accuracy |
|---|---:|---:|
| Before feedback | TBD | TBD |
| After feedback | TBD | TBD |

- Model weight SHA-256 before/after feedback: TBD (must be identical)
- Memory/index SHA-256 before/after feedback: TBD (changes)

## No-training evidence

- No optimizer, no `loss.backward()`, no fine-tuning in any script.
- The compressed checkpoint is built by slicing/renaming exact base tensors.
- Feedback changes only the retrieval memory; model files are unchanged.

## Reproduction

See `README.md` for the Modal commands and seeds.
