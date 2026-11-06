# The 10% challenge — proof

> Draft template. Fill in after the pilot and final locked-exam evaluation.

## Claim

Starting from `openai/gpt-oss-120b` (116.8B total parameters, official MXFP4,
65.3 GB checkpoint), we produce a compressed model with:

- **X% of the base parameter count**
- **Y% of the base checkpoint bytes**
- **the same score** on a commercial benchmark
- **the same score on a locked held-out exam** that was hashed before
  development
- **no weight training** at any point
- **growth from user feedback** through a non-parametric retrieval memory,
  with the locked exam still passing and the model weight hash unchanged

## Base model

- Model: `openai/gpt-oss-120b`
- Total / active parameters: 116.8B / 5.1B
- Official checkpoint: 65.3 GB (MXFP4 experts + BF16 non-expert)
- Source: https://huggingface.co/openai/gpt-oss-120b
- License: Apache 2.0

## Compression

- Keep the top K experts per layer, selected by router usage on task
  calibration data (forward passes only).
- Keep top-4 routing, same as the base. The per-token expert compute is
  unchanged.
- Every kept expert tensor is an exact slice of the corresponding base tensor.
- Non-expert tail is quantized to 8-bit (final; pilot uses BF16 as an upper
  bound).
- No gradients, no optimizer, no fine-tuning, no distillation training.

### Size accounting

| Metric | Base | Compressed | Ratio |
|---|---:|---:|---:|
| Parameters | 116,829,156,672 | TBD | TBD |
| Weight bytes | 65,276,859,410 | TBD | TBD |
| Retrieval index | — | TBD | counted separately |

## Benchmark

- Task: Banking77 banking support intent classification (67 old intents;
  10 new intents for the feedback demo).
- Metric: top-1 accuracy, same retrieval/prompt pipeline for both models.
- Dev: 1,733 held-out train examples (300 used for the pilot).
- Locked exam: 2,680 test examples, SHA-256
  `255db6baeb5adb567c9be5cb17b6182214aed0f539c1a05fd1555c407b4cd602`,
  committed in `locked/banking77_manifest.json` before development.

## Results (dev)

| Benchmark | Base | Compressed | Parse failures |
|---|---:|---:|---:|
| Banking77 (67 intents, 300 dev) | 94.33% | TBD | 0 |
| SQuAD v2 open-book QA MC (300 dev) | 98.33% | TBD | 0 |

## Locked exam (evaluated once, at the end)

| Model | Accuracy |
|---|---:|
| Base | TBD |
| Compressed | TBD |

## Growth with user feedback

- Initial memory: 67 old intents, 6,945 examples.
- New intents: 10, 50 feedback examples (5 per intent).
- New-intent test: 400 examples, SHA-256
  `8107c48c90c5aa75f5cd9978a0bf10f3c493524fecc330016fd94793e4bda3dd`.

| Phase | New-intent accuracy | Old-intent accuracy (dev) |
|---|---:|---:|
| Before feedback | TBD | TBD |
| After feedback | TBD | TBD |

- Model weight SHA-256 before feedback: TBD
- Model weight SHA-256 after feedback: TBD (must be identical)
- Memory/index SHA-256 before feedback: TBD
- Memory/index SHA-256 after feedback: TBD (changed; this is the growth)

## No-training evidence

- No optimizer or `loss.backward()` in any script.
- Compression script only slices tensors; kept expert tensors compare equal to
  the base slices.
- Quantization is deterministic post-training weight quantization; no gradient
  updates.
- Feedback changes only the retrieval memory; model files are unchanged.

## Reproduction

See `README.md` for the Modal commands and seeds.
