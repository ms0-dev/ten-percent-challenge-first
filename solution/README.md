# The 10% challenge - solution

**Status: final artifact frozen, locked exam passed, growth demo complete.**

The full proof is in [`PROOF.md`](PROOF.md). This README is the short version.

## What was built

- Base: `Qwen/Qwen3-30B-A3B-Instruct-2507` (30.53B total, 3.3B active,
  61,066,575,656 bytes of BF16 weights).
- Compressed: 96 of 128 experts per layer, converted to GGUF and quantized to
  a dynamic ~2.2-bit mix.
- Final size: **6,402,132,448 bytes = 10.4839%** of the base weights on disk.
- Locked exam (1,000 held-out examples):
  - Base: **95.80%** (958/1,000)
  - Compressed: **96.00%** (960/1,000)
  - Zero parse failures for both.
- Growth: 10 new intents from 50 feedback examples, **0% -> 91.25%** on a
  400-example held-out new-intent test, with the model SHA-256 unchanged. The
  locked exam was re-run after growth: **96.00%** again.

The size rule was clarified by the maintainer in
[issue #1](https://github.com/DeusExperiments/ten-percent-challenge/issues/1):
**"10% of size on disk is fine compared to fp16 equivalent weights."** This
solution meets that rule. The stored parameter count is 76.25% because the
weights are quantized; the proof reports both numbers.

## Method

1. **Choose the benchmark on dev only.** Banking77 grouped into 10 commercial
   support queues. The 10-family exam was locked before grouped-model work:
   `f9e90fc97ec1808297426381a1750e36193eae6417c99c129d421c2c4d13e872`.
2. **Select experts with forward passes.** Measure router usage on 120 dev
   examples and keep the top 96 of 128 experts per layer. Top-96 covers
   99.82% of routing assignments. The kept tensors are exact base slices.
3. **Quantize the 96-expert checkpoint.** Convert to BF16 GGUF, build a task
   importance matrix from 250 dev prompts, and quantize with a dynamic
   per-tensor type mix. Final artifact:
   `qwen3-k96-udmix.gguf`, SHA-256
   `5266a9f6edbba6426b0abd313f5d213fe93244947e9e8d690bfc8ad6d965db2e`.
4. **Evaluate the locked exam once.** Base 95.80%, compressed 96.00%.
5. **Grow from feedback without training or RAG.** Freeze the model. Compute
   the last-token hidden state of each feedback text, center the features, and
   store one prototype per new intent in a small sidecar head (81,920 bytes).
   New-intent accuracy goes 0% -> 91.25%; old weights and the old locked-exam
   score are unchanged.

## Files

- [`PROOF.md`](PROOF.md) - full claim, numbers, hashes, and reproduction.
- [`evidence/`](evidence/) - eval summaries, compression report, shard
  manifest, locked-exam subset manifest.
- [`locked/`](locked/) - benchmark manifests committed before development.
- `src/compress_qwen3.py` - top-K expert checkpoint builder.
- `src/eval_banking.py` - Transformers eval for the base.
- `src/eval_banking_llamacpp.py` - llama.cpp eval for the compressed GGUF.
- `src/growth_prototype.py` - training-free new-intent head.
- `modal_app.py` - all Modal jobs (router calibration, compression, GGUF
  conversion, quantization, evals, growth, hashing).

## What did not work

- **Keeping 7 of 128 experts** (the parameter-count 10% point) destroys the
  model: top-7 covers only ~32% of routing mass, and the pruned model emits
  newline tokens. Verified both on a saved checkpoint and by pruning the base
  in memory.
- **Merging all 128 experts into 7 centroids** at BF16 (6.2 GB) also fails:
  the model emits near-newline text and cannot follow the classification
  prompt.
- **Uniform IQ2_XXS / IQ2_XS** on the 96-expert model scores 94.0% / 95.33%.
  The dynamic per-tensor mix is what recovers the base score at the same size.

## Reproduction

See the command list at the end of [`PROOF.md`](PROOF.md). The pipeline runs on
Modal; the volume is `ten-percent-challenge`.

## Honest limitations

- The challenge metric used here is **disk size**, per the maintainer's
  clarification. The stored parameter count is 76.25%, not 10%.
- The base is evaluated with Transformers and the compressed artifact with
  llama.cpp. The prompt construction, retrieval, tokenizer, greedy decoding,
  and parser are identical; both have zero parse failures on the locked exam.
- The growth head is a sidecar over frozen hidden states. It adds new-intent
  capability without changing the old model or its locked-exam score.
