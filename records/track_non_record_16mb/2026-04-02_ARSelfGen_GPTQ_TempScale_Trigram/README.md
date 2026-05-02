# Non-Record Draft: AR Self-Gen GPTQ + AR Temp Scale + Output-Side N-gram Tilt

This folder is a contest-shaped candidate fork of the legal record [2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072](../../track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072). It is intentionally scoped to a single self-contained `train_gpt.py` under `records/`, not the repo-wide boundary-integrity harness.

Status: **draft candidate, not yet run.** The prior input-side `TRIGRAM` hash mechanism has been replaced (see "Why we dropped the input-side trigram" below). The new output-side n-gram tilt code has been smoke-tested in isolation but not run end-to-end on 8×H100. Placed in `track_non_record_16mb` because val_bpb is null pending rerun.

> **Caveat:** the parent record this draft forks is at val_bpb 1.1147 (March 25). Current SOTA is well below that (PR #2140 unmerged at 1.05601 3-seed mean). Even with the n-gram tilt working perfectly, this draft cannot beat current SOTA without rebasing onto a newer base. The two novelties here are independent of that stack and could be ported on top of it in a future PR.

## What changed in this revision

1. **Dropped:** `BigramHashEmbedding.trigram_hash` (input-side hash trigram blending into the bigram embedding table). The `TRIGRAM` and `TRIGRAM_ALPHA` env vars are no longer read.
2. **Added:** Output-side Nacrith-style interpolated 1..K-gram tilt in `eval_val` and `eval_val_sliding`. New `OnlineNgramTilt` class + `build_ngram_tilt_hints` helper. Closed-form tilt application via `apply_tilt_to_per_token_nll`.
3. **Kept:** AR self-gen GPTQ, AR-self-gen eval-temperature search, all parent compliance properties.

## Why we dropped the input-side trigram

The prior approach hashed `(t-2, t-1, t)` into the same `bigram_vocab_size`-bucket embedding table and added the result to the bigram embedding output. The previous version of this README already flagged the problem: at vocab=1024 with 3072 buckets, Zipf-realistic token distributions produce ~8× hot buckets (chi² ≈ 78k vs uniform expectation ~3071). Combined with a shared embedding table, the technique was as likely to hurt as to help — a structurally noisy signal mixed into the model's input representation.

## What the new tilt does (output-side, count-based, no hash collisions)

Per Nacrith (Tacconelli, arXiv:2602.19626), maintain explicit per-context count tables for orders 1..K with smooth interpolated backoff:

```
P_uni(t)   = (c(t) + 1) / (N + V)                     # Laplace-smoothed unigram
lambda_k   = n_k / (n_k + eps)                        # n_k = times context c_k seen
P_k(t|c_k) = lambda_k * P_hat_k(t|c_k) + (1 - lambda_k) * P_{k-1}(t)
```

For each scored val position, compute the interpolated argmax `h_t` and its smoothed probability `p_hat`. If `p_hat >= NGRAM_TILT_THRESHOLD`, apply Nacrith's closed-form output-side tilt to the model's per-token NLL:

```
ptl'(y_t) = ptl(y_t) - boost * 1[y_t == h_t] + log1p(q_t * (exp(boost) - 1))
```

where `q_t` is the model's (post-temperature) probability for the hinted token. This is provably normalized over the vocabulary for any prefix-derived `h_t` and `boost >= 0` (see Nacrith §3 / PR #2140's `online_ngram_tilt.py` for the proof).

## Causal compliance (Nacrith C1-C4)

- **C1 causal:** `OnlineNgramTilt.predict_top()` uses only the strict prefix (the ring buffer of previously observed tokens). `observe(t)` updates counts only after the position is scored.
- **C2 normalized:** the closed-form tilt formula's `Z_t = 1 + q_t * (exp(boost) - 1)` is the analytic normalizer over the full vocabulary axis.
- **C3 score-before-update:** `build_ngram_tilt_hints` does a single L→R pass; at position i it predicts using prefix `[0..i-1]` and only then calls `observe(tokens[i])`.
- **C4 single pass:** hints are pre-computed once per eval and re-used across the post-quant, sliding, and sliding-s64 evals.

## Suggested run command

```bash
COMMON_ENV="BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 XSA_LAST_N=11 \
NGRAM_TILT_ENABLED=1 NGRAM_TILT_MAX_ORDER=3 NGRAM_TILT_EPS=5 \
NGRAM_TILT_THRESHOLD=0.5 NGRAM_TILT_BOOST=1.5 \
AR_CALIB_NUM_SEQS=64 AR_CALIB_BATCH_SIZE=8 AR_CALIB_TEMPERATURE=0.8 \
AR_EVAL_TEMP_ENABLED=1 AR_EVAL_TEMP_GRID=0.92,0.96,1.00,1.04,1.08 AR_EVAL_TEMP_NUM_SEQS=16 \
WARMDOWN_ITERS=4000 TARGET_MB=15.88"

for SEED in 42 314 999; do
  env $COMMON_ENV SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 \
    | tee train_seed${SEED}.log
done
```

## New env-var knobs

| Name | Default | Purpose |
|---|---|---|
| `NGRAM_TILT_ENABLED` | `0` | Master switch. Set to `1` to enable. |
| `NGRAM_TILT_MAX_ORDER` | `3` | Maximum n-gram order. 1 = unigram only; 3 = trigram (matches the folder's name). |
| `NGRAM_TILT_EPS` | `5.0` | Backoff smoothing epsilon. `lambda_k = n_k / (n_k + eps)`. |
| `NGRAM_TILT_THRESHOLD` | `0.5` | Gate the tilt only when `p_hat >= threshold`. Below threshold, no tilt is applied. |
| `NGRAM_TILT_BOOST` | `1.5` | Multiplicative boost in log-space. Also called `beta`. |
| `NGRAM_TILT_MAX_CONTEXTS` | `500_000` | Hard cap on total context entries across all orders (random eviction when full). |
| `NGRAM_TILT_MAX_CONTINUATIONS` | `64` | Per-context successor cap (lowest-count eviction when exceeded). Nacrith §3.5 optimization. |

## Ablation Plan

A 2×2 ablation across the two novelties tells the story honestly. Each cell × 3 seeds = 12 runs.

| | `NGRAM_TILT_ENABLED=0` | `NGRAM_TILT_ENABLED=1` |
|---|---|---|
| `AR_EVAL_TEMP_ENABLED=0` | parent baseline replication | tilt-only |
| `AR_EVAL_TEMP_ENABLED=1` | temp-search-only | both novelties |

Decision rule: ship the cell with the lowest 3-seed mean. If the (0,0) parent baseline cell wins, ship as a clean negative result.

## Hypotheses

- **Tilt:** unclear-priors. With vocab=1024 and ~5M val tokens, an order-3 model has plenty of warmup and stable counts. The threshold gate should keep the tilt firing only on high-confidence positions (e.g., named entities reappearing within the val stream, common bigrams the model under-predicts). My honest expected delta is **0 to +0.005 nats**, with high seed variance. The win, if any, is dominated by order-2 lift on common bigrams.
- **AR-eval-temperature:** expected to converge near 1.0 on self-generated tokens, because the model is by construction calibrated to its own samples. A flat curve over `0.92→1.08` is the predicted negative result; a clear minimum away from 1.0 would be surprising and informative.
- **Combined:** the two pieces are mathematically orthogonal — temperature is a global multiplicative scalar on logits before softmax, the tilt is an additive correction on per-position log-probabilities. They compose without interference.

## Ablation Results

_(populated after the runs land — replace with mean ± std per cell)_

| | `NGRAM_TILT=0` | `NGRAM_TILT=1` |
|---|---|---|
| `AR_EVAL_TEMP=0` | TBD | TBD |
| `AR_EVAL_TEMP=1` | TBD | TBD |

## Files

- [train_gpt.py](train_gpt.py): candidate submission script (~2465 lines)
- [requirements.txt](requirements.txt): runtime deps copied from the SOTA lineage
- [submission.json](submission.json): draft metadata stub awaiting real results

## Notes for the next runner

- `OnlineNgramTilt` is pure Python with dict-of-dicts state. For 5M val tokens at order 3, expect ~10-30 seconds spent in `build_ngram_tilt_hints` before the first GPU eval — this is reported as `ngram_tilt:hints_elapsed_ms` in the log. If it dominates the 600s eval budget, consider lowering `NGRAM_TILT_MAX_ORDER` or implementing a C extension.
- Hints are built **redundantly on every rank** (deterministic from val_tokens + seed=fixed). No broadcast needed.
- The non-tilt path is byte-identical to the parent's eval — gating with `NGRAM_TILT_ENABLED=0` recovers the original numerics exactly.
- This draft has **not** been run end-to-end on 8×H100. The OnlineNgramTilt class is smoke-tested in isolation (a/b alternation pattern → 5/5 hint==true after warmup; Laplace-correct empty-state behavior), but the full pipeline including AR temp interaction, GPTQ + tilt, sliding eval + tilt has only been byte-compiled. Bugs may surface on first real run.
- The repo-wide `src/` harness is not part of this submission path and should not be included in a challenge PR.

## Why this still cannot beat current SOTA without a rebase

Current SOTA (PR #2140 unmerged) is at 1.05601. The parent this draft forks is at 1.1147. Best-case n-gram tilt + AR temp delta is ≤ 0.005 nats. The gap to SOTA from this base is **~0.058 nats** — an order of magnitude larger than what these techniques can close. To compete, this draft's two novelties (output-side tilt + AR temp search) need to be ported on top of a current-SOTA base. That's a separate piece of work.

## Credits

- Output-side tilt mechanism: Nacrith (Tacconelli, arXiv:2602.19626v2) — interpolated 1..K-gram with `lambda_k = n_k/(n_k+eps)` backoff and closed-form NLL tilt.
- Closed-form tilt formula and C1-C4 compliance framing: PR #2140 (`online_ngram_tilt.py` by simon-marcus, building on PR #1145 by AnirudhRahul).
- AR self-gen GPTQ + AR temp scale: this folder's parent draft.
- Parent record for the rest of the stack: [2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072](../../track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072).
