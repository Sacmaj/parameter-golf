# W9 LM-First Diagnostic Summary

Date: 2026-05-02

## Result

W9's high validation BPB is best explained by the training schedule and objective path, not by BPB calculation, data/target alignment, optimizer deadness, or an architecture that cannot overfit.

The key finding is that pure language-model pretraining keeps improving validation BPB, while episodic schedules tested so far fail to improve over the checkpoint they start from.

## Core Evidence

| Run | Train LM CE | Total Loss | Full Val BPB | Top Update Bucket |
| --- | ---: | ---: | ---: | --- |
| `pure_pretrain64` | 5.8866 | 5.8866 | 3.5454 | `embedding_output` |
| `pure_pretrain128` | 5.5671 | 5.5671 | 3.4099 | `embedding_output` |
| `pure_pretrain256` | 5.2803 | 5.2803 | 3.1568 | `embedding_output` |
| `pure_pretrain512` | 4.7857 | 4.7857 | 2.9557 | `embedding_output` |
| `episodic64_lm_only_outer` | 6.6472 | 6.6472 | 3.9666 | `adaptation_plasticity` |
| `episodic64_full_aux_outer` | 6.6607 | 4.2124 | 3.9808 | `adaptation_plasticity` |
| `pure256_then_lm_dominant_ep128` | 5.2263 | 5.2867 | 3.1722 | `embedding_output` |
| `pure256_then_full_aux_ep128` | 5.1967 | 3.1813 | 3.1683 | `embedding_output` |

## Interpretation

- `pure_pretrain512` is the best tested path at `2.9557 BPB`.
- The original 64-step episodic path lowers mixed total loss while LM CE remains poor.
- Keeping base LM parameters trainable helps, but episodic variants still do not beat the pure-pretrain checkpoint they start from.
- Community selection diagnostics did not explain the BPB gap; Community_Finder produced similar assignments and did not correlate with better BPB.
- Eval mutates runtime buffers in W9, so diagnostic reporting uses snapshot/restore for the reported BPB path.

## Recommendation

Promote longer pure LM pretraining as the next practical direction. Keep REBUS, community routing, and episodic adaptation diagnostic-only until an episodic schedule can improve validation BPB over its own pre-episodic checkpoint.

The next bounded experiment should be `pure_pretrain1024` or a production schedule that stays in pure LM mode longer before enabling any episodic objective.
