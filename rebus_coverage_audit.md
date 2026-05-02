# Full REBUS Coverage Audit

Date: 2026-05-02

## Bottom Line

PsyArbor W9 does not implement everything in `rebus-lab`. It implements a local, LM-shaped subset of the Layer 2 supervisor ideas plus a few Layer 1 calculation primitives for diagnostics and thresholding. It does not import `rebus-lab`, does not run the Layer 1 convex identification/certificate pipeline, does not use the Layer 3 CLF-CBF controller, and does not implement the Layer 4 `RebusHierarchicalController` contract.

The active root script is also important: `train_gpt.py:648` defines the baseline `GPT` model and has no `REBUS`, `rebus_`, `PsyArborLM`, `supervisor_step`, `community`, or `plasticity` hits from the audit search. The W9 REBUS/PsyArbor implementation lives in `records/track_non_record_16mb/2026-05-01_psyarbor_lab_rebus_calculations_w9/train_gpt.py`, not in the root training entrypoint.

The existing W9 diagnostics point to a training-path bottleneck, not to one broken auxiliary loss. Pure pretraining beats the episodic variants tested, and W9 episodic mode freezes most base LM parameters while updating routing/plasticity controls. In W9, `set_training_phase` gates trainability at `W9 train_gpt.py:2575`, and `outer_parameters` is limited to routers, plasticity controls, hypernets, adapter gates, and selected assembly gate/scale parameters at `W9 train_gpt.py:2584`.

Status terms:

- `direct`: implements the same practical role as `rebus-lab`.
- `adapted`: ports the idea, but changes the object being modeled or the math context.
- `diagnostic-only`: used for reporting or validation, not for training/control decisions.
- `missing`: not implemented in PsyArbor W9.
- `not applicable`: the REBUS control object does not currently have an LM-safe contract.

Usefulness terms:

- `high`: likely useful for the LM architecture if wired to the objective/trainable set correctly.
- `medium`: useful for observability or routing, but not sufficient to improve LM loss alone.
- `low`: weak link to LM quality or already dominated by simpler training.
- `defer`: do not implement until a concrete LM contract exists.

## Source Map

Path aliases used below:

- `W9 train_gpt.py`: `records/track_non_record_16mb/2026-05-01_psyarbor_lab_rebus_calculations_w9/train_gpt.py`
- `REBUS identification.py`: `C:/Users/jjor3/OneDrive/Documents/GitHub/rebus-lab/python/src/rebus/identification.py`
- `REBUS supervisor.py`: `C:/Users/jjor3/OneDrive/Documents/GitHub/rebus-lab/python/src/rebus/supervisor.py`
- `REBUS controller.py`: `C:/Users/jjor3/OneDrive/Documents/GitHub/rebus-lab/python/src/rebus/controller.py`
- `REBUS hierarchical.py`: `C:/Users/jjor3/OneDrive/Documents/GitHub/rebus-lab/python/src/rebus/hierarchical.py`

| Source | What It Establishes |
| --- | --- |
| `train_gpt.py:648` | Active root training script is baseline `GPT`, not PsyArbor/W9. |
| `W9 train_gpt.py:172` | W9 REBUS knobs and kill switches. |
| `W9 train_gpt.py:526`, `:546`, `:569`, `:585`, `:620` | W9 local ports for pinball quantiles, dispersion entropy, Huber attribution, bootstrap CIs, and kappa envelope. |
| `W9 train_gpt.py:2490` | W9 `supervisor_step` runtime path. |
| `W9 train_gpt.py:2881` | W9 sparse route is used only when supervisor is enabled, calibrated, and not CLOSED. |
| `W9 train_gpt.py:4511`, `:4520`, `:4565`, `:4589`, `:4678`, `:4681`, `:4784` | W9 model construction, pretrain/episodic schedule, objective mix, and supervisor call site. |
| `tools/debug_w9_training_path.py:285` | Diagnostic trainability buckets used to isolate base LM vs outer-only behavior. |
| `tools/debug_w9_training_path.py:427`, `:489` | Diagnostic snapshot/restore and eval-buffer mutation checks. |
| `REBUS identification.py:1545` | Full Layer 1 `identify_rebus_bounds` orchestration. |
| `REBUS supervisor.py:150`, `:211`, `:414`, `:547` | Layer 2 entropy, sparse routing, openness score, and runtime step. |
| `REBUS controller.py:24`, `:269` | Layer 3 control envelope and CLF-CBF safety filter. |
| `REBUS hierarchical.py:177`, `:253` | Layer 4 starter two-expert `sparse_topk` wrapper. |

## Current Run Evidence

The existing diagnostic matrices support the same conclusion: W9's REBUS machinery is active in episodic runs, but it is not improving the LM objective under the tested schedule.

Primary matrix: `runs/w9_training_path/20260502_102636_primary64-fullval/matrix_summary.json`

| Run | Final LM CE | Total Loss | Full Val BPB | Top Update Bucket | Final Mode | Growth Events |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `pure_pretrain64` | 5.8866 | 5.8866 | 3.5454 | `embedding_output` | CLOSED | 0 |
| `pretrain64_auxzero` | 5.8866 | 5.8866 | 3.5454 | `embedding_output` | CLOSED | 0 |
| `episodic64_lm_only_outer` | 6.6472 | 6.6472 | 3.9666 | `adaptation_plasticity` | GROW | 0 |
| `episodic64_full_aux_outer` | 6.6607 | 4.2124 | 3.9808 | `adaptation_plasticity` | GROW | 3 |

Extended matrix: `runs/w9_training_path/20260502_103656_extended32-sliceval/matrix_summary.json`

| Pattern | Result |
| --- | --- |
| 32-step pretrain variants | BPB clustered around 3.7331 to 3.7490. |
| 32-step episodic outer-only variants | BPB clustered around 4.1168 to 4.1193 regardless of which aux coefficient was zeroed. |
| 32-step episodic with base LM updates | Better than outer-only episodic in that matrix, but still worse than pretrain. |

Promoted ablation: `runs/w9_training_path/20260502_104503_episodic64_lm_only_base/matrix_summary.json`

| Run | Final LM CE | Total Loss | Full Val BPB | Top Update Bucket | Final Mode | Growth Events |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `episodic64_lm_only_base` | 5.9423 | 5.9423 | 3.7329 | `embedding_output` | GROW | 3 |

Interpretation:

- Pure 64-step pretraining is the best tested route.
- Full auxiliary episodic training can lower total mixed loss while LM CE remains worse.
- Zeroing individual auxiliary terms barely changes episodic results, so no single aux coefficient explains the BPB gap.
- The main bottleneck is the trainable set and schedule: W9 pretraining updates all parameters, while default episodic training updates mostly routing/plasticity controls.
- Eval mutates runtime buffers in current W9 checks. The diagnostic script found mutations in buffers such as route/community EMAs and sparse counters, then restores snapshots for probes. This did not explain the BPB gap in the slice checks, but production eval should still be side-effect-free or snapshot/restored.

## Layer 1 Identification Coverage

| REBUS-lab capability | REBUS-lab source | PsyArbor W9 status | Usefulness for PsyArbor LM | Notes |
| --- | --- | --- | --- | --- |
| Local plant fit with constrained robust ridge / Huber loss | `identification.py:671` | `missing` | `defer` | W9 has no state-space plant `x_{t+1}=Ax+Bu+d`, no `cvxpy`, and no constrained plant fit. LM hidden states could be modeled this way later, but the state/control contract is not defined. |
| Alpha/excess/kappa/budget gate fits | `identification.py:751`, `:808`, `:858`, `:889` | `adapted` for kappa only | `low` | W9 has `fit_kappa_envelope` at `W9 train_gpt.py:620`, but it is a NumPy subgradient envelope over LM loss vs episode index, not the full gate/budget identification pipeline. |
| Moving-block bootstrap resampling | `identification.py:635`, `:956` | `diagnostic-only` | `medium` | W9 has `moving_block_bootstrap_ci` at `W9 train_gpt.py:585`, but it computes CIs over per-step metric series at manifest time. It does not refit scenarios. |
| Bootstrap scenario bank | `identification.py:956` | `missing` | `defer` | W9 assemblies are trainable modules, not sampled scenario dynamics. No `BootstrapScenarios` equivalent is produced. |
| One-sided scalar bounds | `identification.py:1104` | `missing` | `defer` | W9 does not synthesize certified `lambda_lb`, `mu_lb`, `kappa_ub`, or budget upper bounds for controller use. |
| Robust contraction SDP / Lyapunov certificate | `identification.py:1156`, `:1327`, `:1391` | `missing` | `defer` | This is a core part of full REBUS-lab. W9 explicitly avoids `cvxpy`/SDP-style certification. |
| End-to-end `identify_rebus_bounds` orchestration | `identification.py:1545` | `missing` | `defer` | W9 ports a few scalar calculations locally but does not call or mirror the full Layer 1 pipeline. |
| Supervisor gain synthesis | `identification.py:1814` | `missing` | `defer` | W9 thresholds and losses are hand-configured or online-estimated; no `SupervisorGains` are produced. |

Layer 1 summary: W9 uses calculation motifs from Layer 1, not Layer 1 identification. The missing pieces are the full certification stack. They should not be blindly ported until PsyArbor defines what LM "state", "control", "budget", and "safe envelope" mean.

## Layer 2 Supervisor Coverage

| REBUS-lab capability | REBUS-lab source | PsyArbor W9 status | Usefulness for PsyArbor LM | Notes |
| --- | --- | --- | --- | --- |
| Ensemble entropy from bootstrap prediction cloud | `supervisor.py:150` | `adapted` | `medium` | W9 `log_det_cov_entropy` at `W9 train_gpt.py:546` computes log-det covariance over route probability history, not over scenario predictions. It is a routing-dispersion signal. |
| Calibration thresholds | `supervisor.py:272` initialization and threshold calibration | `adapted` | `medium` | W9 supports frozen quantile calibration and online pinball updates. Online threshold updates are at `W9 train_gpt.py:2340`. |
| Openness score combining entropy, health, sparsity, and budget | `supervisor.py:414` | `missing` | `high` | W9 directly checks entropy threshold crossings in `supervisor_step` at `W9 train_gpt.py:2531`. It does not compute a bounded composite openness score. This is one of the more useful missing Layer 2 ideas. |
| CLOSED/GROW/RECOVER mode FSM | `supervisor.py:547` | `adapted` | `medium` | W9 implements immediate threshold-based mode transitions at `W9 train_gpt.py:2541`. It lacks REBUS-lab's openness score and dwell behavior. |
| Sparse routing from prediction errors | `supervisor.py:211` | `adapted` | `high` | W9 sparse routing at `W9 train_gpt.py:2424` uses masked router scores minus per-assembly error EMA, then top-k renormalization. It is useful, but it is not scenario-error routing. |
| Health decay/recovery | `supervisor.py:473` | `adapted` | `high` | W9 `_update_assembly_health` at `W9 train_gpt.py:2399` decays inactive assemblies and refreshes active ones. It allows health up to 1.5, while REBUS-lab clips to 1.0. |
| Growth budget enforcement | `supervisor.py:393`, `:435` | `direct` / `adapted` | `medium` | W9 has `REBUS_GROWTH_BUDGET` at `W9 train_gpt.py:181` and blocks clone/perturb at `:2457`. It is direct in spirit but applied to assemblies. |
| Clone-and-perturb growth | `supervisor.py:180`, `:435` | `adapted` | `medium` | W9 `_maybe_clone_and_perturb` at `W9 train_gpt.py:2454` clones an assembly from a high-health donor to a low-health recipient. REBUS-lab clones scenario matrices near the best-fitting bootstrap draw. |
| Reset / transient recovery floor | `supervisor.py:533` | `missing` in production, `diagnostic-only` in tooling | `medium` | W9 has no production supervisor reset method. `tools/debug_w9_training_path.py:427` snapshots buffers and restores them for diagnostics. |
| Dwell before reopen/close | `supervisor.py:429`, `:573` | `missing` | `medium` | W9 mode transitions can flip on the next threshold crossing. Adding dwell would likely reduce thrashing in episodic routing. |
| Read-only diagnostics for route churn, age, health spread | `supervisor.py:368`, `:380`, `:384` | `partial` | `medium` | W9 reports mode counts, growth events, sparse active fraction, health, and CIs, but not a direct route-churn/age contract. |

Layer 2 summary: this is the layer W9 uses most. The implementation is conceptually aligned, but it substitutes LM routing probabilities and assemblies for REBUS-lab scenario predictions. The highest-value missing Layer 2 item is a bounded composite openness score with dwell/reset semantics, because that maps naturally to W9 without requiring a full SDP controller.

## Layer 3 Controller Coverage

| REBUS-lab capability | REBUS-lab source | PsyArbor W9 status | Usefulness for PsyArbor LM | Notes |
| --- | --- | --- | --- | --- |
| `ControlEnvelope` with runtime bounds | `controller.py:24` | `missing` | `defer` | W9 has no `alpha_bar`, `e_bar`, `omega_bar`, or `barV` control envelope. |
| CLF/CBF gain synthesis | `controller.py:108`, `:121` | `missing` | `defer` | W9 has loss weights and routing thresholds, not CLF/CBF gains. |
| `RebusHybridController.safety_filter` | `controller.py:269` | `missing` | `defer` | The safety filter solves a constrained control action. There is no current LM equivalent for `u_nominal` or bounded `u`. |
| Solver failure fallback | `controller.py:296`, `:331` | `missing` | `defer` | W9 has eval snapshot diagnostics but not controller fallback behavior. |

Layer 3 summary: not implemented and not directly applicable yet. The right next step is not to port `RebusHybridController`; it is to define an LM-safe analog of the control vector and envelope first, if one is needed.

## Layer 4 Hierarchical Coverage

| REBUS-lab capability | REBUS-lab source | PsyArbor W9 status | Usefulness for PsyArbor LM | Notes |
| --- | --- | --- | --- | --- |
| `TopologyConfig` / `ExpertRoutingConfig` schemas | `hierarchical.py:28`, `:121` | `missing` | `defer` | W9 has assemblies, communities, temp branches, temp edges, and adapter bank structures, but not the REBUS Layer 4 schemas. |
| Two-expert flat `sparse_topk` wrapper | `hierarchical.py:177`, `:253` | `missing` | `defer` | W9 does multi-assembly routing, not two-expert Layer 3 expert selection. |
| Dense blending / learned routing / non-flat topology | `hierarchical.py:180` | `missing` by design | `defer` | REBUS-lab itself defers these beyond the starter runtime. W9 should not claim coverage here. |
| Expert lifecycle semantics | `docs/layer4-design.md`, `hierarchical.py:180` | `adapted` only in broad spirit | `medium` | W9's temp branch/edge promotion and adapter-bank consolidation are relevant architecture experiments, but they are not Layer 4 REBUS controller semantics. |

Layer 4 summary: W9 has its own dynamic assembly architecture, but it is not an implementation of REBUS-lab Layer 4. The overlap is architectural inspiration, not interface compatibility.

## What W9 Actually Uses Well

W9 has meaningful REBUS-inspired behavior in these areas:

- Route dispersion and entropy thresholding: `W9 train_gpt.py:546`, `:2490`.
- Online threshold adaptation: `W9 train_gpt.py:526`, `:2340`.
- Per-assembly health and sparse routing: `W9 train_gpt.py:2399`, `:2424`, `:2881`.
- Growth budget and clone/perturb: `W9 train_gpt.py:2454`.
- Diagnostic uncertainty: `W9 train_gpt.py:585`, `:5036`.
- Off-trajectory LM-loss envelope reporting: `W9 train_gpt.py:620`, `:5076`.

These are the pieces most compatible with an LM training experiment. The issue is not that they are absent. The issue is that they mostly act on routing/plasticity state after the base LM has stopped receiving the main improvements.

## Main Gaps

1. The active root training script does not use W9 at all.
2. W9 does not implement full REBUS-lab Layer 1 identification or certification.
3. W9's entropy is route-dispersion entropy, not bootstrap prediction-cloud entropy.
4. W9 lacks REBUS-lab's composite bounded openness score.
5. W9 lacks production reset/dwell behavior, though diagnostic snapshot/restore exists.
6. W9 lacks Layer 3 and Layer 4 controller contracts.
7. W9 eval mutates runtime buffers and should be made side-effect-free or snapshot/restored.
8. The tested episodic schedule freezes most of the base LM and loses to pure pretraining.

## Recommended Next Step

The highest-confidence implementation follow-up is not to port all of REBUS-lab. It is to make W9's existing REBUS-inspired path compete fairly with pretraining:

1. Keep base LM parameters trainable during episodic LM-dominant runs.
2. Make auxiliary objectives secondary and report LM CE as the primary training health metric.
3. Add side-effect-free eval or snapshot/restore around eval.
4. Add a W9 composite openness score using entropy, health deficit, sparsity, and budget fraction.
5. Add dwell/reset semantics after the openness score exists.

Full Layer 1/3/4 REBUS-lab integration should remain deferred until PsyArbor defines an LM-specific state/control/envelope contract. Without that contract, porting SDP certificates or CLF-CBF filters would add complexity without a defensible objective link.
