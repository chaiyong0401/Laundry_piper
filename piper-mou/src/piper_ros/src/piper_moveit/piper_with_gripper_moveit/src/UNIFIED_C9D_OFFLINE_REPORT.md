# Unified C9D offline retraining report

## Scope

The deployed models and their artifacts were not modified. New artifacts are
stored under:

- `unified_c9d_models/grasp`
- `unified_c9d_models/transport`

Both phases use the same per-frame representation:

`[MRL, Fx, Fy, Fz, Mx, My, Mz, active_contact, peak_force]`

Both Channel-A models use the same architecture and inference configuration:

- sequence length: 50
- train/validation stride: 10
- inference stride: 10
- GRU-AE: hidden size 32, one layer, no dropout
- relative transform: first frame
- threshold: validation episode-maximum p95
- MRL miss guard: disabled, so Channel A is evaluated as a pure one-class
  reconstruction model

Weights, scalers, and thresholds are phase-specific.

## Data split

| Phase | Success | Train | Validation | Test | Failure |
|---|---:|---:|---:|---:|---:|
| Grasp | 51 | 35 | 8 | 8 | 15 |
| Transport | 78 | 54 | 12 | 12 | 25 |

Episodes were split chronologically before sliding windows were created.

## Channel A

| Phase | Threshold | Test-success detections | Failure detections | Window ROC-AUC |
|---|---:|---:|---:|---:|
| Grasp | 0.266418 | 0/8 | 4/15 (26.7%) | 0.6574 |
| Transport | 0.269353 | 1/12 (8.3%) | 7/25 (28.0%) | 0.5470 |

For reference, the existing Transport relative-9D first-frame model reports
2/12 test-success detections and 15/25 failure detections at threshold
0.126551. The common C9D model reduces false detections slightly but loses
substantial failure recall.

## Unified offline Channel B

Historical raw files contain cumulative range-normalised taxel displacement,
not the absolute DC-corrected contact used by the current online Transport
Channel B. The offline common B therefore uses the same available signal in
both phases:

`coherent_contact = MRL * active_contact / 16`

It detects framewise changes relative to a three-frame mean, after 30 warm-up
frames, and requires five consecutive threshold exceedances. Its threshold is
calibrated from the validation-success episode statistic at p95.

## Dual-channel held-out results

### Grasp

| Configuration | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 100.0% | 26.7% | 42.1% | 4 | 0 | 11 | 8 |
| B | 100.0% | 20.0% | 33.3% | 3 | 0 | 12 | 8 |
| A OR B | 100.0% | 26.7% | 42.1% | 4 | 0 | 11 | 8 |

Failure overlap: both 3, A-only 1, B-only 0, neither 11.

### Transport

| Configuration | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 87.5% | 28.0% | 42.4% | 7 | 1 | 18 | 11 |
| B | 75.0% | 12.0% | 20.7% | 3 | 1 | 22 | 11 |
| A OR B | 87.5% | 28.0% | 42.4% | 7 | 1 | 18 | 11 |

Failure overlap: both 3, A-only 4, B-only 0, neither 18.

## Interpretation and limitations

The shared architecture is technically valid, but these results do not
demonstrate mutual complementarity: every B detection is already contained in
A, so OR fusion does not improve A.

The result must not be compared directly with the event-labelled real-robot
Transport Contact Guard. The historical training files cannot reproduce that
guard because they do not store the empty-gripper DC reference and absolute
contact trajectory.

Failure files are labelled at episode level and do not record failure onset.
They are suitable for preliminary discrimination checks, not a final
event-level latency or recovery claim.

The new models should not be deployed based on these metrics. A truly unified
absolute Contact Guard requires newly stored absolute tactile data (or the
event-evaluation NPZ format) for both phases.
