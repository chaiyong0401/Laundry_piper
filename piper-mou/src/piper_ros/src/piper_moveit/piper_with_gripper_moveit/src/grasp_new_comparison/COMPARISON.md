# Grasp `grasp_new`: absolute 6D vs absolute contact 9D

## Controlled setup

- Data: `/home/dyros/mcy_ws/grasp_new`
- Episode split: chronological
- Success episodes: 90 (train 62, validation 14, test 14)
- Failure episodes evaluated: 20
- Two failure episodes shorter than 50 frames were excluded in both runs
- Sequence length: 50
- Train/validation stride: 10
- Inference stride: 10
- GRU-AE: hidden size 32, one layer, no dropout
- Relative transform: none
- Threshold: validation episode-maximum p95
- MRL miss guard: disabled
- Seed: 42

Only the per-frame input representation differs between the two runs.

## Results

| Input | Threshold | Test-success FP | Failure TP | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing grasp 6D | 1.015725 | 3/14 | 16/20 | 84.2% | 80.0% | 82.1% | 0.8128 | 0.8559 |
| Absolute contact 9D | 0.836090 | 2/14 | 5/20 | 71.4% | 25.0% | 37.0% | 0.6368 | 0.5990 |

Failure-detection overlap:

- detected by both: 5
- detected only by 6D: 11
- detected only by 9D: 0
- detected by neither: 4

## Interpretation

The existing 6D representation is substantially better for grasp-failure
detection on this split. The 9D representation's five detections are a strict
subset of the 6D detections, so OR fusion between these two Channel-A feature
sets would not increase failure recall.

The 6D representation retains normalized directional-coherence quantities
(`norm`, `mean_unit_x/y/z`) in addition to MRL and peak magnitude. These are
discriminative for an empty or poorly established grasp. The 9D representation
aggregates taxels into resultant force and moment; cancellation between taxels
can suppress the grasp/no-grasp difference, while active-contact and peak-force
values can overlap under motion and sensor noise.

The deployed artifacts in `/home/dyros/mcy_ws/grasp_new` were not modified.
