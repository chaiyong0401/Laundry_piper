# Phase-specific models with a shared detector structure

## Corrected interpretation

Grasp and Transport use separate Channel-A weights and phase-specific inputs,
but the same GRU-AE backbone:

- sequence length: 50
- hidden dimension: 32
- recurrent layers: 1
- dropout: 0
- training/inference stride: 10
- regular anomaly consecutive count: 2
- first-frame relative preprocessing

The existing Transport relative-9D model is preserved:

- test-success detections: 2/12
- failure detections: 15/25

The separately trained Grasp 6-D model with the same backbone gives:

- test-success detections: 0/8
- failure detections: 3/15

No deployed artifact was replaced.

## Shared previous-frame Channel B

Both phases use the same detector structure without an initial stable-contact
baseline:

1. compute active contact and peak taxel force from the current raw frame;
2. subtract the immediately preceding frame;
3. scale both changes using phase-specific train-success statistics;
4. take their RMS magnitude;
5. require three consecutive threshold exceedances.

The threshold is calibrated at p95 from validation-success episode statistics.
Thus the structure and physical quantities are shared while the numerical
scale and threshold remain phase-specific.

## Historical episode-labelled evaluation

### Grasp

| Configuration | Failure detections | Test-success detections |
|---|---:|---:|
| A | 3/15 | 0/8 |
| B | 5/15 | 0/8 |
| A OR B | 6/15 | 0/8 |

Failure overlap: both 2, A-only 1, B-only 3, neither 9.

### Transport

| Configuration | Failure detections | Test-success detections |
|---|---:|---:|
| A (existing relative-9D) | 15/25 | 2/12 |
| B (previous-frame contact/peak) | 10/25 | 2/12 |
| A OR B | 15/25 | 3/12 |

Failure overlap: both 10, A-only 5, B-only 0, neither 10.

These files have whole-episode labels and no event onset, so this is only a
preliminary discrimination evaluation.

## Event-labelled Transport evaluation

After excluding the interrupted gradual trial and the physically slipped file
that had been mislabelled stable:

| Configuration | Controlled failure | Stable false triggers |
|---|---:|---:|
| A | 35/40 | 3/31 |
| B (previous-frame contact/peak) | 29/40 | 2/31 |
| A OR B | 37/40 | 4/31 |

Failure overlap: both 27, A-only 8, B-only 2, neither 3.

This is genuine, though modest, complementary coverage. It trades lower
contact-loss recall for a detector definition that is applicable to both
phases without a stable initial baseline. The existing absolute-baseline
Transport B remains stronger for contact loss (39/40).

## GPU status

The active `rs` Python uses PyTorch `2.9.0+cu128`, but the host currently
reports:

`Failed to initialize NVML: Driver/library version mismatch`

and `torch.cuda.is_available()` is false. The Grasp model was therefore trained
with the CPU fallback. GPU versus CPU changes training speed, not the intended
architecture. GPU retraining can be repeated after the NVIDIA driver/library
installation is repaired or the host is rebooted.
