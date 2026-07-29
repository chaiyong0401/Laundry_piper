# Final phase-specific dual-channel evaluation

## Evaluated design

Channel A uses the same GRU-AE backbone in both phases:

- sequence length: 50
- hidden dimension: 32
- recurrent layers: 1
- dropout: 0
- train/inference stride: 10

The phase-specific Channel-A inputs are:

- Grasp: absolute/cumulative 6D, no relative transform
- Transport: stable-grasp relative 9D

The shared previous-frame Channel B uses:

- active-contact and peak-force;
- current frame minus the immediately preceding frame;
- phase-specific robust scaling and validation-success p95 threshold;
- 30-frame warm-up;
- three consecutive threshold exceedances.

Fusion is `A OR B`.

## Historical episode-labelled evaluation

### Grasp

| Configuration | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: absolute 6D GRU-AE | 84.2% | 80.0% | 82.1% | 16 | 3 | 4 | 11 |
| B: previous-frame | 100.0% | 45.0% | 62.1% | 9 | 0 | 11 | 14 |
| A OR B | 85.7% | 90.0% | 87.8% | 18 | 3 | 2 | 11 |

Failure overlap: both 7, A-only 9, B-only 2, neither 2.

Previous-frame B improves Grasp recall from 80.0% to 90.0% without adding a
test-success false positive.

### Transport

| Configuration | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: relative 9D GRU-AE | 88.2% | 60.0% | 71.4% | 15 | 2 | 10 | 10 |
| B: previous-frame | 83.3% | 40.0% | 54.1% | 10 | 2 | 15 | 10 |
| A OR B | 83.3% | 60.0% | 69.8% | 15 | 3 | 10 | 9 |

Failure overlap: both 10, A-only 5, B-only 0, neither 10.

The whole-episode Transport labels do not show complementarity. They do not
record failure onset and are therefore secondary to the event-labelled result.

## Transport event-labelled evaluation

There are 41 controlled contact-loss trials and 31 stable trials.

### Previous-frame Channel B

| Configuration | Controlled failure TP | Stable false trigger | Event F1 |
|---|---:|---:|---:|
| A | 36/41 | 3/31 | 90.0% |
| B | 29/41 | 2/31 | 79.5% |
| A OR B | 38/41 | 4/31 | 90.5% |

Failure overlap: both 27, A-only 9, B-only 2, neither 3.

Previous-frame B provides two genuine detections not provided by Channel A,
but increases stable false triggers from 3 to 4.

### Stable-baseline Channel B

| Configuration | Controlled failure TP | Stable false trigger | Event F1 |
|---|---:|---:|---:|
| A | 36/41 | 3/31 | 90.0% |
| B | 40/41 | 0/31 | 97.6% |
| A OR B | 40/41 | 3/31 | 94.1% |

Failure overlap: both 36, A-only 0, B-only 4, neither 1.

Stable-baseline B dominates previous-frame B for Transport. It detects four
trials not detected by A; A does not add a detection beyond stable-baseline B
on this event set.

## Design decision

If an identical Channel-B implementation in both phases is mandatory, the
previous-frame detector is defensible:

- Grasp A OR B: 18/20 failure detections
- Transport event A OR B: 38/41 controlled-loss detections

However, it discards substantial Transport performance compared with the
stable-baseline guard.

The evidence-supported deployment design is therefore:

- Grasp: absolute 6D Channel A + previous-frame Channel B
- Transport: relative 9D Channel A + stable-baseline Channel B

Both B variants monitor the same physical quantities and use the same
consecutive contact-change decision mechanism. They differ only in the
reference policy available in each phase. This is best described as a
phase-conditioned reference strategy rather than claiming an identical
Channel-B implementation.

Historical results use whole-episode labels. Event-labelled Transport results
should be used for claims about detection timing and contact-loss performance.
