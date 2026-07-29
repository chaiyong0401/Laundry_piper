# `basket_isr.py` offline event replay

## Detector configuration

- Channel A: stable-grasp relative 9D GRU-AE
- Channel B signal: active-contact and peak-force
- Channel B reference: immediately preceding frame
- Channel B robust scales: `0.0191631977`, `0.0038665195`
- Channel B score threshold: `28.0570573`
- Channel B warm-up: 30 frames
- Channel B consecutive count: 3
- Fusion: A OR B

## Stable trials

| Configuration | False-trigger trials | Specificity |
|---|---:|---:|
| A | 3/31 | 90.3% |
| B | 2/31 | 93.5% |
| A OR B | 4/31 | 87.1% |

## Abrupt contact loss

| Configuration | Detected | Recall | Median latency from onset |
|---|---:|---:|---:|
| A | 20/20 | 100.0% | 0.504 s |
| B | 16/20 | 80.0% | 0.473 s |
| A OR B | 20/20 | 100.0% | 0.442 s |

## Gradual contact loss

| Configuration | Detected | Recall | Median latency from onset | Median lead before loss |
|---|---:|---:|---:|---:|
| A | 16/21 | 76.2% | 0.191 s | 0.386 s |
| B | 13/21 | 61.9% | 0.085 s | 0.534 s |
| A OR B | 18/21 | 85.7% | 0.095 s | 0.419 s |

Channel B had one pre-onset trigger in the gradual-loss set. It was not
counted as a true positive.

## Overall controlled contact loss

| Configuration | Detected | Recall | Stable false triggers | Event F1 |
|---|---:|---:|---:|---:|
| A | 36/41 | 87.8% | 3/31 | 90.0% |
| B | 29/41 | 70.7% | 2/31 | 79.5% |
| A OR B | 38/41 | 92.7% | 4/31 | 90.5% |

Failure overlap: both 27, A-only 9, B-only 2, neither 3.

Three gradual perturbation trials did not result in physical contact loss and
were excluded from the failure denominator. After perturbation onset, A
responded in 3/3, B in 2/3, and A OR B in 3/3 of these no-loss trials.

This is an offline replay of existing event logs. A new online robot run is
still required to validate scheduling, sensor timing, and recovery behavior in
`basket_isr.py`.
