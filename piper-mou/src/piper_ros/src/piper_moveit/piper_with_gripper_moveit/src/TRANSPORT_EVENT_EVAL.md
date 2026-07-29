# Transport A/B event evaluation

The evaluation mode records detector output and ground-truth event markers
separately. It does not retrain or relabel the model.

## 1. Start one condition

Use `--monitor_only` for the detector ablation so the first OR trigger does not
truncate observations from the other channel.

```bash
python3 basket_ori.py \
  --mode new_9d_relative \
  --eval_log \
  --monitor_only \
  --eval_trial_type abrupt
```

Valid trial types are `stable`, `abrupt`, and `gradual`. Stable trials require
no event markers.

For a controlled gradual-release trial, use:

```bash
python3 basket_ori.py \
  --mode new_9d_relative \
  --threshold_a 0.1259 \
  --infer_stride 10 \
  --consec 2 \
  --eval_log \
  --monitor_only \
  --eval_trial_type gradual \
  --gradual_opening 0.03 \
  --gradual_duration 5 \
  --gradual_observation 1
```

After the stable-grasp baselines are ready, `basket_ori.py` slowly commands the
gripper from its closed position to `--gradual_opening`. It records
`perturb_start` automatically at the command onset. Press only `l` when the
garment physically leaves the grasp. If no `l` is received by the end of the
opening command plus `--gradual_observation`, the trial is retained with status
`gradual_no_loss`; do not delete it.

## 2. Start the keyboard marker once

Run the marker once in another terminal:

```bash
python3 mark_transport_event.py
```

The process stays open. No Enter key is required:

- press `p` when a manually imposed pull begins (`gradual` records this
  automatically)
- press `l` when complete contact loss occurs
- press `e` for an optional manual trial-end marker
- press `q` to close the marker

Wait for `[EVENT-EVAL] trial started` before pressing a marker key.

For the primary controlled evaluation, use reproducible `stable` and `abrupt`
conditions. Do not claim a controlled gradual-slip result if slow manual pulls
cannot be reproduced consistently. Naturally occurring slow slips may be
reported separately with synchronized video, but they should not be mixed with
the controlled latency experiment.

The current and subsequent trial type can still be changed with a one-shot
command if needed:

```bash
python3 mark_transport_event.py type=gradual
```

## 3. Calculate A-only, B-only, and OR metrics

```bash
python3 evaluate_transport_events.py \
  --log_dir /home/dyros/mcy_ws/transport_event_eval \
  --deadline_s 2
```

The script reports event precision, event recall, F1, stable-trial false
triggers, median detection latency, and gradual-slip lead time. It also writes
`transport_event_report.json`.

## Closed-loop recovery trials

Omit `--monitor_only` to retain the normal motion-abort and recovery behavior:

```bash
python3 basket_ori.py \
  --mode new_9d_relative \
  --eval_log \
  --eval_trial_type abrupt
```

Detector ablation and closed-loop recovery should be reported as separate
experiments because recovery truncates the original tactile event.
