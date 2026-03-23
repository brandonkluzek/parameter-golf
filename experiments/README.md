# Experiment Ops

This directory contains tracked experiment manifests for the root experiment runner:

```bash
python -m tools.experiments bootstrap-runpod --host <host> --train-shards 1
python -m tools.experiments prepare experiments/baseline_1gpu_smoke.toml
python -m tools.experiments run-remote experiments/baseline_1gpu_smoke.toml --host <host>
python -m tools.experiments enqueue experiments/timing_ladder.toml --all-variants
python -m tools.experiments show-set experiments/queue_sets/bringup_1gpu.toml
python -m tools.experiments enqueue-set experiments/queue_sets/bringup_1gpu.toml
python -m tools.experiments queue-runner --host <host> --capacity-gpus 8
python -m tools.experiments queue-status --host <host>
python -m tools.experiments status <run_id>
python -m tools.experiments collect <run_id>
python -m tools.experiments promote <run_id>
python -m tools.experiments queue-cancel <job_id>
```

The manifests here are the tracked control plane.
Tracked queue batches live under `experiments/queue_sets/`.
Local run state is written under `.runs/` and is intentionally ignored by git.

## Queue Model

- Use `run-remote` when you want to launch one run immediately and manage it by hand.
- Use `enqueue-set` when you want the tracked, named batch for a wave such as bring-up or checkpoint calibration.
- Use raw `enqueue` only when you are doing an ad hoc manifest matrix or a one-off queue experiment that is not yet promoted into a tracked queue-set.
- `queue-runner` always runs on the local machine. It watches local queue state under `.runs/queue/` and launches work onto the remote host over SSH.
- `queue-status` reads the local queue state and shows what the local scheduler believes is queued, blocked, running, skipped, failed, or completed.
- `status <run_id>` is per-run and reads the run heartbeat / summary, not the whole queue.

## Queue State Meanings

- `queued`: ready to launch when a compatible GPU slot is free
- `blocked`: manifest is valid, but a hard dependency is missing, usually a required env like `INIT_STATE_DICT_PATH`
  or an explicit manual block reason for a future pilot that still needs code support
- `skipped`: estimator says the run violates a hard gate before launch, such as `over_byte_cap` or `over_eval_budget`
- `launching`: queue-runner has claimed the job and is creating the remote `tmux` session
- `running`: remote session is alive or the heartbeat is updating normally
- `collecting`: remote summary says the run completed and queue-runner is copying logs/artifacts back into `.runs/<run_id>/`
- `completed`: collect succeeded and the run is now in the local registry
- `failed` / `failed_overrun`: remote session died unexpectedly or the observed step time exceeded the configured overrun threshold
- `cancelled`: operator cancelled the queued or active job

## Queue Files

- `.runs/queue/events.jsonl`: append-only queue event log
- `.runs/queue/state.json`: materialized current queue state
- `.runs/queue/hosts/<host>.json`: host-specific capacity / drain / active-slot state
- `.runs/queue/runner.log`: scheduler loop log

Treat these as local scheduler state, not as submission artifacts.

## What The Runner Does

- `bootstrap-runpod` is the idempotent pod sync/update command. It checks out the current git SHA at `/workspace/parameter-golf`, verifies the official `runpod/parameter-golf:latest` environment shape, and ensures the requested tokenizer plus at least `--train-shards N` training shards are present.
- `prepare` resolves a manifest plus any matrix expansion or `KEY=VALUE` overrides and writes the local run scaffold under `.runs/<run_id>/`.
- `run-remote` launches a single manifest variant in a remote `tmux` session and records the remote metadata locally.
- `enqueue`, `queue-runner`, `queue-status`, and `queue-cancel` provide a small queueing layer for shared 1-GPU or 8-GPU hosts.
- `show-set` and `enqueue-set` expand tracked queue batches from `experiments/queue_sets/`.
- `status`, `collect`, and `promote` are the main post-launch lifecycle commands.

## Queue Command Hierarchy

For the live 1xH100 workflow, the command choice should be:

1. `bootstrap-runpod`
2. `show-set`
3. `enqueue-set`
4. `queue-runner`
5. `queue-status`
6. `status <run_id>` only when you want to inspect one run in detail
7. `collect` only for manual recovery or manual-run workflows; queue-runner already auto-collects completed queued jobs

For checkpoint-gated work:

1. record the checkpoint path from the completed fullshape run
2. inject it with `--set INIT_STATE_DICT_PATH=<remote_path>` at `show-set` or `enqueue-set` time
3. confirm the set is no longer `blocked`

## Pod Sync Rule

- Use `--train-shards 1` only for smoke and bring-up.
- Before `baseline_1gpu_fullshape` or any checkpoint-producing run, rerun:

```bash
python -m tools.experiments bootstrap-runpod --host <host> --train-shards 80
```

- The bootstrap summary reports the current shard count, tokenizer presence, and whether the pod had to top up its cached dataset.

## Minimal Workflow

1. Bootstrap the remote host once:

```bash
python -m tools.experiments bootstrap-runpod --host <host> --train-shards 1
```

2. Inspect the available variants for a manifest:

```bash
python -m tools.experiments list-variants experiments/timing_ladder.toml
```

3. Launch one manifest directly:

```bash
python -m tools.experiments run-remote experiments/baseline_1gpu_smoke.toml --host <host>
```

4. Watch local state and collect artifacts when the run finishes:

```bash
python -m tools.experiments status <run_id>
python -m tools.experiments collect <run_id>
```

5. Promote successful runs into the tracked result area:

```bash
python -m tools.experiments promote <run_id>
```

## Queue Workflow

Use a tracked queue-set for the first live wave:

```bash
python -m tools.experiments bootstrap-runpod --host <host> --train-shards 1
python -m tools.experiments show-set experiments/queue_sets/bringup_1gpu.toml
python -m tools.experiments enqueue-set experiments/queue_sets/bringup_1gpu.toml
python -m tools.experiments queue-runner --host <host> --capacity-gpus 1 --train-shards 1
python -m tools.experiments queue-status --host <host>
```

After the first fullshape checkpoint is collected, top the pod up to the full dataset and unlock the checkpoint-gated sweeps:

```bash
python -m tools.experiments bootstrap-runpod --host <host> --train-shards 80
python -m tools.experiments show-set experiments/queue_sets/checkpoint_calibration_1gpu.toml --set INIT_STATE_DICT_PATH=<remote_path>
python -m tools.experiments enqueue-set experiments/queue_sets/checkpoint_calibration_1gpu.toml --set INIT_STATE_DICT_PATH=<remote_path>
```

The exploratory research wave stays separate from the checkpoint-calibration lane:

```bash
python -m tools.experiments show-set experiments/queue_sets/research_wave_1gpu.toml
python -m tools.experiments enqueue-set experiments/queue_sets/research_wave_1gpu.toml
```

The future blocked placeholders are tracked explicitly and should remain blocked until the required code lands:

```bash
python -m tools.experiments show-set experiments/queue_sets/future_pilots_blocked.toml
```

Expected first-wave behavior:

- `baseline_1gpu_smoke` should launch first and validate the whole path
- `baseline_1gpu_fullshape` should produce the first reusable checkpoint
- the timing anchors should follow in queue priority order
- the checkpoint-calibration set should stay `blocked` until `INIT_STATE_DICT_PATH` is supplied
- the future blocked queue-set should show explicit manual block reasons instead of launching

## 1xH100 Operator Sequence

Use this exact sequence for the current pod:

1. Commit and push tracked changes.
2. Sync the pod for smoke work:

```bash
python -m tools.experiments bootstrap-runpod --host <host> --train-shards 1
```

3. Dry inspect the first wave:

```bash
python -m tools.experiments show-set experiments/queue_sets/bringup_1gpu.toml
```

4. Enqueue and run it:

```bash
python -m tools.experiments enqueue-set experiments/queue_sets/bringup_1gpu.toml
python -m tools.experiments queue-runner --host <host> --capacity-gpus 1 --train-shards 1
```

5. Watch the queue from another terminal:

```bash
python -m tools.experiments queue-status --host <host> --watch
```

6. After the smoke path is healthy, top the pod up to the full dataset:

```bash
python -m tools.experiments bootstrap-runpod --host <host> --train-shards 80
```

7. Record the reusable fullshape checkpoint in `docs/competition/STATUS.md`.
8. Unlock and enqueue the checkpoint-dependent wave with `INIT_STATE_DICT_PATH`.

## Raw Queue Workflow

Use the queue when you have a manifest matrix or a shared host and want reproducible scheduling instead of manual `tmux` juggling.

```bash
python -m tools.experiments enqueue experiments/timing_ladder.toml --all-variants
python -m tools.experiments queue-runner --host <host> --capacity-gpus 1
python -m tools.experiments queue-status --host <host>
```

For 8xH100 nodes, switch `--capacity-gpus` to `8`. Jobs inherit their GPU requirement from the manifest resource class, so the runner can pack 1-GPU jobs or reserve a full 8-GPU host for record-style runs.

## Operational Notes

- Remote deployment requires a clean local git tree because the runner checks out the current commit SHA on the remote host.
- Commit and push tracked tooling changes before syncing the pod. The remote host only sees the checked-out git SHA, not uncommitted local files.
- The default remote root is `/workspace/parameter-golf`; override it with `--remote-root` if your host layout differs.
- The intended pod image is `runpod/parameter-golf:latest`.
- `queue-runner` should usually be the only process launching queued jobs onto a given host. Avoid mixing manual `run-remote` launches onto the same pod unless you intend to manage the interaction yourself.
- Queue-runner already auto-collects completed queued runs, updates `.runs/index.jsonl`, and frees the slot. Manual `collect` is mainly for recovery or non-queued runs.
- `.runs/` is local state, not a source of truth for competition submissions. Promote or copy the evidence you want to keep.

Initial manifest set:

- `baseline_1gpu_smoke.toml`
- `baseline_1gpu_fullshape.toml`
- `export_embedding_floor_sweep.toml`
- `export_mlp_bank_sweep.toml`
- `export_latek_island_sweep.toml`
- `eval_stride_sweep.toml`
- `timing_ladder.toml`
- `pair_feature_bigram_ladder.toml`
- `late_export_alignment_pilot.toml`
- `architecture_trade_suite.toml`
- `candidate_buffer_weighting_pilot.toml`

Superseded but kept for backward compatibility:

- `export_family_sweep.toml`

Initial tracked queue sets:

- `queue_sets/bringup_1gpu.toml`
- `queue_sets/checkpoint_calibration_1gpu.toml`
- `queue_sets/research_wave_1gpu.toml`
- `queue_sets/future_pilots_blocked.toml`

Use `python -m tools.experiments list-variants <manifest>` to inspect matrix-expanded variants before launching.
