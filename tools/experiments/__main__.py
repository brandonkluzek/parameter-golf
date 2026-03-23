from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .estimator import estimate_manifest
from .launch import (
    bootstrap_runpod_host,
    collect_remote_run,
    ensure_clean_git,
    git_origin_url,
    git_sha,
    kill_tmux_session,
    launch_remote_run,
    remote_file_text,
    required_gpus_for_resource_class,
    run_dir_for,
    stage_local_run,
    tmux_session_exists,
)
from .manifest import available_variants, resolve_manifest
from .promotion import promote_run
from .queue import (
    ACTIVE_JOB_STATES,
    active_jobs_for_host,
    allocate_gpu_slots,
    cancel_job,
    classify_remote_snapshot,
    enqueue_manifest,
    free_gpu_slots,
    host_slug,
    host_state_payload,
    is_host_draining,
    load_queue_state,
    queue_eta_seconds,
    queue_log,
    queued_jobs,
    rebuild_queue_state,
    replace_host_state,
    resolved_manifest_for_job,
    update_job,
)
from .queue_sets import enqueue_queue_set, expand_queue_set, queue_set_to_dict
from .registry import rebuild_index
from .remote import SSHConfig
from .textlog import parse_train_log_text


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_set_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _build_ssh_config(args: argparse.Namespace) -> SSHConfig:
    return SSHConfig(
        host=args.host,
        user=getattr(args, "user", None),
        port=getattr(args, "port", None),
        identity=getattr(args, "identity", None),
        options=getattr(args, "ssh_option", None) or [],
    )


def _load_remote_metadata(run_id: str) -> dict[str, object]:
    run_dir = run_dir_for(run_id)
    return json.loads((run_dir / "remote.json").read_text(encoding="utf-8"))


def _remote_ssh_from_metadata(remote: dict[str, object]) -> SSHConfig:
    return SSHConfig(
        host=str(remote["host"]),
        user=remote.get("user") or None,
        port=remote.get("port"),
        identity=remote.get("identity") or None,
    )


def _write_remote_metadata(run_id: str, metadata: dict[str, object]) -> None:
    run_dir = run_dir_for(run_id)
    (run_dir / "remote.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _float_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _progress_fields(
    *,
    step: object,
    iterations: object,
    elapsed_train_ms: object,
    step_avg_ms: object,
    max_wallclock_s: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    step_value = _float_value(step)
    iterations_value = _float_value(iterations)
    elapsed_value = _float_value(elapsed_train_ms)
    step_avg_value = _float_value(step_avg_ms)
    if step_value is not None and iterations_value is not None and iterations_value > 0:
        progress_percent = max(0.0, min(step_value / iterations_value, 1.0)) * 100.0
        payload["progress_percent"] = round(progress_percent, 3)
    if step_value is not None and iterations_value is not None and step_avg_value is not None and step_value > 0:
        remaining_steps = max(iterations_value - step_value, 0.0)
        payload["eta_to_iteration_finish_ms"] = round(remaining_steps * step_avg_value, 3)
    if elapsed_value is not None and max_wallclock_s is not None and max_wallclock_s > 0:
        payload["eta_to_wallclock_stop_ms"] = round(max(0.0, max_wallclock_s * 1000.0 - elapsed_value), 3)
    return payload


def _status_payload(run_dir: Path, heartbeat: dict[str, object]) -> dict[str, object]:
    payload = {"run_id": heartbeat.get("run_id", run_dir.name), "phase": heartbeat.get("phase")}
    payload["timestamp_iso"] = heartbeat.get("timestamp_iso")
    payload["step"] = heartbeat.get("step")
    payload["iterations"] = heartbeat.get("iterations")
    payload["elapsed_train_ms"] = heartbeat.get("elapsed_train_ms")
    payload["step_avg_ms"] = heartbeat.get("step_avg_ms")
    payload["last_event"] = heartbeat.get("last_event")
    max_wallclock_s: float | None = None
    manifest_path = run_dir / "manifest.resolved.toml"
    estimate_path = run_dir / "estimate.json"
    if manifest_path.exists():
        resolved = resolve_manifest(manifest_path)
        max_wallclock_s = float(resolved.env.get("MAX_WALLCLOCK_SECONDS", 600))
        payload["manifest"] = {
            "variant_name": resolved.variant_name,
            "resource_class": resolved.resource_class,
            "script_path": resolved.script_path,
        }
    payload.update(
        _progress_fields(
            step=heartbeat.get("step"),
            iterations=heartbeat.get("iterations"),
            elapsed_train_ms=heartbeat.get("elapsed_train_ms"),
            step_avg_ms=heartbeat.get("step_avg_ms"),
            max_wallclock_s=max_wallclock_s,
        )
    )
    if estimate_path.exists() and heartbeat.get("step_avg_ms") is not None:
        estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
        predicted_step_ms = estimate["predicted_timing"]["step_ms"]
        payload["estimator"] = {
            "predicted_step_ms": predicted_step_ms,
            "observed_minus_predicted_step_ms": round(float(heartbeat["step_avg_ms"]) - float(predicted_step_ms), 3),
            "predicted_total_bytes": estimate["predicted_bytes"]["total_bytes"],
        }
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["final_metrics"] = summary.get("metrics", {})
        payload["artifacts"] = summary.get("artifacts", {})
    return payload


def _resolve_for_cli(args: argparse.Namespace):
    return resolve_manifest(
        args.manifest,
        variant_name=getattr(args, "variant", None),
        variant_index=getattr(args, "variant_index", None),
        env_overrides=_parse_set_overrides(getattr(args, "set", None) or []),
    )


def _launch_manual_run(args: argparse.Namespace) -> dict[str, object]:
    ensure_clean_git()
    git_sha_value = git_sha()
    repo_url = args.repo_url or git_origin_url()
    resolved = _resolve_for_cli(args)
    estimate = estimate_manifest(resolved).values
    run_id = f"{resolved.variant_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    stage_local_run(run_id, resolved, estimate)
    metadata = launch_remote_run(
        resolved,
        ssh_config=_build_ssh_config(args),
        git_sha_value=git_sha_value,
        repo_url=repo_url,
        remote_root=args.remote_root,
        run_id=run_id,
    )
    _write_remote_metadata(run_id, metadata)
    return {"run_id": run_id, "session_name": metadata["session_name"], "remote_run_dir": metadata["remote_run_dir"]}


def _remote_snapshot(remote: dict[str, object]) -> dict[str, object]:
    ssh_config = _remote_ssh_from_metadata(remote)
    remote_run_dir = str(remote["remote_run_dir"])
    heartbeat_text = remote_file_text(ssh_config, f"{remote_run_dir}/heartbeat.json")
    summary_text = remote_file_text(ssh_config, f"{remote_run_dir}/summary.json")
    heartbeat = json.loads(heartbeat_text) if heartbeat_text else None
    summary = json.loads(summary_text) if summary_text else None
    tmux_alive = tmux_session_exists(ssh_config, str(remote["session_name"]))
    return {"heartbeat": heartbeat, "summary": summary, "tmux_alive": tmux_alive}


def _queue_job_brief(job: dict[str, object]) -> dict[str, object]:
    payload = {
        "job_id": job["job_id"],
        "variant": job["resolved_variant_name"],
        "priority": job["priority"],
        "required_gpus": job["required_gpus"],
        "state": job["state"],
        "reason": job.get("state_reason", ""),
        "run_id": job.get("run_id", ""),
        "assigned_gpu_ids": job.get("assigned_gpu_ids", []),
        "predicted_total_bytes": job.get("estimate", {}).get("predicted_bytes", {}).get("total_bytes"),
        "predicted_step_ms": job.get("estimate", {}).get("predicted_timing", {}).get("step_ms"),
    }
    heartbeat = dict(job.get("last_heartbeat") or {})
    if heartbeat:
        payload["phase"] = heartbeat.get("phase")
        payload["step"] = heartbeat.get("step")
        payload["iterations"] = heartbeat.get("iterations")
        payload["elapsed_train_ms"] = heartbeat.get("elapsed_train_ms")
        payload["step_avg_ms"] = heartbeat.get("step_avg_ms")
        payload.update(
            _progress_fields(
                step=heartbeat.get("step"),
                iterations=heartbeat.get("iterations"),
                elapsed_train_ms=heartbeat.get("elapsed_train_ms"),
                step_avg_ms=heartbeat.get("step_avg_ms"),
                max_wallclock_s=float(resolved_manifest_for_job(job).env.get("MAX_WALLCLOCK_SECONDS", 600) or 600),
            )
        )
    return payload


def _queue_status_payload(host: str | None = None) -> dict[str, object]:
    state = rebuild_queue_state()
    jobs = [dict(job) for job in dict(state.get("jobs", {})).values()]
    hosts = dict(state.get("hosts", {}))
    payload: dict[str, object] = {
        "meta": state.get("meta", {}),
        "hosts": list(hosts.values()) if host is None else [hosts.get(host_slug(host), {})],
    }
    if host is not None:
        slug = host_slug(host)
        payload["eta_seconds"] = queue_eta_seconds(state, slug, int(dict(hosts.get(slug, {})).get("capacity_gpus", 1) or 1))
    payload["running"] = [_queue_job_brief(job) for job in jobs if job.get("state") in ACTIVE_JOB_STATES]
    payload["queued"] = [_queue_job_brief(job) for job in queued_jobs(state)]
    payload["blocked"] = [_queue_job_brief(job) for job in jobs if job.get("state") == "blocked"]
    payload["skipped"] = [_queue_job_brief(job) for job in jobs if job.get("state") == "skipped"]
    payload["failed"] = [_queue_job_brief(job) for job in jobs if str(job.get("state", "")).startswith("failed")]
    payload["completed"] = [_queue_job_brief(job) for job in jobs if job.get("state") == "completed"]
    payload["cancelled"] = [_queue_job_brief(job) for job in jobs if job.get("state") == "cancelled"]
    return payload


def _launch_queue_job(
    job: dict[str, object],
    *,
    ssh_config: SSHConfig,
    git_sha_value: str,
    repo_url: str,
    remote_root: str,
    assigned_gpu_ids: list[int],
    host_name_slug: str,
) -> None:
    resolved = resolved_manifest_for_job(job)
    run_id = f"{resolved.variant_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    update_job(
        str(job["job_id"]),
        {
            "state": "launching",
            "state_reason": "launching",
            "run_id": run_id,
            "assigned_gpu_ids": assigned_gpu_ids,
            "host_slug": host_name_slug,
        },
    )
    try:
        stage_local_run(run_id, resolved, dict(job["estimate"]))
        metadata = launch_remote_run(
            resolved,
            ssh_config=ssh_config,
            git_sha_value=git_sha_value,
            repo_url=repo_url,
            remote_root=remote_root,
            run_id=run_id,
            assigned_gpu_ids=assigned_gpu_ids,
            queue_job_id=str(job["job_id"]),
        )
        _write_remote_metadata(run_id, metadata)
        update_job(
            str(job["job_id"]),
            {
                "state": "running",
                "state_reason": "",
                "remote": metadata,
                "run_id": run_id,
                "assigned_gpu_ids": assigned_gpu_ids,
                "host_slug": host_name_slug,
            },
        )
        queue_log(f"launched job={job['job_id']} run_id={run_id} slots={assigned_gpu_ids}")
    except Exception as exc:
        update_job(
            str(job["job_id"]),
            {
                "state": "failed",
                "state_reason": f"launch_error:{exc}",
                "run_id": run_id,
                "assigned_gpu_ids": [],
            },
        )
        queue_log(f"launch_failed job={job['job_id']} error={exc}")


def _reconcile_active_job(job: dict[str, object]) -> None:
    remote = dict(job.get("remote") or {})
    if not remote:
        update_job(str(job["job_id"]), {"state": "failed", "state_reason": "missing_remote_metadata"})
        return
    try:
        snapshot = _remote_snapshot(remote)
        heartbeat = snapshot["heartbeat"]
        summary = snapshot["summary"]
        predicted_step_ms = float(job.get("estimate", {}).get("predicted_timing", {}).get("step_ms") or 0.0)
        new_state, reason = classify_remote_snapshot(
            heartbeat=heartbeat,
            summary=summary,
            tmux_alive=bool(snapshot["tmux_alive"]),
            predicted_step_ms=predicted_step_ms,
        )
        if new_state == "failed_overrun":
            kill_tmux_session(_remote_ssh_from_metadata(remote), str(remote["session_name"]))
            update_job(
                str(job["job_id"]),
                {"state": "failed_overrun", "state_reason": reason, "last_heartbeat": heartbeat or {}, "last_summary": summary or {}},
            )
            queue_log(f"failed_overrun job={job['job_id']} run_id={job.get('run_id', '')}")
            return
        if new_state == "completed":
            update_job(str(job["job_id"]), {"state": "collecting", "state_reason": "collecting", "last_summary": summary or {}})
            run_id = str(job["run_id"])
            local_run_dir = run_dir_for(run_id)
            collect_remote_run(remote, local_run_dir)
            rebuild_index()
            update_job(
                str(job["job_id"]),
                {"state": "completed", "state_reason": reason, "last_heartbeat": heartbeat or {}, "last_summary": summary or {}},
            )
            queue_log(f"completed job={job['job_id']} run_id={run_id}")
            return
        if new_state == "failed":
            update_job(
                str(job["job_id"]),
                {"state": "failed", "state_reason": reason, "last_heartbeat": heartbeat or {}, "last_summary": summary or {}},
            )
            queue_log(f"failed job={job['job_id']} run_id={job.get('run_id', '')} reason={reason}")
            return
        update_job(
            str(job["job_id"]),
            {"state": "running", "state_reason": reason, "last_heartbeat": heartbeat or {}, "last_summary": summary or {}},
        )
    except Exception as exc:
        update_job(str(job["job_id"]), {"state": "failed", "state_reason": f"monitor_error:{exc}"})
        queue_log(f"monitor_error job={job['job_id']} error={exc}")


def command_prepare(args: argparse.Namespace) -> int:
    resolved = _resolve_for_cli(args)
    estimate = estimate_manifest(resolved).values
    _print_json({"resolved_manifest": resolved.to_dict(), "estimate": estimate})
    return 0


def command_list_variants(args: argparse.Namespace) -> int:
    _print_json({"manifest": str(Path(args.manifest).resolve()), "variants": available_variants(args.manifest)})
    return 0


def command_bootstrap_runpod(args: argparse.Namespace) -> int:
    ensure_clean_git()
    summary = bootstrap_runpod_host(
        _build_ssh_config(args),
        git_sha_value=git_sha(),
        repo_url=args.repo_url or git_origin_url(),
        remote_root=args.remote_root,
        dataset_variant=args.dataset_variant,
        train_shards=args.train_shards,
    )
    _print_json(summary)
    return 0


def command_run_remote(args: argparse.Namespace) -> int:
    payload = _launch_manual_run(args)
    _print_json(payload)
    return 0


def command_status(args: argparse.Namespace) -> int:
    run_dir = run_dir_for(args.run_id)
    heartbeat_path = run_dir / "heartbeat.json"
    train_log_path = run_dir / "train.log"
    if heartbeat_path.exists():
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        _print_json(_status_payload(run_dir, heartbeat))
        return 0
    if (run_dir / "remote.json").exists():
        remote = _load_remote_metadata(args.run_id)
        snapshot = _remote_snapshot(remote)
        if snapshot["heartbeat"]:
            _print_json(_status_payload(run_dir, dict(snapshot["heartbeat"])))
            return 0
        remote_log = remote_file_text(_remote_ssh_from_metadata(remote), f"{remote['remote_run_dir']}/train.log")
        if remote_log:
            _print_json(parse_train_log_text(remote_log))
            return 0
        if train_log_path.exists():
            _print_json(parse_train_log_text(train_log_path.read_text(encoding="utf-8")))
            return 0
    if train_log_path.exists():
        _print_json(parse_train_log_text(train_log_path.read_text(encoding="utf-8")))
        return 0
    raise FileNotFoundError(f"No heartbeat or train log available for run {args.run_id}")


def command_collect(args: argparse.Namespace) -> int:
    remote = _load_remote_metadata(args.run_id)
    collect_remote_run(remote, run_dir_for(args.run_id))
    if (run_dir_for(args.run_id) / "summary.json").exists():
        rebuild_index()
    return 0


def command_promote(args: argparse.Namespace) -> int:
    target = promote_run(run_dir_for(args.run_id), destination=Path(args.destination).resolve() if args.destination else None)
    print(str(target))
    return 0


def command_enqueue(args: argparse.Namespace) -> int:
    jobs = enqueue_manifest(
        args.manifest,
        variant_name=args.variant,
        variant_index=args.variant_index,
        all_variants=args.all_variants,
        env_overrides=_parse_set_overrides(args.set or []),
        priority=args.priority,
        count=args.count,
    )
    _print_json({"enqueued": [_queue_job_brief(job) for job in jobs]})
    return 0


def command_show_set(args: argparse.Namespace) -> int:
    queue_set = expand_queue_set(args.queue_set, cli_env_overrides=_parse_set_overrides(args.set or []))
    _print_json(queue_set_to_dict(queue_set))
    return 0


def command_enqueue_set(args: argparse.Namespace) -> int:
    result = enqueue_queue_set(args.queue_set, cli_env_overrides=_parse_set_overrides(args.set or []))
    _print_json(
        {
            "queue_set": result["queue_set"],
            "enqueued": [_queue_job_brief(job) for job in result["enqueued"]],
        }
    )
    return 0


def command_queue_cancel(args: argparse.Namespace) -> int:
    state = rebuild_queue_state()
    job = dict(dict(state.get("jobs", {})).get(args.job_id, {}))
    if not job:
        raise ValueError(f"Unknown job_id: {args.job_id}")
    if job.get("state") in ACTIVE_JOB_STATES and job.get("remote"):
        remote = dict(job["remote"])
        kill_tmux_session(_remote_ssh_from_metadata(remote), str(remote["session_name"]))
    cancel_job(args.job_id)
    _print_json({"cancelled": args.job_id})
    return 0


def command_queue_status(args: argparse.Namespace) -> int:
    while True:
        _print_json(_queue_status_payload(args.host))
        if not args.watch:
            return 0
        time.sleep(args.poll_seconds)


def command_queue_runner(args: argparse.Namespace) -> int:
    ensure_clean_git()
    git_sha_value = git_sha()
    repo_url = args.repo_url or git_origin_url()
    ssh_config = _build_ssh_config(args)
    summary = bootstrap_runpod_host(
        ssh_config,
        git_sha_value=git_sha_value,
        repo_url=repo_url,
        remote_root=args.remote_root,
        dataset_variant=args.dataset_variant,
        train_shards=args.train_shards,
    )
    host_name_slug = host_slug(ssh_config.host)
    queue_log(
        "runner_started "
        f"host={ssh_config.host} capacity={args.capacity_gpus} "
        f"train_shards={summary.get('dataset', {}).get('current_train_shards', '')}"
    )
    try:
        while True:
            state = rebuild_queue_state()
            for job in active_jobs_for_host(state, host_name_slug):
                _reconcile_active_job(job)
                state = rebuild_queue_state()

            host_state = host_state_payload(host=ssh_config.host, capacity_gpus=args.capacity_gpus, state=state)
            replace_host_state(host_state)
            state = rebuild_queue_state()

            draining = is_host_draining(state, args.capacity_gpus)
            free_slots = free_gpu_slots(state, host_name_slug, args.capacity_gpus)
            queue = queued_jobs(state)

            if draining and args.capacity_gpus == 8:
                if len(free_slots) == 8:
                    target = next((job for job in queue if int(job.get("required_gpus", 1)) == 8), None)
                    if target is not None:
                        _launch_queue_job(
                            target,
                            ssh_config=ssh_config,
                            git_sha_value=git_sha_value,
                            repo_url=repo_url,
                            remote_root=args.remote_root,
                            assigned_gpu_ids=list(range(8)),
                            host_name_slug=host_name_slug,
                        )
                time.sleep(args.poll_seconds)
                continue

            launched = False
            while free_slots:
                target = next(
                    (
                        job
                        for job in queue
                        if int(job.get("required_gpus", 1)) == 1
                        and allocate_gpu_slots(free_slots, int(job.get("required_gpus", 1))) is not None
                    ),
                    None,
                )
                if args.capacity_gpus == 1:
                    target = next((job for job in queue if int(job.get("required_gpus", 1)) == 1), None)
                if target is None:
                    break
                assigned_gpu_ids = allocate_gpu_slots(free_slots, int(target.get("required_gpus", 1)))
                if assigned_gpu_ids is None:
                    break
                _launch_queue_job(
                    target,
                    ssh_config=ssh_config,
                    git_sha_value=git_sha_value,
                    repo_url=repo_url,
                    remote_root=args.remote_root,
                    assigned_gpu_ids=assigned_gpu_ids,
                    host_name_slug=host_name_slug,
                )
                launched = True
                state = rebuild_queue_state()
                queue = queued_jobs(state)
                free_slots = free_gpu_slots(state, host_name_slug, args.capacity_gpus)

            host_state = host_state_payload(host=ssh_config.host, capacity_gpus=args.capacity_gpus, state=state)
            replace_host_state(host_state)
            if not launched:
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        queue_log(f"runner_stopped host={ssh_config.host}")
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("manifest")
    prepare.add_argument("--variant")
    prepare.add_argument("--variant-index", type=int)
    prepare.add_argument("--set", action="append")
    prepare.set_defaults(func=command_prepare)

    list_variants = sub.add_parser("list-variants")
    list_variants.add_argument("manifest")
    list_variants.set_defaults(func=command_list_variants)

    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("manifest")
    enqueue.add_argument("--variant")
    enqueue.add_argument("--variant-index", type=int)
    enqueue.add_argument("--all-variants", action="store_true")
    enqueue.add_argument("--set", action="append")
    enqueue.add_argument("--priority", type=int, default=100)
    enqueue.add_argument("--count", type=int, default=1)
    enqueue.set_defaults(func=command_enqueue)

    show_set = sub.add_parser("show-set")
    show_set.add_argument("queue_set")
    show_set.add_argument("--set", action="append")
    show_set.set_defaults(func=command_show_set)

    enqueue_set = sub.add_parser("enqueue-set")
    enqueue_set.add_argument("queue_set")
    enqueue_set.add_argument("--set", action="append")
    enqueue_set.set_defaults(func=command_enqueue_set)

    for name, handler in (
        ("bootstrap-runpod", command_bootstrap_runpod),
        ("run-remote", command_run_remote),
        ("queue-runner", command_queue_runner),
    ):
        cmd = sub.add_parser(name)
        if name == "run-remote":
            cmd.add_argument("manifest")
            cmd.add_argument("--variant")
            cmd.add_argument("--variant-index", type=int)
            cmd.add_argument("--set", action="append")
        cmd.add_argument("--host", required=True)
        cmd.add_argument("--user")
        cmd.add_argument("--port", type=int)
        cmd.add_argument("--identity")
        cmd.add_argument("--ssh-option", action="append")
        cmd.add_argument("--remote-root", default="/workspace/parameter-golf")
        cmd.add_argument("--repo-url")
        if name == "bootstrap-runpod":
            cmd.add_argument("--dataset-variant", default="sp1024")
            cmd.add_argument("--train-shards", type=int, default=1)
        if name == "queue-runner":
            cmd.add_argument("--capacity-gpus", type=int, choices=[1, 8], required=True)
            cmd.add_argument("--poll-seconds", type=float, default=15.0)
            cmd.add_argument("--dataset-variant", default="sp1024")
            cmd.add_argument("--train-shards", type=int, default=1)
        cmd.set_defaults(func=handler)

    status = sub.add_parser("status")
    status.add_argument("run_id")
    status.set_defaults(func=command_status)

    queue_status = sub.add_parser("queue-status")
    queue_status.add_argument("--host")
    queue_status.add_argument("--watch", action="store_true")
    queue_status.add_argument("--poll-seconds", type=float, default=15.0)
    queue_status.set_defaults(func=command_queue_status)

    collect = sub.add_parser("collect")
    collect.add_argument("run_id")
    collect.set_defaults(func=command_collect)

    promote = sub.add_parser("promote")
    promote.add_argument("run_id")
    promote.add_argument("--destination")
    promote.set_defaults(func=command_promote)

    queue_cancel = sub.add_parser("queue-cancel")
    queue_cancel.add_argument("job_id")
    queue_cancel.set_defaults(func=command_queue_cancel)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
