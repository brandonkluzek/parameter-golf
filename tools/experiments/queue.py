from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from . import QUEUE_DIR, QUEUE_EVENTS_PATH, QUEUE_HOSTS_DIR, QUEUE_RUNNER_LOG, QUEUE_STATE_PATH
from .estimator import estimate_manifest
from .launch import required_gpus_for_resource_class
from .manifest import available_variants, resolve_manifest, resolved_manifest_from_dict, ResolvedManifest

ACTIVE_JOB_STATES = {"launching", "running", "collecting"}


def _now_unix() -> float:
    return time.time()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def host_slug(host: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", host).strip("-").lower() or "host"


def queue_log(message: str) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    with QUEUE_RUNNER_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{_now_iso()} {message}\n")


def _empty_state() -> dict[str, object]:
    return {"jobs": {}, "hosts": {}, "meta": {"updated_at_iso": _now_iso(), "event_count": 0}}


def _write_host_state(host_state: dict[str, object]) -> None:
    QUEUE_HOSTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = str(host_state["host_slug"])
    (QUEUE_HOSTS_DIR / f"{slug}.json").write_text(json.dumps(host_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_state(state: dict[str, object]) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    state["meta"] = {**dict(state.get("meta", {})), "updated_at_iso": _now_iso()}
    QUEUE_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for host_state in dict(state.get("hosts", {})).values():
        _write_host_state(dict(host_state))


def append_queue_event(event_type: str, payload: dict[str, object]) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "type": event_type,
        "timestamp_unix": _now_unix(),
        "timestamp_iso": _now_iso(),
        **payload,
    }
    with QUEUE_EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def rebuild_queue_state() -> dict[str, object]:
    state = _empty_state()
    jobs = state["jobs"]
    hosts = state["hosts"]
    if QUEUE_EVENTS_PATH.exists():
        for line in QUEUE_EVENTS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            state["meta"]["event_count"] = int(state["meta"]["event_count"]) + 1
            event_type = event["type"]
            if event_type == "enqueue_job":
                job = dict(event["job"])
                jobs[str(job["job_id"])] = job
            elif event_type == "update_job":
                job_id = str(event["job_id"])
                if job_id in jobs:
                    jobs[job_id].update(dict(event.get("changes", {})))
    if QUEUE_HOSTS_DIR.exists():
        for host_file in sorted(QUEUE_HOSTS_DIR.glob("*.json")):
            host_state = json.loads(host_file.read_text(encoding="utf-8"))
            hosts[str(host_state["host_slug"])] = host_state
    _write_state(state)
    return state


def load_queue_state() -> dict[str, object]:
    if QUEUE_STATE_PATH.exists():
        return json.loads(QUEUE_STATE_PATH.read_text(encoding="utf-8"))
    return rebuild_queue_state()


def replace_host_state(host_state: dict[str, object]) -> dict[str, object]:
    _write_host_state(host_state)
    return rebuild_queue_state()


def update_job(job_id: str, changes: dict[str, object]) -> dict[str, object]:
    append_queue_event("update_job", {"job_id": job_id, "changes": changes})
    return rebuild_queue_state()


def _initial_job_state(resolved: ResolvedManifest, estimate: dict[str, object]) -> tuple[str, str]:
    if not bool(resolved.checks.get("script_exists")):
        return "blocked", "script_missing"
    if resolved.manual_block_reason:
        return "blocked", resolved.manual_block_reason
    for key in resolved.required_nonempty_env:
        if not str(resolved.env.get(key, "")).strip():
            return "blocked", f"required_env_missing:{key}"
    warning_flags = set(estimate.get("warning_flags", []))
    if "over_byte_cap" in warning_flags:
        return "skipped", "estimated_over_byte_cap"
    if "over_eval_budget" in warning_flags:
        return "skipped", "estimated_over_eval_budget"
    return "queued", ""


def _new_job_id() -> str:
    return f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _resolved_jobs_for_enqueue(
    manifest: str,
    *,
    variant_name: str | None,
    variant_index: int | None,
    all_variants: bool,
    env_overrides: dict[str, str],
) -> list[ResolvedManifest]:
    if all_variants:
        return [resolve_manifest(manifest, variant_name=name, env_overrides=env_overrides) for name in available_variants(manifest)]
    return [resolve_manifest(manifest, variant_name=variant_name, variant_index=variant_index, env_overrides=env_overrides)]


def enqueue_manifest(
    manifest: str,
    *,
    variant_name: str | None = None,
    variant_index: int | None = None,
    all_variants: bool = False,
    env_overrides: dict[str, str] | None = None,
    priority: int = 100,
    count: int = 1,
) -> list[dict[str, object]]:
    env_overrides = env_overrides or {}
    enqueued: list[dict[str, object]] = []
    for resolved in _resolved_jobs_for_enqueue(
        manifest,
        variant_name=variant_name,
        variant_index=variant_index,
        all_variants=all_variants,
        env_overrides=env_overrides,
    ):
        estimate = estimate_manifest(resolved).values
        for _ in range(count):
            state, reason = _initial_job_state(resolved, estimate)
            job = {
                "job_id": _new_job_id(),
                "enqueue_time_unix": _now_unix(),
                "enqueue_time_iso": _now_iso(),
                "manifest_path": resolved.manifest_path,
                "resolved_variant_name": resolved.variant_name,
                "env_overrides": env_overrides,
                "priority": int(priority),
                "required_gpus": required_gpus_for_resource_class(resolved.resource_class),
                "state": state,
                "state_reason": reason,
                "run_id": "",
                "assigned_gpu_ids": [],
                "host_slug": "",
                "estimate": estimate,
                "warning_flags": list(estimate.get("warning_flags", [])),
                "resolved_manifest": resolved.to_dict(),
            }
            append_queue_event("enqueue_job", {"job": job})
            enqueued.append(job)
    rebuild_queue_state()
    return enqueued


def cancel_job(job_id: str) -> dict[str, object]:
    return update_job(job_id, {"state": "cancelled", "state_reason": "cancelled"})


def queued_jobs(state: dict[str, object]) -> list[dict[str, object]]:
    jobs = [dict(job) for job in dict(state.get("jobs", {})).values() if job.get("state") == "queued"]
    return sorted(jobs, key=lambda job: (-int(job.get("priority", 0)), float(job.get("enqueue_time_unix", 0.0)), str(job.get("job_id", ""))))


def active_jobs_for_host(state: dict[str, object], host_name_slug: str) -> list[dict[str, object]]:
    jobs = [
        dict(job)
        for job in dict(state.get("jobs", {})).values()
        if job.get("host_slug") == host_name_slug and job.get("state") in ACTIVE_JOB_STATES
    ]
    return sorted(jobs, key=lambda job: str(job["job_id"]))


def free_gpu_slots(state: dict[str, object], host_name_slug: str, capacity_gpus: int) -> list[int]:
    used: set[int] = set()
    for job in active_jobs_for_host(state, host_name_slug):
        for slot in job.get("assigned_gpu_ids", []):
            used.add(int(slot))
    return [slot for slot in range(capacity_gpus) if slot not in used]


def is_host_draining(state: dict[str, object], capacity_gpus: int) -> bool:
    if capacity_gpus != 8:
        return False
    queue = queued_jobs(state)
    return bool(queue and int(queue[0].get("required_gpus", 1)) == 8)


def allocate_gpu_slots(free_slots: list[int], required_gpus: int) -> list[int] | None:
    if required_gpus == 1 and free_slots:
        return [free_slots[0]]
    if required_gpus == 8 and len(free_slots) == 8:
        return free_slots[:8]
    return None


def should_fail_overrun(heartbeat: dict[str, object] | None, predicted_step_ms: float) -> bool:
    if not heartbeat or predicted_step_ms <= 0:
        return False
    step = int(heartbeat.get("step") or 0)
    elapsed_ms = float(heartbeat.get("elapsed_train_ms") or 0.0)
    step_avg_ms = heartbeat.get("step_avg_ms")
    if step_avg_ms is None:
        return False
    if step < 50 and elapsed_ms < 60000.0:
        return False
    return float(step_avg_ms) > predicted_step_ms * 1.35


def classify_remote_snapshot(
    *,
    heartbeat: dict[str, object] | None,
    summary: dict[str, object] | None,
    tmux_alive: bool,
    predicted_step_ms: float,
) -> tuple[str, str]:
    if summary and str(summary.get("status")) == "completed":
        return "completed", "remote_summary_completed"
    if should_fail_overrun(heartbeat, predicted_step_ms):
        return "failed_overrun", "observed_step_avg_exceeded_threshold"
    if tmux_alive or heartbeat:
        return "running", ""
    return "failed", "remote_session_missing_without_summary"


def resolved_manifest_for_job(job: dict[str, object]) -> ResolvedManifest:
    return resolved_manifest_from_dict(dict(job["resolved_manifest"]))


def host_state_payload(
    *,
    host: str,
    capacity_gpus: int,
    state: dict[str, object],
) -> dict[str, object]:
    slug = host_slug(host)
    active = active_jobs_for_host(state, slug)
    free = free_gpu_slots(state, slug, capacity_gpus)
    return {
        "host": host,
        "host_slug": slug,
        "capacity_gpus": capacity_gpus,
        "free_gpu_slots": free,
        "active_jobs": [job["job_id"] for job in active],
        "draining": is_host_draining(state, capacity_gpus),
        "updated_at_iso": _now_iso(),
    }


def queue_eta_seconds(state: dict[str, object], host_name_slug: str, capacity_gpus: int) -> float:
    total = 0.0
    for job in active_jobs_for_host(state, host_name_slug):
        heartbeat = job.get("last_heartbeat") or {}
        if heartbeat and heartbeat.get("elapsed_train_ms") is not None:
            manifest = resolved_manifest_for_job(job)
            max_wallclock = float(manifest.env.get("MAX_WALLCLOCK_SECONDS", 600) or 600)
            total += max(0.0, max_wallclock - float(heartbeat["elapsed_train_ms"]) / 1000.0)
    queue = queued_jobs(state)
    if capacity_gpus == 8:
        total += sum(
            float(resolved_manifest_for_job(job).env.get("MAX_WALLCLOCK_SECONDS", 600) or 600)
            for job in queue
            if int(job.get("required_gpus", 1)) == 8
        )
        one_gpu_jobs = [job for job in queue if int(job.get("required_gpus", 1)) == 1]
        if one_gpu_jobs:
            total += sum(float(resolved_manifest_for_job(job).env.get("MAX_WALLCLOCK_SECONDS", 600) or 600) for job in one_gpu_jobs) / 8.0
    else:
        total += sum(float(resolved_manifest_for_job(job).env.get("MAX_WALLCLOCK_SECONDS", 600) or 600) for job in queue)
    return total
