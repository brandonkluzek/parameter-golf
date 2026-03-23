from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import REPO_ROOT
from .queue import enqueue_manifest

try:
    import tomllib
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Python 3.11+ is required for the experiment tooling") from exc


def _stringify_env_map(values: dict[str, object] | None) -> dict[str, str]:
    if not values:
        return {}
    return {str(key): str(value) for key, value in values.items()}


@dataclass(frozen=True)
class QueueSetJob:
    manifest: str
    variant: str | None
    variant_index: int | None
    priority: int
    count: int
    env_overrides: dict[str, str]


@dataclass(frozen=True)
class QueueSet:
    path: str
    name: str
    notes: str
    jobs: list[QueueSetJob]


def _load_queue_set(path: str | Path) -> dict[str, object]:
    queue_set_path = Path(path)
    with queue_set_path.open("rb") as f:
        raw = tomllib.load(f)
    if "name" not in raw:
        raise ValueError(f"Queue set {queue_set_path} is missing required key: name")
    if "job" not in raw or not isinstance(raw["job"], list) or not raw["job"]:
        raise ValueError(f"Queue set {queue_set_path} must contain at least one [[job]] entry")
    return raw


def _resolve_manifest_path(queue_set_path: Path, manifest_value: str) -> str:
    manifest_path = Path(manifest_value)
    if manifest_path.is_absolute():
        return str(manifest_path.resolve())
    candidate = (queue_set_path.parent / manifest_path).resolve()
    if candidate.exists():
        return str(candidate)
    return str((REPO_ROOT / manifest_path).resolve())


def expand_queue_set(path: str | Path, *, cli_env_overrides: dict[str, str] | None = None) -> QueueSet:
    queue_set_path = Path(path).resolve()
    raw = _load_queue_set(queue_set_path)
    default_priority = int(raw.get("priority", 100))
    default_count = int(raw.get("count", 1))
    default_env = _stringify_env_map(raw.get("set"))
    cli_env_overrides = cli_env_overrides or {}

    jobs: list[QueueSetJob] = []
    for entry in raw["job"]:
        if "manifest" not in entry:
            raise ValueError(f"Queue set {queue_set_path} has a [[job]] entry without manifest")
        env = dict(default_env)
        env.update(_stringify_env_map(entry.get("set")))
        env.update(cli_env_overrides)
        jobs.append(
            QueueSetJob(
                manifest=_resolve_manifest_path(queue_set_path, str(entry["manifest"])),
                variant=str(entry["variant"]) if entry.get("variant") is not None else None,
                variant_index=int(entry["variant_index"]) if entry.get("variant_index") is not None else None,
                priority=int(entry.get("priority", default_priority)),
                count=int(entry.get("count", default_count)),
                env_overrides=env,
            )
        )
    return QueueSet(
        path=str(queue_set_path),
        name=str(raw["name"]),
        notes=str(raw.get("notes", "")),
        jobs=jobs,
    )


def queue_set_to_dict(queue_set: QueueSet) -> dict[str, object]:
    return {
        "path": queue_set.path,
        "name": queue_set.name,
        "notes": queue_set.notes,
        "jobs": [
            {
                "manifest": job.manifest,
                "variant": job.variant,
                "variant_index": job.variant_index,
                "priority": job.priority,
                "count": job.count,
                "env_overrides": job.env_overrides,
            }
            for job in queue_set.jobs
        ],
    }


def enqueue_queue_set(path: str | Path, *, cli_env_overrides: dict[str, str] | None = None) -> dict[str, object]:
    queue_set = expand_queue_set(path, cli_env_overrides=cli_env_overrides)
    enqueued: list[dict[str, object]] = []
    for job in queue_set.jobs:
        enqueued.extend(
            enqueue_manifest(
                job.manifest,
                variant_name=job.variant,
                variant_index=job.variant_index,
                env_overrides=job.env_overrides,
                priority=job.priority,
                count=job.count,
            )
        )
    return {"queue_set": queue_set_to_dict(queue_set), "enqueued": enqueued}
