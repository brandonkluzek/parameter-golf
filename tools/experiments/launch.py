from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any
from pathlib import Path

from . import REPO_ROOT, RUNS_DIR
from .manifest import ResolvedManifest, write_resolved_manifest
from .remote import SSHConfig, run_ssh, scp_from


def git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to get git SHA")
    return result.stdout.strip()


def git_origin_url() -> str:
    result = subprocess.run(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to get git origin URL")
    return result.stdout.strip()


def ensure_clean_git() -> None:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to inspect git status")
    if result.stdout.strip():
        raise RuntimeError("Git working tree is dirty. Commit and push before git-based remote deployment.")


def required_gpus_for_resource_class(resource_class: str) -> int:
    if "8x" in resource_class or "8xh100" in resource_class:
        return 8
    return 1


def run_dir_for(run_id: str) -> Path:
    return RUNS_DIR / run_id


def copy_source_snapshot(run_dir: Path, script_path: Path) -> None:
    source_dir = run_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_path, source_dir / script_path.name)


def stage_local_run(run_id: str, resolved: ResolvedManifest, estimate: dict[str, object]) -> Path:
    local_run_dir = run_dir_for(run_id)
    local_run_dir.mkdir(parents=True, exist_ok=False)
    write_resolved_manifest(local_run_dir / "manifest.resolved.toml", resolved)
    copy_source_snapshot(local_run_dir, Path(resolved.script_path))
    (local_run_dir / "estimate.json").write_text(json.dumps(estimate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return local_run_dir


def needs_dataset_sync(current_train_shards: int, requested_train_shards: int, tokenizer_present: bool) -> bool:
    return current_train_shards < requested_train_shards or not tokenizer_present


def parse_bootstrap_summary_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "dataset" in payload and "git" in payload:
            return payload
    raise ValueError("bootstrap-runpod did not return a JSON summary")


def bootstrap_runpod_host(
    ssh_config: SSHConfig,
    *,
    git_sha_value: str,
    repo_url: str,
    remote_root: str,
    dataset_variant: str = "sp1024",
    train_shards: int = 1,
) -> dict[str, Any]:
    remote_root = remote_root.rstrip("/")
    quoted_remote_root = json.dumps(remote_root)
    quoted_repo_url = json.dumps(repo_url)
    quoted_git_sha = json.dumps(git_sha_value)
    quoted_variant = json.dumps(dataset_variant)
    remote_cmd = f"""
set -euo pipefail
REMOTE_ROOT={quoted_remote_root}
REPO_URL={quoted_repo_url}
REQUESTED_SHA={quoted_git_sha}
DATASET_VARIANT={quoted_variant}
REQUESTED_TRAIN_SHARDS={int(train_shards)}
mkdir -p "$(dirname "$REMOTE_ROOT")"
BOOTSTRAP_GIT_SHA_BEFORE=""
if [ ! -d "$REMOTE_ROOT/.git" ]; then
  rm -rf "$REMOTE_ROOT"
  git clone --quiet "$REPO_URL" "$REMOTE_ROOT"
else
  BOOTSTRAP_GIT_SHA_BEFORE="$(git -C "$REMOTE_ROOT" rev-parse HEAD || true)"
fi
git -C "$REMOTE_ROOT" fetch --all --tags --quiet
git -C "$REMOTE_ROOT" checkout --force --quiet "$REQUESTED_SHA"
cd "$REMOTE_ROOT"
export BOOTSTRAP_GIT_SHA_BEFORE REPO_URL REQUESTED_SHA DATASET_VARIANT REQUESTED_TRAIN_SHARDS REMOTE_ROOT
python3 - <<'PY'
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

from data.cached_challenge_fineweb import (
    REMOTE_ROOT_PREFIX,
    artifact_paths_for_tokenizer,
    dataset_dir_for_variant,
    get,
    load_manifest,
    local_path_for_remote,
)

remote_root = Path(os.environ["REMOTE_ROOT"])
requested_sha = os.environ["REQUESTED_SHA"]
dataset_variant = os.environ["DATASET_VARIANT"]
requested_train_shards = int(os.environ["REQUESTED_TRAIN_SHARDS"])
before_sha = os.environ.get("BOOTSTRAP_GIT_SHA_BEFORE", "")

manifest = load_manifest(skip_manifest_download=False)
dataset_dir_name = dataset_dir_for_variant(dataset_variant)
dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir_name), None)
if dataset_entry is None:
    raise ValueError(f"dataset {{dataset_dir_name}} not found in manifest")
tokenizer_name = dataset_entry.get("tokenizer_name")
tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
if tokenizer_entry is None:
    raise ValueError(f"tokenizer {{tokenizer_name}} not found in manifest")

dataset_dir = local_path_for_remote(f"{{REMOTE_ROOT_PREFIX}}/datasets/{{dataset_dir_name}}")
tokenizer_paths = [local_path_for_remote(f"{{REMOTE_ROOT_PREFIX}}/{{path}}") for path in artifact_paths_for_tokenizer(tokenizer_entry)]
train_shards_before = len(list(dataset_dir.glob("fineweb_train_*.bin"))) if dataset_dir.exists() else 0
tokenizer_present_before = all(path.exists() for path in tokenizer_paths)
dataset_updated = train_shards_before < requested_train_shards or not tokenizer_present_before
if dataset_updated:
    val_shards = int((dataset_entry.get("stats") or {{}}).get("files_val", 0))
    dataset_prefix = f"{{REMOTE_ROOT_PREFIX}}/datasets/{{dataset_dir_name}}"
    for i in range(val_shards):
        get(f"{{dataset_prefix}}/fineweb_val_{{i:06d}}.bin")
    for i in range(requested_train_shards):
        get(f"{{dataset_prefix}}/fineweb_train_{{i:06d}}.bin")
    for artifact_path in artifact_paths_for_tokenizer(tokenizer_entry):
        get(f"{{REMOTE_ROOT_PREFIX}}/{{artifact_path}}")

train_shards_after = len(list(dataset_dir.glob("fineweb_train_*.bin"))) if dataset_dir.exists() else 0
tokenizer_present_after = all(path.exists() for path in tokenizer_paths)
after_sha = subprocess.run(
    ["git", "-C", str(remote_root), "rev-parse", "HEAD"],
    text=True,
    capture_output=True,
    check=True,
).stdout.strip()

gpu_names = []
if shutil.which("nvidia-smi"):
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if gpu_result.returncode == 0:
        gpu_names = [line.strip() for line in gpu_result.stdout.splitlines() if line.strip()]

import torch

workspace_usage = shutil.disk_usage("/workspace")
summary = {{
    "host": {{
        "hostname": socket.gethostname(),
        "remote_root": str(remote_root),
        "python3_path": shutil.which("python3") or "",
        "torchrun_path": shutil.which("torchrun") or "",
        "nvidia_smi_path": shutil.which("nvidia-smi") or "",
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()),
        "gpu_names": gpu_names,
        "workspace_bytes_total": int(workspace_usage.total),
        "workspace_bytes_used": int(workspace_usage.used),
        "workspace_bytes_free": int(workspace_usage.free),
    }},
    "git": {{
        "repo_url": os.environ["REPO_URL"],
        "requested_sha": requested_sha,
        "before_sha": before_sha,
        "after_sha": after_sha,
    }},
    "dataset": {{
        "variant": dataset_variant,
        "dataset_dir": str(dataset_dir),
        "tokenizer_paths": [str(path) for path in tokenizer_paths],
        "tokenizer_present_before": bool(tokenizer_present_before),
        "tokenizer_present_after": bool(tokenizer_present_after),
        "current_train_shards": int(train_shards_after),
        "train_shards_before": int(train_shards_before),
        "train_shards_requested": int(requested_train_shards),
        "dataset_updated": bool(dataset_updated),
    }},
}}
print(json.dumps(summary, sort_keys=True))
PY
"""
    result = run_ssh(ssh_config, remote_cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "bootstrap-runpod failed")
    summary = parse_bootstrap_summary_output(result.stdout)
    summary["ssh"] = {
        "host": ssh_config.host,
        "user": ssh_config.user or "",
        "port": ssh_config.port or 22,
        "identity": ssh_config.identity or "",
        "target": ssh_config.target(),
    }
    return summary


def launch_remote_run(
    resolved: ResolvedManifest,
    *,
    ssh_config: SSHConfig,
    git_sha_value: str,
    repo_url: str,
    remote_root: str,
    run_id: str,
    assigned_gpu_ids: list[int] | None = None,
    extra_env: dict[str, str] | None = None,
    queue_job_id: str | None = None,
) -> dict[str, object]:
    remote_root = remote_root.rstrip("/")
    remote_run_dir = f"{remote_root}/.runs/{run_id}"
    session_name = run_id[:60]
    env = dict(resolved.env)
    env.update(
        {
            "RUN_ID": run_id,
            "RUN_DIR": remote_run_dir,
            "METRICS_JSONL_PATH": f"{remote_run_dir}/metrics.jsonl",
            "HEARTBEAT_JSON_PATH": f"{remote_run_dir}/heartbeat.json",
            "SUMMARY_JSON_PATH": f"{remote_run_dir}/summary.json",
            "EXPERIMENT_NAME": resolved.variant_name,
            "MANIFEST_SHA256": resolved.manifest_sha256,
        }
    )
    if queue_job_id:
        env["QUEUE_JOB_ID"] = queue_job_id
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items()})
    if assigned_gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in assigned_gpu_ids)
        world_size = len(assigned_gpu_ids)
    else:
        world_size = required_gpus_for_resource_class(resolved.resource_class)

    script_path = Path(resolved.script_path)
    remote_script_path = f"{remote_root}/{script_path.relative_to(REPO_ROOT).as_posix()}"
    env_exports = " ".join(f"{key}={json.dumps(str(value))}" for key, value in sorted(env.items()))
    launch_cmd = (
        f"cd {remote_root} && mkdir -p {remote_run_dir} && "
        f"{env_exports} torchrun --standalone --nproc_per_node={world_size} {remote_script_path} "
        f"2>&1 | tee {remote_run_dir}/stdout.log"
    )
    remote_cmd = f"""
set -euo pipefail
if [ ! -d {remote_root}/.git ]; then
  git clone {repo_url} {remote_root}
fi
git -C {remote_root} fetch --all --tags
git -C {remote_root} checkout --force {git_sha_value}
mkdir -p {remote_run_dir}
tmux kill-session -t {session_name} >/dev/null 2>&1 || true
tmux new-session -d -s {session_name} {json.dumps(launch_cmd)}
printf '%s\\n' {json.dumps(session_name)}
"""
    result = run_ssh(ssh_config, remote_cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Remote launch failed")
    return {
        "run_id": run_id,
        "git_sha": git_sha_value,
        "repo_url": repo_url,
        "host": ssh_config.host,
        "user": ssh_config.user,
        "port": ssh_config.port,
        "identity": ssh_config.identity,
        "remote_root": remote_root,
        "remote_run_dir": remote_run_dir,
        "session_name": session_name,
        "world_size": world_size,
        "assigned_gpu_ids": assigned_gpu_ids or list(range(world_size)),
        "launch_command": launch_cmd,
    }


def collect_remote_run(remote: dict[str, object], local_run_dir: Path) -> None:
    ssh_config = SSHConfig(
        host=str(remote["host"]),
        user=remote.get("user") or None,
        port=remote.get("port"),
        identity=remote.get("identity") or None,
    )
    files = [
        "train.log",
        "stdout.log",
        "metrics.jsonl",
        "heartbeat.json",
        "summary.json",
        "final_model.pt",
        "final_model.int8.ptz",
        "final_model.export.ptz",
    ]
    failures: list[str] = []
    for name in files:
        result = scp_from(ssh_config, f"{remote['remote_run_dir']}/{name}", local_run_dir / name)
        if result.returncode != 0 and name in {"train.log", "summary.json"}:
            failures.append(f"{name}: {result.stderr.strip() or 'copy failed'}")
    if failures:
        raise RuntimeError("; ".join(failures))


def remote_file_text(ssh_config: SSHConfig, remote_path: str) -> str | None:
    result = run_ssh(ssh_config, f"test -f {remote_path} && cat {remote_path}")
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout


def tmux_session_exists(ssh_config: SSHConfig, session_name: str) -> bool:
    result = run_ssh(ssh_config, f"tmux has-session -t {session_name}", capture_output=True)
    return result.returncode == 0


def kill_tmux_session(ssh_config: SSHConfig, session_name: str) -> subprocess.CompletedProcess[str]:
    return run_ssh(ssh_config, f"tmux kill-session -t {session_name}", capture_output=True)
