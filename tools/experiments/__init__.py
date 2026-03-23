"""Experiment operations for Parameter Golf."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / ".runs"
QUEUE_DIR = RUNS_DIR / "queue"
QUEUE_EVENTS_PATH = QUEUE_DIR / "events.jsonl"
QUEUE_STATE_PATH = QUEUE_DIR / "state.json"
QUEUE_HOSTS_DIR = QUEUE_DIR / "hosts"
QUEUE_RUNNER_LOG = QUEUE_DIR / "runner.log"
