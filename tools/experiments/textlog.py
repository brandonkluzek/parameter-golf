from __future__ import annotations

import re


TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+train_loss:(?P<train_loss>[-+0-9.]+)\s+train_time:(?P<train_time_ms>\d+)ms\s+step_avg:(?P<step_avg_ms>[-+0-9.]+)ms"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+val_loss:(?P<val_loss>[-+0-9.]+)\s+val_bpb:(?P<val_bpb>[-+0-9.]+)\s+train_time:(?P<train_time_ms>\d+)ms\s+step_avg:(?P<step_avg_ms>[-+0-9.]+)ms"
)
ARTIFACT_RE = re.compile(r"Total submission size(?: [^:]+)?: (?P<bytes>\d+) bytes")
FINAL_RE = re.compile(
    r"final(?:_[a-z0-9_]+)?_roundtrip(?:_exact)?\s+val_loss:(?P<val_loss>[-+0-9.]+)\s+val_bpb:(?P<val_bpb>[-+0-9.]+)(?:\s+eval_time:(?P<eval_time_ms>\d+)ms)?",
    re.IGNORECASE,
)


def parse_train_log_text(text: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for line in text.splitlines():
        if match := TRAIN_RE.search(line):
            result["last_train"] = {
                "step": int(match.group("step")),
                "iterations": int(match.group("iterations")),
                "train_loss": float(match.group("train_loss")),
                "train_time_ms": int(match.group("train_time_ms")),
                "step_avg_ms": float(match.group("step_avg_ms")),
            }
        if match := VAL_RE.search(line):
            result["last_val"] = {
                "step": int(match.group("step")),
                "iterations": int(match.group("iterations")),
                "val_loss": float(match.group("val_loss")),
                "val_bpb": float(match.group("val_bpb")),
                "train_time_ms": int(match.group("train_time_ms")),
                "step_avg_ms": float(match.group("step_avg_ms")),
            }
        if match := ARTIFACT_RE.search(line):
            result["artifact_bytes"] = int(match.group("bytes"))
        if match := FINAL_RE.search(line):
            result["final_roundtrip"] = {
                "val_loss": float(match.group("val_loss")),
                "val_bpb": float(match.group("val_bpb")),
                "eval_time_ms": int(match.group("eval_time_ms")) if match.group("eval_time_ms") else None,
            }
    return result
