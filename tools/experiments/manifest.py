from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import REPO_ROOT

try:
    import tomllib
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Python 3.11+ is required for the experiment tooling") from exc


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "value"


def _stringify_env_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _toml_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    return json.dumps(str(value))


@dataclass
class PromotionConfig:
    candidate: str = "exploratory"
    name: str = ""
    blurb: str = ""
    track_override: str = ""


@dataclass
class ResolvedManifest:
    manifest_path: str
    manifest_name: str
    variant_name: str
    baseline: str
    script_path: str
    track: str
    resource_class: str
    seed_mode: str
    estimator_family: str
    manual_block_reason: str = ""
    required_nonempty_env: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    env: dict[str, str] = field(default_factory=dict)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    matrix_values: dict[str, object] = field(default_factory=dict)
    checks: dict[str, object] = field(default_factory=dict)
    manifest_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["promotion"] = asdict(self.promotion)
        return data

    def to_toml(self) -> str:
        lines = [
            f'name = {_toml_literal(self.manifest_name)}',
            f'manifest_path = {_toml_literal(self.manifest_path)}',
            f'manifest_name = {_toml_literal(self.manifest_name)}',
            f'variant_name = {_toml_literal(self.variant_name)}',
            f'baseline = {_toml_literal(self.baseline)}',
            f'script_path = {_toml_literal(self.script_path)}',
            f'track = {_toml_literal(self.track)}',
            f'resource_class = {_toml_literal(self.resource_class)}',
            f'seed_mode = {_toml_literal(self.seed_mode)}',
            f'estimator_family = {_toml_literal(self.estimator_family)}',
            f'manual_block_reason = {_toml_literal(self.manual_block_reason)}',
            f'required_nonempty_env = {_toml_literal(self.required_nonempty_env)}',
            f'tags = {_toml_literal(self.tags)}',
            f'notes = {_toml_literal(self.notes)}',
            f'manifest_sha256 = {_toml_literal(self.manifest_sha256)}',
            "",
            "[promotion]",
        ]
        for key, value in asdict(self.promotion).items():
            lines.append(f"{key} = {_toml_literal(value)}")
        for section_name, values in (("matrix_values", self.matrix_values), ("checks", self.checks), ("env", self.env)):
            lines.append("")
            lines.append(f"[{section_name}]")
            for key, value in sorted(values.items()):
                lines.append(f"{key} = {_toml_literal(value)}")
        return "\n".join(lines) + "\n"


def _load_raw_manifest(path: Path) -> dict[str, object]:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    required = ["name", "baseline", "script_path", "track", "resource_class", "seed_mode"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Manifest {path} is missing required keys: {', '.join(missing)}")
    return raw


def _matrix_variants(raw: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    matrix = raw.get("matrix", {})
    if not matrix:
        return [(str(raw.get("variant_name", raw["name"])), {})]
    keys = sorted(matrix)
    values = [matrix[key] for key in keys]
    variants: list[tuple[str, dict[str, object]]] = []
    for combo in itertools.product(*values):
        matrix_values = dict(zip(keys, combo, strict=True))
        suffix = "__".join(f"{_slug(key)}-{_slug(value)}" for key, value in matrix_values.items())
        variants.append((f"{raw['name']}__{suffix}", matrix_values))
    return variants


def available_variants(manifest_path: str | Path) -> list[str]:
    path = Path(manifest_path)
    raw = _load_raw_manifest(path)
    return [name for name, _ in _matrix_variants(raw)]


def resolve_manifest(
    manifest_path: str | Path,
    *,
    variant_name: str | None = None,
    variant_index: int | None = None,
    env_overrides: dict[str, str] | None = None,
) -> ResolvedManifest:
    path = Path(manifest_path)
    raw = _load_raw_manifest(path)
    variants = _matrix_variants(raw)
    if variant_name is not None:
        selected = [item for item in variants if item[0] == variant_name]
        if not selected:
            raise ValueError(f"Unknown variant {variant_name!r}. Available: {', '.join(name for name, _ in variants)}")
        chosen_name, matrix_values = selected[0]
    else:
        chosen_idx = variant_index or 0
        if chosen_idx < 0 or chosen_idx >= len(variants):
            raise ValueError(f"Variant index {chosen_idx} out of range for {path}")
        chosen_name, matrix_values = variants[chosen_idx]

    env = {key: _stringify_env_value(value) for key, value in raw.get("env", {}).items()}
    for key, value in matrix_values.items():
        env[key] = _stringify_env_value(value)
    if env_overrides:
        env.update({key: _stringify_env_value(value) for key, value in env_overrides.items()})

    script_path = Path(str(raw["script_path"]))
    if not script_path.is_absolute():
        script_path = (REPO_ROOT / script_path).resolve()
    checks = {
        "script_exists": script_path.exists(),
        "data_path_exists": Path(env["DATA_PATH"]).resolve().exists() if "DATA_PATH" in env else None,
        "tokenizer_path_exists": Path(env["TOKENIZER_PATH"]).resolve().exists() if "TOKENIZER_PATH" in env else None,
    }
    manifest_hash_payload = {
        "manifest_path": str(path.resolve()),
        "variant_name": chosen_name,
        "matrix_values": matrix_values,
        "env": env,
        "track": raw["track"],
        "resource_class": raw["resource_class"],
        "seed_mode": raw["seed_mode"],
        "manual_block_reason": str(raw.get("manual_block_reason", "")),
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest_hash_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    promotion_raw = raw.get("promotion", {})
    return ResolvedManifest(
        manifest_path=str(path.resolve()),
        manifest_name=str(raw["name"]),
        variant_name=chosen_name,
        baseline=str(raw["baseline"]),
        script_path=str(script_path),
        track=str(raw["track"]),
        resource_class=str(raw["resource_class"]),
        seed_mode=str(raw["seed_mode"]),
        estimator_family=str(raw.get("estimator_family", "root_public_2026_03_23")),
        manual_block_reason=str(raw.get("manual_block_reason", "")),
        required_nonempty_env=[str(value) for value in raw.get("required_nonempty_env", [])],
        tags=[str(tag) for tag in raw.get("tags", [])],
        notes=str(raw.get("notes", "")),
        env=env,
        promotion=PromotionConfig(
            candidate=str(promotion_raw.get("candidate", "exploratory")),
            name=str(promotion_raw.get("name", "")),
            blurb=str(promotion_raw.get("blurb", "")),
            track_override=str(promotion_raw.get("track_override", "")),
        ),
        matrix_values=matrix_values,
        checks=checks,
        manifest_sha256=manifest_sha256,
    )


def write_resolved_manifest(path: str | Path, resolved: ResolvedManifest) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(resolved.to_toml(), encoding="utf-8")


def resolved_manifest_from_dict(data: dict[str, object]) -> ResolvedManifest:
    promotion_data = data.get("promotion", {})
    return ResolvedManifest(
        manifest_path=str(data["manifest_path"]),
        manifest_name=str(data["manifest_name"]),
        variant_name=str(data.get("variant_name", data["manifest_name"])),
        baseline=str(data["baseline"]),
        script_path=str(data["script_path"]),
        track=str(data["track"]),
        resource_class=str(data["resource_class"]),
        seed_mode=str(data["seed_mode"]),
        estimator_family=str(data.get("estimator_family", "root_public_2026_03_23")),
        manual_block_reason=str(data.get("manual_block_reason", "")),
        required_nonempty_env=[str(value) for value in data.get("required_nonempty_env", [])],
        tags=[str(tag) for tag in data.get("tags", [])],
        notes=str(data.get("notes", "")),
        env={str(key): str(value) for key, value in dict(data.get("env", {})).items()},
        promotion=PromotionConfig(
            candidate=str(dict(promotion_data).get("candidate", "exploratory")),
            name=str(dict(promotion_data).get("name", "")),
            blurb=str(dict(promotion_data).get("blurb", "")),
            track_override=str(dict(promotion_data).get("track_override", "")),
        ),
        matrix_values={str(key): value for key, value in dict(data.get("matrix_values", {})).items()},
        checks={str(key): value for key, value in dict(data.get("checks", {})).items()},
        manifest_sha256=str(data.get("manifest_sha256", "")),
    )
