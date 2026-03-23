from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .manifest import ResolvedManifest

BYTE_CAP = 16_000_000
TRAIN_BUDGET_SECONDS = 600.0
EVAL_BUDGET_SECONDS = 600.0
CODE_BYTES_DEFAULT = 52_000


@dataclass
class Estimate:
    values: dict[str, Any]


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    return int(env.get(key, default))


def _env_float(env: dict[str, str], key: str, default: float) -> float:
    return float(env.get(key, default))


def _coeff_for_family(
    family: str,
    *,
    export_scheme: str,
    export_compressor: str,
    lowbit_categories: tuple[str, ...],
    fp16_patterns: tuple[str, ...],
    int8_patterns: tuple[str, ...],
    embed_clip_range: int,
    mlp_clip_range: int,
    attn_clip_range: int,
    bigram_clip_range: int,
) -> float:
    zstd_int6_coeff = 0.55
    if export_scheme == "int8_zlib":
        return 0.926
    if family == "embed":
        if any("tok_emb" in pattern for pattern in fp16_patterns):
            return 1.90
        if any("tok_emb" in pattern for pattern in int8_patterns):
            return 0.926
        if "embed" in lowbit_categories:
            if export_compressor == "zstd":
                return 0.525 if embed_clip_range <= 15 else zstd_int6_coeff
            return 0.698
    if family == "mlp":
        if mlp_clip_range <= 15:
            return 0.525 if export_compressor == "zstd" else 0.60
        return zstd_int6_coeff if export_compressor == "zstd" else 0.698
    if family == "attn":
        if attn_clip_range <= 31:
            return zstd_int6_coeff if export_compressor == "zstd" else 0.698
    if family == "bigram":
        if bigram_clip_range <= 31:
            return zstd_int6_coeff if export_compressor == "zstd" else 0.698
    return 0.926


def estimate_manifest(resolved: ResolvedManifest) -> Estimate:
    env = resolved.env
    d = _env_int(env, "MODEL_DIM", 512)
    l_s = _env_int(env, "NUM_LAYERS", 9)
    l_u = l_s
    v = _env_int(env, "VOCAB_SIZE", 1024)
    m = _env_float(env, "MLP_MULT", 2.0)
    batch_tok = _env_int(env, "TRAIN_BATCH_TOKENS", 524_288)
    train_seq = _env_int(env, "TRAIN_SEQ_LEN", 1024)
    heads = _env_int(env, "NUM_HEADS", 8)
    kv_heads = _env_int(env, "NUM_KV_HEADS", 4)
    tie_embeddings = env.get("TIE_EMBEDDINGS", "1") not in {"0", "false", "False"}
    int6_layer_start = _env_int(env, "INT6_LAYER_START", -1)
    int6_layer_end = _env_int(env, "INT6_LAYER_END", -1)
    export_scheme = env.get("EXPORT_SCHEME", "")
    if not export_scheme:
        if env.get("QAT_INT6", "0") in {"1", "true", "True"} or env.get("FP16_EMBED_EXPORT", "0") in {"1", "true", "True"}:
            export_scheme = "mixed_family"
        elif int6_layer_start >= 0 and int6_layer_end >= int6_layer_start:
            export_scheme = "mixed_family"
        elif _env_int(env, "BIGRAM_VOCAB_SIZE", 0) > 0 or _env_int(env, "EVAL_STRIDE", 0) > 0:
            export_scheme = "mixed_family"
        else:
            export_scheme = "int8_zlib"
    export_compressor = env.get("EXPORT_COMPRESSOR", "")
    if not export_compressor:
        export_compressor = "zstd" if env.get("USE_ZSTD", "0") in {"1", "true", "True"} else "zlib"
    lowbit_categories = tuple(
        part.strip()
        for part in env.get("EXPORT_LOWBIT_CATEGORIES", "attn,mlp").split(",")
        if part.strip()
    )
    fp16_patterns = tuple(part.strip() for part in env.get("EXPORT_FP16_PATTERNS", "").split(",") if part.strip())
    int8_patterns = tuple(part.strip() for part in env.get("EXPORT_INT8_PATTERNS", "tok_emb").split(",") if part.strip())
    if env.get("FP16_EMBED_EXPORT", "0") in {"1", "true", "True"} and not any("tok_emb" in pattern for pattern in fp16_patterns):
        fp16_patterns = fp16_patterns + ("tok_emb",)
        int8_patterns = tuple(pattern for pattern in int8_patterns if "tok_emb" not in pattern)
    embed_clip_range = _env_int(env, "EXPORT_EMBED_CLIP_RANGE", 31)
    mlp_clip_range = _env_int(env, "EXPORT_MLP_CLIP_RANGE", 31)
    attn_clip_range = _env_int(env, "EXPORT_ATTN_CLIP_RANGE", 31)
    bigram_clip_range = _env_int(env, "EXPORT_BIGRAM_CLIP_RANGE", 31)
    eval_stride_hint = _env_int(env, "EVAL_STRIDE", 0)
    final_eval_mode = env.get("FINAL_EVAL_MODE", "sliding" if eval_stride_hint > 0 else "standard")
    final_eval_stride = _env_int(env, "FINAL_EVAL_STRIDE", eval_stride_hint or 64)
    code_bytes = _env_int(env, "EST_CODE_BYTES", CODE_BYTES_DEFAULT)

    n_emb = v * d
    n_att = 3 * l_s * d * d
    n_mlp = int(round(2 * m * l_s * d * d))
    n_ctl = int(round(6 * l_s * d))
    bigram_buckets = _env_int(env, "BIGRAM_HASH_BUCKETS", _env_int(env, "BIGRAM_VOCAB_SIZE", 0))
    bigram_dim = _env_int(env, "BIGRAM_HASH_DIM", _env_int(env, "BIGRAM_DIM", d))
    n_bigram = bigram_buckets * bigram_dim

    c_emb = _coeff_for_family(
        "embed",
        export_scheme=export_scheme,
        export_compressor=export_compressor,
        lowbit_categories=lowbit_categories,
        fp16_patterns=fp16_patterns,
        int8_patterns=int8_patterns,
        embed_clip_range=embed_clip_range,
        mlp_clip_range=mlp_clip_range,
        attn_clip_range=attn_clip_range,
        bigram_clip_range=bigram_clip_range,
    )
    c_att = _coeff_for_family(
        "attn",
        export_scheme=export_scheme,
        export_compressor=export_compressor,
        lowbit_categories=lowbit_categories,
        fp16_patterns=fp16_patterns,
        int8_patterns=int8_patterns,
        embed_clip_range=embed_clip_range,
        mlp_clip_range=mlp_clip_range,
        attn_clip_range=attn_clip_range,
        bigram_clip_range=bigram_clip_range,
    )
    c_mlp = _coeff_for_family(
        "mlp",
        export_scheme=export_scheme,
        export_compressor=export_compressor,
        lowbit_categories=lowbit_categories,
        fp16_patterns=fp16_patterns,
        int8_patterns=int8_patterns,
        embed_clip_range=embed_clip_range,
        mlp_clip_range=mlp_clip_range,
        attn_clip_range=attn_clip_range,
        bigram_clip_range=bigram_clip_range,
    )
    c_bigram = _coeff_for_family(
        "bigram",
        export_scheme=export_scheme,
        export_compressor=export_compressor,
        lowbit_categories=lowbit_categories,
        fp16_patterns=fp16_patterns,
        int8_patterns=int8_patterns,
        embed_clip_range=embed_clip_range,
        mlp_clip_range=mlp_clip_range,
        attn_clip_range=attn_clip_range,
        bigram_clip_range=bigram_clip_range,
    )
    c_ctl = 1.90

    model_bytes = int(round(c_emb * n_emb + c_att * n_att + c_mlp * n_mlp + c_ctl * n_ctl + c_bigram * n_bigram))
    total_bytes = model_bytes + code_bytes
    byte_slack = BYTE_CAP - total_bytes

    phi_dense = (batch_tok / 524_288.0) * (l_u / 9.0) * (((3.0 + 2.0 * m) * d * d) / (7.0 * 512.0 * 512.0))
    phi_attn = (batch_tok / 524_288.0) * (l_u / 9.0) * (d / 512.0) * (train_seq / 1024.0)
    delta_longctx = 0.0 if train_seq <= 2048 else 16.0 * (train_seq / 4096.0)
    delta_qat = 12.0 if env.get("QAT_INT6", "0") in {"1", "true", "True"} else 0.0
    predicted_step_ms = 18.0 + 17.0 * phi_dense + 8.35 * phi_attn + delta_longctx + delta_qat
    train_tokens_within_budget = int((TRAIN_BUDGET_SECONDS * 1000.0) * batch_tok / max(predicted_step_ms, 1.0))

    eval_seq = _env_int(env, "FINAL_EVAL_SEQ_LEN", _env_int(env, "EVAL_SEQ_LEN", train_seq))
    eval_shape = (l_u / 9.0) * (0.87 * (eval_seq / 1024.0) + 0.13 * (((3.0 + 2.0 * m) / 7.0) * (d / 512.0)))
    if final_eval_mode == "sliding":
        predicted_eval_seconds = 70.0 * (64.0 / final_eval_stride) * eval_shape
    else:
        predicted_eval_seconds = 16.0 * eval_shape

    warnings: list[str] = []
    if not tie_embeddings or heads != 8 or kv_heads != 4:
        warnings.append("unsupported_family")
    if total_bytes > BYTE_CAP:
        warnings.append("over_byte_cap")
    if predicted_eval_seconds > EVAL_BUDGET_SECONDS:
        warnings.append("over_eval_budget")
    if train_seq >= 4096:
        warnings.append("context_overpriced")
    embed_protected = any("tok_emb" in pattern for pattern in fp16_patterns) or any("tok_emb" in pattern for pattern in int8_patterns)
    if export_scheme != "int8_zlib" and not embed_protected:
        warnings.append("fragile_embedding_cut")

    bpb_priors: list[dict[str, Any]] = []
    if final_eval_mode == "sliding" and final_eval_stride == 64:
        bpb_priors.append({"move": "sliding_stride_64", "delta_bpb": -0.033})
    if train_seq == 2048:
        bpb_priors.append({"move": "train_seq_2048", "delta_bpb": -0.0185})
    if train_seq >= 4096:
        bpb_priors.append({"move": "train_seq_4096", "delta_bpb": -0.0045})
    if export_scheme == "mixed_family" and any("tok_emb" in pattern for pattern in fp16_patterns):
        bpb_priors.append({"move": "embedding_protection", "delta_bpb": -0.006})
    if export_scheme == "mixed_family" and mlp_clip_range <= 15:
        bpb_priors.append({"move": "mlp_int5_capacity_financing", "delta_bpb": -0.003})

    baseline_total = 15_863_489
    baseline_step_ms = 43.54
    baseline_eval_seconds = 16.0
    return Estimate(
        {
            "family_counts": {
                "embedding": n_emb,
                "attention": n_att,
                "mlp": n_mlp,
                "control": n_ctl,
                "bigram": n_bigram,
            },
            "predicted_bytes": {
                "code_bytes": code_bytes,
                "embedding_bytes": int(round(c_emb * n_emb)),
                "attention_bytes": int(round(c_att * n_att)),
                "mlp_bytes": int(round(c_mlp * n_mlp)),
                "control_bytes": int(round(c_ctl * n_ctl)),
                "bigram_bytes": int(round(c_bigram * n_bigram)),
                "model_bytes": model_bytes,
                "total_bytes": total_bytes,
                "byte_slack": byte_slack,
                "delta_vs_baseline": total_bytes - baseline_total,
            },
            "predicted_timing": {
                "step_ms": round(predicted_step_ms, 3),
                "train_tokens_600s": train_tokens_within_budget,
                "delta_step_ms_vs_baseline": round(predicted_step_ms - baseline_step_ms, 3),
            },
            "predicted_eval": {
                "mode": final_eval_mode,
                "stride": final_eval_stride if final_eval_mode == "sliding" else None,
                "seconds": round(predicted_eval_seconds, 3),
                "delta_seconds_vs_baseline": round(predicted_eval_seconds - baseline_eval_seconds, 3),
            },
            "structural_deltas": {
                "n_emb_delta_vs_baseline": n_emb - (1024 * 512),
                "n_att_delta_vs_baseline": n_att - (3 * 9 * 512 * 512),
                "n_mlp_delta_vs_baseline": n_mlp - (2 * 2 * 9 * 512 * 512),
            },
            "budget": {
                "byte_cap": BYTE_CAP,
                "train_budget_seconds": TRAIN_BUDGET_SECONDS,
                "eval_budget_seconds": EVAL_BUDGET_SECONDS,
            },
            "warning_flags": warnings,
            "bpb_priors": bpb_priors,
        }
    )
