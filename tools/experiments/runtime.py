from __future__ import annotations

import io
import math
import zlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

try:
    import zstandard as zstd_mod
except ImportError:
    zstd_mod = None


@dataclass
class ExportConfig:
    export_scheme: str
    export_compressor: str
    export_lowbit_categories: tuple[str, ...]
    export_int8_patterns: tuple[str, ...]
    export_fp16_patterns: tuple[str, ...]
    export_embed_clip_range: int
    export_attn_clip_range: int
    export_mlp_clip_range: int
    export_bigram_clip_range: int
    control_tensor_name_patterns: tuple[str, ...]
    keep_float_fp32_name_patterns: tuple[str, ...]
    keep_float_max_numel: int
    keep_float_store_dtype: torch.dtype
    per_row_scale_dtype: torch.dtype
    int8_clip_q: float
    lowbit_clip_q: float


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def eval_val_sliding_window(
    *,
    seq_len: int,
    model_for_logits,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
) -> tuple[float, float]:
    total_tokens = val_tokens.numel() - 1
    num_windows = max((total_tokens - seq_len) // stride + 1, 1)
    win_start = (num_windows * rank) // world_size
    win_end = (num_windows * (rank + 1)) // world_size
    eval_batch = int(__import__("os").environ.get("SW_EVAL_BATCH", 32))

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model_for_logits.eval()
    with torch.inference_mode():
        window_list = list(range(win_start, win_end))
        num_batches = (len(window_list) + eval_batch - 1) // eval_batch
        for batch_idx in range(num_batches):
            batch_wins = window_list[batch_idx * eval_batch : (batch_idx + 1) * eval_batch]
            if not batch_wins:
                continue
            inputs = torch.stack(
                [val_tokens[w * stride : w * stride + seq_len] for w in batch_wins]
            ).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model_for_logits.forward_logits(inputs)

            scored_logits = logits[:, -stride:, :].reshape(-1, logits.size(-1))
            targets = torch.stack(
                [
                    val_tokens[w * stride + seq_len - stride + 1 : w * stride + seq_len + 1]
                    for w in batch_wins
                ]
            ).to(device=device, dtype=torch.int64).reshape(-1)
            loss = F.cross_entropy(scored_logits.float(), targets, reduction="sum")
            val_loss_sum += loss.to(torch.float64)
            val_token_count += float(targets.numel())

            prev_ids = torch.stack(
                [
                    val_tokens[w * stride + seq_len - stride : w * stride + seq_len]
                    for w in batch_wins
                ]
            ).to(device=device, dtype=torch.int64).reshape(-1)
            tgt_ids = targets
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model_for_logits.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def _keep_float_tensor(name: str, t: Tensor, config: ExportConfig) -> Tensor:
    if any(pattern in name for pattern in config.keep_float_fp32_name_patterns):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        return t.to(dtype=config.keep_float_store_dtype).contiguous()
    return t


def _classify_export_family(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    if "bigram" in name:
        return "bigram"
    return "other"


def _quantize_float_tensor(t: Tensor, config: ExportConfig) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), config.int8_clip_q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=config.per_row_scale_dtype).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), config.int8_clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def _quantize_lowbit_per_row(t: Tensor, config: ExportConfig, clip_range: int) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), config.lowbit_clip_q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(clip_range)).clamp_min(1.0 / float(max(clip_range, 1)))
        q = torch.clamp(torch.round(clipped / scale[:, None]), -clip_range, clip_range).to(torch.int8).contiguous()
        return q, scale.to(dtype=config.per_row_scale_dtype).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), config.lowbit_clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(max(clip_range, 1)) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -clip_range, clip_range).to(torch.int8).contiguous()
    return q, scale


def _quantize_state_dict_int8(config: ExportConfig, state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= config.keep_float_max_numel:
            kept = _keep_float_tensor(name, t, config)
            if t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = _quantize_float_tensor(t, config)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def _quantize_state_dict_mixed(config: ExportConfig, state_dict: dict[str, Tensor]):
    export: dict[str, Tensor] = {}
    meta: dict[str, dict[str, object]] = {}
    dtypes: dict[str, str] = {}
    stats = {
        "param_count": 0,
        "num_tensors": 0,
        "num_float_tensors": 0,
        "num_nonfloat_tensors": 0,
        "baseline_tensor_bytes": 0,
        "export_payload_bytes": 0,
    }
    lowbit_categories = set(config.export_lowbit_categories)

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        dtypes[name] = str(t.dtype).removeprefix("torch.")

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            export[name] = t
            meta[name] = {"type": "passthrough"}
            stats["export_payload_bytes"] += tensor_nbytes(t)
            continue

        family = _classify_export_family(name)
        if any(pattern in name for pattern in config.control_tensor_name_patterns):
            kept = t.float().contiguous()
            export[name] = kept
            meta[name] = {"type": "passthrough_ctrl", "family": family}
            stats["export_payload_bytes"] += tensor_nbytes(kept)
            continue
        if any(pattern in name for pattern in config.export_fp16_patterns):
            kept = t.to(dtype=torch.float16).contiguous()
            export[name] = kept
            meta[name] = {"type": "passthrough_fp16", "family": family}
            stats["export_payload_bytes"] += tensor_nbytes(kept)
            continue
        if t.numel() <= config.keep_float_max_numel:
            kept = _keep_float_tensor(name, t, config)
            export[name] = kept
            meta[name] = {"type": "passthrough_small", "family": family}
            stats["export_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        if any(pattern in name for pattern in config.export_int8_patterns):
            q, s = _quantize_float_tensor(t, config)
            export[name + ".q"] = q
            export[name + ".scale"] = s
            meta[name] = {"type": "int8", "family": family}
            stats["export_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
            continue

        clip_range = None
        if family in lowbit_categories:
            if family == "embed":
                clip_range = config.export_embed_clip_range
            elif family == "mlp":
                clip_range = config.export_mlp_clip_range
            elif family == "attn":
                clip_range = config.export_attn_clip_range
            elif family == "bigram":
                clip_range = config.export_bigram_clip_range
        if clip_range is not None:
            q, s = _quantize_lowbit_per_row(t, config, clip_range=clip_range)
            export[name + ".q"] = q
            export[name + ".scale"] = s
            meta[name] = {
                "type": f"int{5 if clip_range <= 15 else 6}",
                "family": family,
                "clip_range": clip_range,
            }
            stats["export_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
            continue

        q, s = _quantize_float_tensor(t, config)
        export[name + ".q"] = q
        export[name + ".scale"] = s
        meta[name] = {"type": "int8", "family": family}
        stats["export_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    return {
        "__quant_format__": "mixed_family_v1",
        "export": export,
        "meta": meta,
        "dtypes": dtypes,
    }, stats


def dequantize_exported_state(obj: dict[str, object]) -> dict[str, Tensor]:
    quant_format = obj.get("__quant_format__")
    if quant_format == "int8_clean_per_row_v1":
        out: dict[str, Tensor] = {}
        qmeta = obj.get("qmeta", {})
        passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
        for name, q in obj["quantized"].items():
            dtype = getattr(torch, obj["dtypes"][name])
            s = obj["scales"][name]
            if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
                s = s.to(dtype=torch.float32)
                out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
            else:
                scale = float(s.item())
                out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
        for name, t in obj["passthrough"].items():
            out_t = t.detach().to("cpu").contiguous()
            orig_dtype = passthrough_orig_dtypes.get(name)
            if isinstance(orig_dtype, str):
                out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
            out[name] = out_t
        return out
    if quant_format == "mixed_family_v1":
        out: dict[str, Tensor] = {}
        export = obj["export"]
        meta = obj["meta"]
        dtypes = obj["dtypes"]
        for name, info in meta.items():
            dtype = getattr(torch, dtypes[name])
            export_type = info["type"]
            if export_type.startswith("passthrough"):
                out_t = export[name].detach().to("cpu").contiguous()
                if out_t.dtype != dtype:
                    out_t = out_t.to(dtype=dtype).contiguous()
                out[name] = out_t
                continue
            q = export[name + ".q"]
            s = export[name + ".scale"]
            if getattr(s, "ndim", 0) > 0:
                out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
            else:
                out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
        return out
    raise ValueError(f"Unsupported quant format: {quant_format}")


def export_state_dict(config: ExportConfig, state_dict: dict[str, Tensor]) -> tuple[dict[str, object], dict[str, int | float], str]:
    if config.export_scheme == "int8_zlib":
        obj, stats = _quantize_state_dict_int8(config, state_dict)
        return obj, stats, "int8_zlib"
    if config.export_scheme == "mixed_family":
        obj, stats = _quantize_state_dict_mixed(config, state_dict)
        return obj, stats, f"mixed_family_{config.export_compressor}"
    raise ValueError(f"Unsupported export scheme: {config.export_scheme}")


def compress_export_blob(raw: bytes, compressor: str) -> tuple[bytes, str]:
    if compressor == "zlib":
        return zlib.compress(raw, level=9), "zlib"
    if compressor == "zstd":
        if zstd_mod is None:
            raise RuntimeError("EXPORT_COMPRESSOR=zstd requested but zstandard is not installed")
        return zstd_mod.ZstdCompressor(level=22).compress(raw), "zstd-22"
    raise ValueError(f"Unsupported compressor: {compressor}")


def decompress_export_blob(blob: bytes, compressor: str) -> bytes:
    if compressor == "zlib":
        return zlib.decompress(blob)
    if compressor == "zstd":
        if zstd_mod is None:
            raise RuntimeError("EXPORT_COMPRESSOR=zstd requested but zstandard is not installed")
        return zstd_mod.ZstdDecompressor().decompress(blob)
    raise ValueError(f"Unsupported compressor: {compressor}")
