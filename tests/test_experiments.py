from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from tools.experiments import REPO_ROOT
from tools.experiments.__main__ import _queue_job_brief, _queue_status_payload, _status_payload
from tools.experiments.estimator import estimate_manifest
from tools.experiments.launch import bootstrap_runpod_host, needs_dataset_sync
from tools.experiments.manifest import available_variants, resolve_manifest, write_resolved_manifest
from tools.experiments.promotion import promote_run
from tools.experiments.registry import rebuild_index
from tools.experiments.remote import SSHConfig
from tools.experiments.textlog import parse_train_log_text


class ExperimentOpsTests(unittest.TestCase):
    def test_manifest_variants_expand(self) -> None:
        manifest = REPO_ROOT / "experiments" / "timing_ladder.toml"
        variants = available_variants(manifest)
        self.assertGreaterEqual(len(variants), 12)
        resolved = resolve_manifest(manifest, variant_name=variants[0])
        self.assertEqual(resolved.manifest_name, "timing_ladder")
        self.assertIn("NUM_LAYERS", resolved.env)
        self.assertIn("TRAIN_SEQ_LEN", resolved.env)

    def test_estimator_matches_baseline_anchor_reasonably(self) -> None:
        manifest = REPO_ROOT / "experiments" / "baseline_1gpu_fullshape.toml"
        resolved = resolve_manifest(manifest)
        estimate = estimate_manifest(resolved).values
        self.assertAlmostEqual(estimate["predicted_timing"]["step_ms"], 43.35, delta=2.0)
        self.assertAlmostEqual(estimate["predicted_bytes"]["total_bytes"], 15_900_000, delta=250_000)
        self.assertNotIn("over_byte_cap", estimate["warning_flags"])

    def test_parse_existing_train_log(self) -> None:
        log_path = REPO_ROOT / "records" / "track_10min_16mb" / "2026-03-17_NaiveBaseline" / "train.log"
        parsed = parse_train_log_text(log_path.read_text(encoding="utf-8"))
        self.assertIn("last_train", parsed)
        self.assertIn("last_val", parsed)
        self.assertIn("final_roundtrip", parsed)
        self.assertAlmostEqual(parsed["final_roundtrip"]["val_bpb"], 1.2244, delta=1e-3)

    def test_resolved_manifest_roundtrip(self) -> None:
        resolved = resolve_manifest(REPO_ROOT / "experiments" / "export_family_sweep.toml")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "manifest.resolved.toml"
            write_resolved_manifest(out_path, resolved)
            reloaded = resolve_manifest(out_path)
            self.assertEqual(reloaded.variant_name, resolved.variant_name)
            self.assertEqual(reloaded.env["TRAIN_SEQ_LEN"], resolved.env["TRAIN_SEQ_LEN"])
            self.assertEqual(reloaded.required_nonempty_env, ["INIT_STATE_DICT_PATH"])

    def test_manual_block_reason_roundtrip(self) -> None:
        resolved = resolve_manifest(REPO_ROOT / "experiments" / "architecture_trade_suite.toml")
        self.assertEqual(
            resolved.manual_block_reason,
            "requires constant-byte architecture grid support and non-integer MLP hidden sizing",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "manifest.resolved.toml"
            write_resolved_manifest(out_path, resolved)
            reloaded = resolve_manifest(out_path)
            self.assertEqual(reloaded.manual_block_reason, resolved.manual_block_reason)

    def test_status_payload_includes_progress_and_eta(self) -> None:
        resolved = resolve_manifest(REPO_ROOT / "experiments" / "baseline_1gpu_smoke.toml")
        heartbeat = {
            "run_id": "run_a",
            "phase": "train",
            "timestamp_iso": "2026-03-23T18:30:00Z",
            "step": 25,
            "iterations": 100,
            "elapsed_train_ms": 2500.0,
            "step_avg_ms": 100.0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_a"
            run_dir.mkdir()
            write_resolved_manifest(run_dir / "manifest.resolved.toml", resolved)
            payload = _status_payload(run_dir, heartbeat)
        self.assertEqual(payload["progress_percent"], 25.0)
        self.assertEqual(payload["eta_to_iteration_finish_ms"], 7500.0)
        self.assertEqual(payload["eta_to_wallclock_stop_ms"], 117500.0)
        self.assertEqual(payload["eta_to_completion_ms"], 7500.0)

    def test_queue_job_brief_includes_running_progress(self) -> None:
        resolved = resolve_manifest(REPO_ROOT / "experiments" / "baseline_1gpu_smoke.toml")
        job = {
            "job_id": "job-1",
            "resolved_variant_name": resolved.variant_name,
            "priority": 100,
            "required_gpus": 1,
            "state": "running",
            "state_reason": "",
            "run_id": "run_a",
            "assigned_gpu_ids": [0],
            "estimate": {"predicted_bytes": {"total_bytes": 123}, "predicted_timing": {"step_ms": 42.0}},
            "resolved_manifest": resolved.to_dict(),
            "last_heartbeat": {
                "phase": "train",
                "step": 50,
                "iterations": 200,
                "elapsed_train_ms": 5000.0,
                "step_avg_ms": 100.0,
            },
        }
        payload = _queue_job_brief(job)
        self.assertEqual(payload["progress_percent"], 25.0)
        self.assertEqual(payload["eta_to_iteration_finish_ms"], 15000.0)
        self.assertEqual(payload["eta_to_wallclock_stop_ms"], 115000.0)
        self.assertEqual(payload["eta_to_completion_ms"], 15000.0)

    def test_queue_status_payload_exposes_current_and_total_eta(self) -> None:
        resolved = resolve_manifest(REPO_ROOT / "experiments" / "baseline_1gpu_smoke.toml")
        job = {
            "job_id": "job-1",
            "resolved_variant_name": resolved.variant_name,
            "priority": 100,
            "required_gpus": 1,
            "state": "running",
            "state_reason": "",
            "run_id": "run_a",
            "assigned_gpu_ids": [0],
            "host_slug": "pod-a",
            "estimate": {"predicted_bytes": {"total_bytes": 123}, "predicted_timing": {"step_ms": 42.0}},
            "resolved_manifest": resolved.to_dict(),
            "last_heartbeat": {
                "phase": "train",
                "step": 50,
                "iterations": 200,
                "elapsed_train_ms": 5000.0,
                "step_avg_ms": 100.0,
            },
        }
        state = {
            "jobs": {"job-1": job},
            "hosts": {"pod-a": {"host": "pod-a", "host_slug": "pod-a", "capacity_gpus": 1}},
            "meta": {"updated_at_iso": "2026-03-23T00:00:00Z", "event_count": 1},
        }
        with mock.patch("tools.experiments.__main__.rebuild_queue_state", return_value=state):
            with mock.patch("tools.experiments.__main__.queue_eta_seconds", return_value=321.0):
                payload = _queue_status_payload("pod-a")
        self.assertEqual(payload["queue_eta_seconds"], 321.0)
        self.assertEqual(payload["eta_seconds"], 321.0)
        self.assertEqual(payload["current_run_eta_seconds"], 15.0)
        self.assertEqual(payload["current_run"]["eta_to_completion_ms"], 15000.0)

    def test_promote_run_creates_record_draft(self) -> None:
        resolved = resolve_manifest(REPO_ROOT / "experiments" / "baseline_1gpu_smoke.toml")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_dir = tmp_path / "run_a"
            run_dir.mkdir()
            (run_dir / "source").mkdir()
            write_resolved_manifest(run_dir / "manifest.resolved.toml", resolved)
            (run_dir / "source" / "train_gpt.py").write_text("print('hello')\n", encoding="utf-8")
            (run_dir / "train.log").write_text("step:1/1 train_loss:1.0 train_time:10ms step_avg:10.0ms\n", encoding="utf-8")
            summary = {
                "run_id": "run_a",
                "git_sha": "abc123",
                "trainer": {"final_eval_mode": "standard", "final_eval_stride": 64},
                "metrics": {
                    "final_roundtrip_val_loss": 2.0,
                    "final_roundtrip_val_bpb": 1.2,
                    "final_eval_time_ms": 1234,
                },
                "artifacts": {
                    "raw_model_bytes": 1,
                    "export_model_bytes": 2,
                    "export_total_bytes": 3,
                    "code_bytes": 4,
                },
                "end_time_iso": "2026-03-23T12:00:00Z",
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            promoted = promote_run(run_dir, destination=tmp_path / "promoted_record")
            self.assertTrue((promoted / "README.md").exists())
            self.assertTrue((promoted / "submission.json").exists())
            submission = json.loads((promoted / "submission.json").read_text(encoding="utf-8"))
            self.assertEqual(submission["val_bpb"], 1.2)

    def test_registry_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / ".runs"
            (runs_dir / "run_a").mkdir(parents=True)
            (runs_dir / "run_a" / "summary.json").write_text(json.dumps({"run_id": "run_a", "status": "completed"}), encoding="utf-8")
            index_path = rebuild_index(runs_dir)
            self.assertTrue(index_path.exists())
            self.assertIn("run_a", index_path.read_text(encoding="utf-8"))

    def test_dataset_sync_decision_rules(self) -> None:
        self.assertTrue(needs_dataset_sync(0, 1, False))
        self.assertTrue(needs_dataset_sync(1, 80, True))
        self.assertFalse(needs_dataset_sync(80, 80, True))

    def test_bootstrap_runpod_returns_structured_summary(self) -> None:
        fake_stdout = json.dumps(
            {
                "host": {"hostname": "pod-a", "gpu_count": 1},
                "git": {"before_sha": "old", "after_sha": "new", "requested_sha": "new"},
                "dataset": {
                    "variant": "sp1024",
                    "current_train_shards": 80,
                    "train_shards_before": 1,
                    "train_shards_requested": 80,
                    "dataset_updated": True,
                    "tokenizer_present_after": True,
                },
            }
        )
        with mock.patch("tools.experiments.launch.run_ssh") as run_ssh_mock:
            run_ssh_mock.return_value = mock.Mock(returncode=0, stdout=fake_stdout + "\n", stderr="")
            summary = bootstrap_runpod_host(
                SSHConfig(host="pod.example", user="root", port=22, identity="~/.ssh/id_test"),
                git_sha_value="new",
                repo_url="https://github.com/openai/parameter-golf.git",
                remote_root="/workspace/parameter-golf",
                train_shards=80,
            )
        self.assertEqual(summary["git"]["after_sha"], "new")
        self.assertEqual(summary["dataset"]["current_train_shards"], 80)
        self.assertTrue(summary["dataset"]["dataset_updated"])
        self.assertEqual(summary["ssh"]["target"], "root@pod.example")

    def test_pair_feature_estimator_bigram_counts(self) -> None:
        control = resolve_manifest(
            REPO_ROOT / "experiments" / "pair_feature_bigram_ladder.toml",
            variant_name="pair_feature_bigram_ladder__bigram-vocab-size-0",
        )
        ladder = resolve_manifest(
            REPO_ROOT / "experiments" / "pair_feature_bigram_ladder.toml",
            variant_name="pair_feature_bigram_ladder__bigram-vocab-size-4096",
        )
        control_estimate = estimate_manifest(control).values
        ladder_estimate = estimate_manifest(ladder).values
        self.assertEqual(control_estimate["family_counts"]["bigram"], 0)
        self.assertGreater(ladder_estimate["family_counts"]["bigram"], 0)
        self.assertEqual(ladder_estimate["predicted_eval"]["mode"], "sliding")


if __name__ == "__main__":
    unittest.main()
