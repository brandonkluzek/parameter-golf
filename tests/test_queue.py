from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.experiments import REPO_ROOT
import tools.experiments.queue as queue_mod
from tools.experiments.queue_sets import enqueue_queue_set, expand_queue_set


class QueueOpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self._old_paths = (
            queue_mod.QUEUE_DIR,
            queue_mod.QUEUE_EVENTS_PATH,
            queue_mod.QUEUE_STATE_PATH,
            queue_mod.QUEUE_HOSTS_DIR,
            queue_mod.QUEUE_RUNNER_LOG,
        )
        queue_mod.QUEUE_DIR = tmp_path / ".runs" / "queue"
        queue_mod.QUEUE_EVENTS_PATH = queue_mod.QUEUE_DIR / "events.jsonl"
        queue_mod.QUEUE_STATE_PATH = queue_mod.QUEUE_DIR / "state.json"
        queue_mod.QUEUE_HOSTS_DIR = queue_mod.QUEUE_DIR / "hosts"
        queue_mod.QUEUE_RUNNER_LOG = queue_mod.QUEUE_DIR / "runner.log"

    def tearDown(self) -> None:
        (
            queue_mod.QUEUE_DIR,
            queue_mod.QUEUE_EVENTS_PATH,
            queue_mod.QUEUE_STATE_PATH,
            queue_mod.QUEUE_HOSTS_DIR,
            queue_mod.QUEUE_RUNNER_LOG,
        ) = self._old_paths
        self._tmp.cleanup()

    def test_enqueue_rebuild_and_repeat_count(self) -> None:
        manifest = str(REPO_ROOT / "experiments" / "baseline_1gpu_smoke.toml")
        jobs = queue_mod.enqueue_manifest(manifest, priority=123, count=2)
        self.assertEqual(len(jobs), 2)
        state = queue_mod.load_queue_state()
        self.assertEqual(len(state["jobs"]), 2)
        queue = queue_mod.queued_jobs(state)
        self.assertEqual(len(queue), 2)
        self.assertTrue(all(int(job["priority"]) == 123 for job in queue))

    def test_enqueue_all_variants(self) -> None:
        manifest = str(REPO_ROOT / "experiments" / "timing_ladder.toml")
        jobs = queue_mod.enqueue_manifest(manifest, all_variants=True)
        self.assertEqual(len(jobs), 12)
        state = queue_mod.load_queue_state()
        self.assertEqual(len(state["jobs"]), 12)

    def test_required_env_blocking(self) -> None:
        manifest = str(REPO_ROOT / "experiments" / "export_family_sweep.toml")
        jobs = queue_mod.enqueue_manifest(manifest)
        self.assertEqual(jobs[0]["state"], "blocked")
        self.assertIn("required_env_missing:INIT_STATE_DICT_PATH", jobs[0]["state_reason"])

    def test_manual_block_reason_blocks(self) -> None:
        manifest = str(REPO_ROOT / "experiments" / "architecture_trade_suite.toml")
        jobs = queue_mod.enqueue_manifest(manifest)
        self.assertEqual(jobs[0]["state"], "blocked")
        self.assertEqual(
            jobs[0]["state_reason"],
            "requires constant-byte architecture grid support and non-integer MLP hidden sizing",
        )

    def test_estimated_over_byte_cap_skips(self) -> None:
        manifest_path = Path(self._tmp.name) / "too_big.toml"
        manifest_path.write_text(
            """
name = "too_big"
baseline = "root_train_gpt"
script_path = "train_gpt.py"
track = "non_record"
resource_class = "runpod_1xh100"
seed_mode = "fixed"

[env]
DATA_PATH = "./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
VOCAB_SIZE = 1024
TRAIN_SEQ_LEN = 1024
TRAIN_BATCH_TOKENS = 524288
NUM_LAYERS = 24
MODEL_DIM = 4096
MLP_MULT = 4
ITERATIONS = 10
VAL_LOSS_EVERY = 0
WARMUP_STEPS = 0
MAX_WALLCLOCK_SECONDS = 10
""".strip()
            + "\n",
            encoding="utf-8",
        )
        jobs = queue_mod.enqueue_manifest(str(manifest_path))
        self.assertEqual(jobs[0]["state"], "skipped")
        self.assertEqual(jobs[0]["state_reason"], "estimated_over_byte_cap")

    def test_priority_ordering_slot_packing_and_drain_mode(self) -> None:
        one_gpu_manifest = str(REPO_ROOT / "experiments" / "baseline_1gpu_smoke.toml")
        eight_gpu_manifest = Path(self._tmp.name) / "eight_gpu.toml"
        eight_gpu_manifest.write_text(
            """
name = "eight_gpu"
baseline = "root_train_gpt"
script_path = "train_gpt.py"
track = "non_record"
resource_class = "runpod_8xh100"
seed_mode = "fixed"

[env]
DATA_PATH = "./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
VOCAB_SIZE = 1024
TRAIN_SEQ_LEN = 1024
TRAIN_BATCH_TOKENS = 524288
NUM_LAYERS = 9
MODEL_DIM = 512
MLP_MULT = 2
ITERATIONS = 10
VAL_LOSS_EVERY = 0
WARMUP_STEPS = 0
MAX_WALLCLOCK_SECONDS = 10
""".strip()
            + "\n",
            encoding="utf-8",
        )
        low = queue_mod.enqueue_manifest(one_gpu_manifest, priority=100)[0]
        high = queue_mod.enqueue_manifest(str(eight_gpu_manifest), priority=200)[0]
        state = queue_mod.load_queue_state()
        queue = queue_mod.queued_jobs(state)
        self.assertEqual(queue[0]["job_id"], high["job_id"])
        self.assertTrue(queue_mod.is_host_draining(state, 8))

        queue_mod.update_job(low["job_id"], {"state": "running", "host_slug": "pod-a", "assigned_gpu_ids": [0]})
        state = queue_mod.load_queue_state()
        self.assertEqual(queue_mod.free_gpu_slots(state, "pod-a", 8), [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(queue_mod.allocate_gpu_slots(queue_mod.free_gpu_slots(state, "pod-a", 8), 1), [1])
        self.assertIsNone(queue_mod.allocate_gpu_slots(queue_mod.free_gpu_slots(state, "pod-a", 8), 8))

    def test_overrun_and_restart_classification(self) -> None:
        heartbeat = {"step": 250, "elapsed_train_ms": 400000.0, "step_avg_ms": 1200.0}
        self.assertTrue(queue_mod.should_fail_overrun(heartbeat, predicted_step_ms=50.0))
        self.assertEqual(
            queue_mod.classify_remote_snapshot(
                heartbeat=heartbeat,
                summary=None,
                tmux_alive=True,
                predicted_step_ms=50.0,
            )[0],
            "failed_overrun",
        )
        healthy_but_slow = {"step": 55, "elapsed_train_ms": 70000.0, "step_avg_ms": 100.0}
        self.assertFalse(queue_mod.should_fail_overrun(healthy_but_slow, predicted_step_ms=50.0))
        self.assertEqual(
            queue_mod.classify_remote_snapshot(
                heartbeat=None,
                summary={"status": "completed"},
                tmux_alive=False,
                predicted_step_ms=50.0,
            )[0],
            "completed",
        )
        self.assertEqual(
            queue_mod.classify_remote_snapshot(
                heartbeat=None,
                summary=None,
                tmux_alive=False,
                predicted_step_ms=50.0,
            )[0],
            "failed",
        )

    def test_queue_set_expansion_precedence(self) -> None:
        queue_set_path = Path(self._tmp.name) / "set.toml"
        queue_set_path.write_text(
            """
name = "test_set"
priority = 100
count = 2

[set]
BASE = "default"
SHARED = "queue"

[[job]]
manifest = "experiments/baseline_1gpu_smoke.toml"
priority = 110

[[job]]
manifest = "experiments/export_family_sweep.toml"
set = { INIT_STATE_DICT_PATH = "/remote/checkpoint.pt", SHARED = "job" }
""".strip()
            + "\n",
            encoding="utf-8",
        )
        queue_set = expand_queue_set(queue_set_path, cli_env_overrides={"SHARED": "cli"})
        self.assertEqual(queue_set.jobs[0].priority, 110)
        self.assertEqual(queue_set.jobs[0].count, 2)
        self.assertEqual(queue_set.jobs[0].env_overrides["BASE"], "default")
        self.assertEqual(queue_set.jobs[0].env_overrides["SHARED"], "cli")
        self.assertEqual(queue_set.jobs[1].env_overrides["INIT_STATE_DICT_PATH"], "/remote/checkpoint.pt")
        self.assertEqual(queue_set.jobs[1].env_overrides["SHARED"], "cli")

    def test_enqueue_queue_set_and_checkpoint_override(self) -> None:
        bringup = REPO_ROOT / "experiments" / "queue_sets" / "bringup_1gpu.toml"
        result = enqueue_queue_set(bringup)
        self.assertEqual(len(result["enqueued"]), 5)
        self.assertTrue(all(job["state"] == "queued" for job in result["enqueued"]))
        bringup_variants = {job["resolved_variant_name"] for job in result["enqueued"]}
        self.assertIn("timing_ladder__mlp-mult-2__num-layers-9__train-seq-len-1024", bringup_variants)
        self.assertIn("timing_ladder__mlp-mult-2__num-layers-9__train-seq-len-2048", bringup_variants)
        self.assertIn("timing_ladder__mlp-mult-3__num-layers-11__train-seq-len-1024", bringup_variants)

        calibration = REPO_ROOT / "experiments" / "queue_sets" / "checkpoint_calibration_1gpu.toml"
        blocked = enqueue_queue_set(calibration)
        self.assertEqual(len(blocked["enqueued"]), 10)
        self.assertTrue(all(job["state"] == "blocked" for job in blocked["enqueued"]))
        self.assertTrue(
            all("required_env_missing:INIT_STATE_DICT_PATH" in job["state_reason"] for job in blocked["enqueued"])
        )

        unblocked = enqueue_queue_set(calibration, cli_env_overrides={"INIT_STATE_DICT_PATH": "/remote/checkpoint.pt"})
        self.assertEqual(len(unblocked["enqueued"]), 10)
        self.assertTrue(all(job["state"] == "queued" for job in unblocked["enqueued"]))

    def test_research_and_future_queue_sets(self) -> None:
        research = REPO_ROOT / "experiments" / "queue_sets" / "research_wave_1gpu.toml"
        research_result = enqueue_queue_set(research)
        self.assertEqual(len(research_result["enqueued"]), 9)
        self.assertTrue(all(job["state"] == "queued" for job in research_result["enqueued"]))

        future = REPO_ROOT / "experiments" / "queue_sets" / "future_pilots_blocked.toml"
        future_result = enqueue_queue_set(future)
        self.assertEqual(len(future_result["enqueued"]), 2)
        self.assertTrue(all(job["state"] == "blocked" for job in future_result["enqueued"]))
        reasons = {job["state_reason"] for job in future_result["enqueued"]}
        self.assertIn("requires constant-byte architecture grid support and non-integer MLP hidden sizing", reasons)
        self.assertIn("requires proxy-buffer and candidate-buffer weighting support", reasons)


if __name__ == "__main__":
    unittest.main()
