"""Production-length synthetic acceptance. Never evidence of scientific readiness.

Only the image encoder is replaced: a deterministic tiny backbone emits 768
features and accepts the production LoRA wrappers. The actual decoder, dropout,
optimizer, sampling, cache writer, snapshots, recovery and evaluator are used.
The separate real-MAE smoke remains required for architecture/device readiness.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn

from raid_experiments import fewshot_train as training
from raid_experiments.fewshot import DEVELOPMENT_TASKS, OFFICIAL_GOAL_TASKS, LIBERO_COMMIT, create_goal_protocol
from raid_experiments.fewshot_report import CONTROLS, freeze_development, report_fewshot
from src.cache_mae_features import cache_protocol_tasks

LEARNED = ("head_ft", "lora_head_ft")


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        return x + self.proj(self.qkv(x).reshape(-1, 3, 8).mean(1)).tanh() * 0.1


class TinyEncoder(nn.Module):
    """Synthetic pixels-to-features model; deliberately not a MAE implementation."""
    def __init__(self):
        super().__init__()
        self.mae = nn.Module()
        self.mae.input = nn.Linear(3, 8)
        self.mae.blocks = nn.ModuleList([nn.Module() for _ in range(12)])
        for block in self.mae.blocks:
            block.attn = TinyAttention()
        self.mae.output = nn.Linear(8, 768)
        self.requires_grad_(False)

    def encode_frames(self, images):
        x = images.float().mean((1, 2)) / 255.0
        x = self.mae.input(x)
        for block in self.mae.blocks:
            x = block.attn(x)
        return nn.functional.normalize(self.mae.output(x), dim=-1)


def fixture_encoder(_checkpoint, device="cpu"):
    # Loading a deterministic pretrained fixture must not perturb training RNG.
    with torch.random.fork_rng():
        torch.manual_seed(7919)
        return TinyEncoder().to(device).eval()


def build_fixture(root):
    data = root / "data" / "libero_goal"
    data.mkdir(parents=True)
    for task_index, task in enumerate(OFFICIAL_GOAL_TASKS):
        with h5py.File(data / f"{task}_demo.hdf5", "w") as handle:
            for demo in range(50):
                length = 2 + demo % 3
                group = handle.create_group(f"data/demo_{demo}")
                frames = np.empty((length + 1, 4, 4, 3), dtype=np.uint8)
                for step in range(length + 1):
                    frames[step] = [demo * 4, task_index * 20, step * 30]
                group.create_group("obs").create_dataset("agentview_rgb", data=frames)
                # Labels are coherent with the observed demo/task/time channels.
                actions = np.array([[demo / 50, task_index / 10, step / 10,
                                     (demo + step) / 50, -demo / 50, step / 20,
                                     float((demo + step) % 2)] for step in range(length)], dtype=np.float32)
                group.create_dataset("actions", data=actions)
    protocol_path = root / "protocol.json"
    protocol = create_goal_protocol(data.parent, protocol_path)
    checkpoint = root / "synthetic-encoder.pth"
    checkpoint.write_bytes(b"tiny synthetic acceptance encoder; not pretrained MAE")
    upstream = root / "fixture-upstream"
    upstream.mkdir()
    def git(command, **kwargs):
        return SimpleNamespace(stdout=LIBERO_COMMIT if command[-2:] == ["rev-parse", "HEAD"] else "")
    with patch("src.cache_mae_features.subprocess.run", side_effect=git):
        manifest = cache_protocol_tasks(protocol_path, data.parent, root / "features", checkpoint,
            upstream, device="cpu", batch_size=8, execution_mode="synthetic", encoder_factory=fixture_encoder)
    return protocol, manifest


def adaptation(root, output, method, **kwargs):
    return training.adapt_target(root / "protocol.json", root / "features",
        root / "source" / "source_seed42.pt", output, task=DEVELOPMENT_TASKS[0],
        method=method, budget=5, seed=42, updates=1000, device="cpu",
        batch_size=64, microbatch_size=8, run_mode="synthetic",
        mae_checkpoint=root / "synthetic-encoder.pth", **kwargs)


def condition(output):
    return output / DEVELOPMENT_TASKS[0] / "5shot" / "seed42"


def evaluate(root, output, methods, step):
    return training.evaluate_task(root / "protocol.json", root / "features",
        root / "source" / "source_seed42.pt", output, task=DEVELOPMENT_TASKS[0],
        budget=5, seed=42, methods=methods, selected_updates={m: step for m in methods if m in LEARNED},
        device="cpu", run_mode="synthetic", mae_checkpoint=root / "synthetic-encoder.pth")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def signal_worker(root, output, method, step, signum, resume):
    """Deliver a real OS signal immediately after a production optimizer checkpoint."""
    save = training._atomic_save
    def save_and_signal(payload, path):
        save(payload, path)
        if Path(path).name == f"{method}.last.pt" and payload["step"] == step:
            os.kill(os.getpid(), signum)
    with patch("src.mae_encoder.MAEImageEncoder.from_checkpoint", side_effect=fixture_encoder), \
            patch.object(training, "_atomic_save", side_effect=save_and_signal):
        adaptation(root, output, method, resume_checkpoint=resume)


class ProductionAcceptanceTests(unittest.TestCase):
    def test_snapshot_serialization_ignores_incidental_python_aliases(self):
        first = {"curve": [{"step": 200, "loss": 1.0}, {"step": 500, "loss": 0.5}],
                 "state_dict": {"weight": torch.arange(3.)}}
        # Unpickling the prefix then appending creates equal but distinct strings.
        import io
        prefix = torch.load(io.BytesIO(training._snapshot_bytes({"curve": first["curve"][:1]})), weights_only=False)
        resumed = {"curve": prefix["curve"] + first["curve"][1:], "state_dict": first["state_dict"]}
        self.assertTrue(training._same_checkpoint_value(first, resumed))
        self.assertEqual(hashlib.sha256(training._snapshot_bytes(first)).hexdigest(),
                         hashlib.sha256(training._snapshot_bytes(resumed)).hexdigest())

    def test_report_reaches_both_bootstrap_roles_and_returns_without_protocol(self):
        from raid_experiments import fewshot_report as reporting
        protocol = {"task_roles": {"development": ["dev"], "final_test": ["test"]}}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(reporting, "validate_goal_protocol", return_value=protocol), \
                    patch.object(reporting, "_paired_bootstrap", return_value=None) as bootstrap:
                report = reporting.report_fewshot(temp, "fixture-protocol")
            calls = [(call.args[3], call.args[4]) for call in bootstrap.call_args_list]
            self.assertEqual(len(calls), 30)
            self.assertEqual(set(calls), {(role, budget) for role in ("development", "final_test") for budget in (2, 5, 10)})
            self.assertEqual(report["expected_canonical_evaluations"], 12)
            self.assertEqual(reporting.report_fewshot(temp)["valid_evaluation_artifacts"], 0)

    def test_all_repetitions_nested_support_sampling_and_query_isolation(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as temp, \
                patch("src.mae_encoder.MAEImageEncoder.from_checkpoint", side_effect=fixture_encoder):
            root = Path(temp)
            protocol, _ = build_fixture(root)
            task = DEVELOPMENT_TASKS[0]
            query_reads = []
            original_read = h5py.Dataset.__getitem__
            query_by_file = {str(Path(spec["path"]).resolve()): set(spec.get("query", []))
                             for spec in protocol["tasks"].values()}
            def read(dataset, index):
                if dataset.name.endswith("/actions"):
                    demo = int(dataset.name.split("/")[-2].removeprefix("demo_"))
                    if demo in query_by_file.get(str(Path(dataset.file.filename).resolve()), set()):
                        query_reads.append((dataset.file.filename, demo))
                        raise AssertionError("query label read before valid evaluation")
                return original_read(dataset, index)
            with patch.object(h5py.Dataset, "__getitem__", new=read):
                for seed in (42, 43, 44):
                    source_dir = root / f"source-{seed}"
                    training.train_source(root / "protocol.json", root / "features", source_dir,
                        seed=seed, updates=2, device="cpu", batch_size=64, microbatch_size=8, run_mode="synthetic")
                    nested = protocol["tasks"][task]["support_budgets_by_seed"][str(seed)]
                    self.assertEqual(nested["2"], nested["5"][:2])
                    self.assertEqual(nested["5"], nested["10"][:5])
                    for budget in (2, 5, 10):
                        schedules = {}
                        for method in LEARNED:
                            seen = []
                            original_loss = nn.functional.mse_loss
                            def loss(prediction, target, *args, **kwargs):
                                seen.append(target.detach().clone())
                                return original_loss(prediction, target, *args, **kwargs)
                            with patch.object(nn.functional, "mse_loss", side_effect=loss):
                                training.adapt_target(root / "protocol.json", root / "features",
                                    source_dir / f"source_seed{seed}.pt", root / "runs", task=task,
                                    method=method, budget=budget, seed=seed, updates=2, device="cpu",
                                    batch_size=64, microbatch_size=8, run_mode="synthetic",
                                    mae_checkpoint=root / "synthetic-encoder.pth")
                            self.assertEqual(len(seen), 16)
                            schedules[method] = torch.cat(seen)
                        torch.testing.assert_close(schedules["head_ft"], schedules["lora_head_ft"], rtol=0, atol=0)
                # Invalid scientific conditions must fail before any query labels open.
                for task in protocol["task_roles"]["final_test"]:
                    with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                        training.evaluate_task(root / "protocol.json", root / "features",
                            root / "source-42" / "source_seed42.pt", root / "relocated-output",
                            task=task, budget=5, seed=42, methods=("lora_head_ft",), device="cpu")
                self.assertEqual(query_reads, [])

    def test_production_calibration_hooks_measure_one_warmup_and_update_session(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as temp, \
                patch("src.mae_encoder.MAEImageEncoder.from_checkpoint", side_effect=fixture_encoder):
            root = Path(temp)
            build_fixture(root)
            profiles = {method: {"warmup_updates": 5} for method in ("source", *LEARNED)}
            training.train_source(root / "protocol.json", root / "features", root / "source", updates=25,
                device="cpu", batch_size=64, microbatch_size=8, run_mode="synthetic", timing_profile=profiles["source"])
            for method in LEARNED:
                training.adapt_target(root / "protocol.json", root / "features", root / "source" / "source_seed42.pt",
                    root / "runs", task=DEVELOPMENT_TASKS[0], method=method, budget=5, seed=42, updates=25,
                    device="cpu", batch_size=64, microbatch_size=8, run_mode="synthetic",
                    mae_checkpoint=root / "synthetic-encoder.pth", timing_profile=profiles[method])
            for profile in profiles.values():
                self.assertEqual(len(profile["warmup_update_seconds"]), 5)
                self.assertEqual(len(profile["measured_update_seconds"]), 20)
                measured = sum(profile["warmup_update_seconds"] + profile["measured_update_seconds"])
                self.assertGreater(profile["initialization_seconds"], 0)
                self.assertGreater(profile["total_seconds"], measured + profile["initialization_seconds"])
                for key in ("checkpoint_write_seconds", "checkpoint_serialization_seconds", "snapshot_publish_seconds"):
                    self.assertTrue(profile[key])
                    self.assertTrue(all(value > 0 for value in profile[key]))
                self.assertIsNone(profile["peak_memory_bytes"])
            self.assertEqual(len(profiles["source"]["source_validation_seconds"]), 1)
            report = evaluate(root, root / "runs", (*CONTROLS, *LEARNED), 25)
            timings = report["performance"]["measurements"]
            self.assertEqual(set(timings), set((*CONTROLS, *LEARNED)))
            self.assertTrue(all(row["cached_compute_ms_per_transition"] > 0 for row in timings.values()))
            self.assertGreater(timings["lora_head_ft"]["raw_frame_encoding_seconds"], 0)
            self.assertTrue(all(timings[m]["source_retention_seconds"] > 0 for m in ("frozen_source", *LEARNED)))
            immutable = condition(root / "runs") / "evaluation_head25_lora25.json"
            before = immutable.read_bytes()
            evaluate(root, root / "runs", (*CONTROLS, *LEARNED), 25)
            self.assertEqual(immutable.read_bytes(), before)

    def test_calibration_schema_rejects_missing_nonfinite_and_partial_components(self):
        import copy
        from raid_experiments.fewshot_calibration import _validate_measurements
        # Schema unit fixture only; never written as a bundle/receipt.
        measurements = {key: 1.0 for key in (
            "mae_encoder_initialization_seconds", "feature_cache_wall_seconds", "raw_frame_encoding_seconds",
            "retrieval_bank_construction_seconds", "query_label_and_feature_loading_seconds",
            "eight_method_query_evaluation_seconds", "checkpoint_io_copy_seconds", "source_recovery_seconds",
            "peak_memory_bytes", "artifact_storage_bytes", "static_asset_cache_bytes", "source_run_storage_bytes",
            "head_ft_run_storage_bytes", "lora_head_ft_run_storage_bytes")}
        measurements["training_profiles"] = {}
        for learner in ("source", *LEARNED):
            measurements[f"{learner}_warmup_update_seconds"] = [1.] * 5
            measurements[f"{learner}_update_seconds"] = [1.] * 20
            measurements[f"{learner}_initialization_and_data_prep_seconds"] = 1.
            measurements["training_profiles"][learner] = {"peak_memory_bytes": 1,
                "checkpoint_write_seconds": [1.], "checkpoint_serialization_seconds": [1.], "snapshot_publish_seconds": [1.]}
        measurements["training_profiles"]["source"]["source_validation_seconds"] = [1.]
        measurements["per_method_query_ms_per_transition"] = {m: 1. for m in (*CONTROLS, *LEARNED)}
        measurements["source_retention_seconds"] = {m: 1. for m in ("frozen_source", *LEARNED)}
        measurements["adaptation_recovery_seconds"] = {m: 1. for m in LEARNED}
        _validate_measurements(measurements)
        for method in (*CONTROLS, *LEARNED):
            for value in (None, float("nan"), float("inf"), 0., -1.):
                bad = copy.deepcopy(measurements)
                bad["per_method_query_ms_per_transition"][method] = value
                with self.assertRaises(ValueError):
                    _validate_measurements(bad)
        for learner in ("source", *LEARNED):
            bad = copy.deepcopy(measurements)
            bad[f"{learner}_warmup_update_seconds"].pop()
            with self.assertRaisesRegex(ValueError, "5 warmup"):
                _validate_measurements(bad)
            bad = copy.deepcopy(measurements)
            del bad["training_profiles"][learner]["checkpoint_write_seconds"]
            with self.assertRaisesRegex(ValueError, "actual"):
                _validate_measurements(bad)

    def test_inventory_and_forecast_count_writes_validation_and_selected_updates(self):
        from raid_experiments.fewshot import workload_inventory
        from raid_experiments.fewshot_assets import _forecast_calibration_workload
        timings = {key: 1. for key in ("source_training_20_updates_seconds", "head_adaptation_20_updates_seconds",
            "lora_adaptation_20_updates_seconds", "feature_cache_wall_seconds", "development_baseline_bundle_seconds",
            "development_candidate_bundle_seconds", "confirmation_bundle_seconds", "retrieval_bank_build_seconds",
            "retention_pass_seconds", "checkpoint_io_bundle_seconds", "checkpoint_write_seconds",
            "source_validation_pass_seconds", "source_recovery_seconds", "recovery_seconds_per_adaptation",
            "total_artifact_storage_bytes", "feature_cache_encoder_initialization_seconds",
            "source_initialization_and_data_prep_seconds", "head_ft_initialization_and_data_prep_seconds",
            "lora_head_ft_initialization_and_data_prep_seconds")}
        timings.update(static_asset_cache_bytes=1024**3, source_run_storage_bytes=1024**3,
                       head_ft_run_storage_bytes=1024**3, lora_head_ft_run_storage_bytes=1024**3)
        budget = {"planning_budget_usd": 250., "protected_reserve_usd": 25., "workload_margin": .25}
        pricing = {"gpu_hourly_usd": 1., "storage_gb_month_usd": .1, "retention_months": 1}
        for scope, conditions, writes, updates, evaluation, banks, retention, recovery, initialization, storage in (
                ("all", 18, 801, 2400, 60, 48, 96, 51, 52, 295),
                ("five_shot", 6, 465, 1200, 48, 24, 60, 27, 28, 151)):
            inventory = workload_inventory(scope)
            self.assertEqual(inventory["checkpoint_writes"], writes)
            self.assertEqual(inventory["source_validation_passes"], 60)
            self.assertEqual(inventory["confirmation_updates"], {"head_ft": 1000, "lora_head_ft": 1000})
            forecast = _forecast_calibration_workload(timings, pricing, budget, 30., scope=scope)
            seconds = 300 + updates + 1 + evaluation + banks + retention + writes + recovery + initialization + 60
            self.assertAlmostEqual(forecast["workload_hours_with_margin"], seconds / 3600 * 1.25)
            self.assertEqual(forecast["storage_gb"], storage)
            self.assertAlmostEqual(forecast["total_cost_usd"], seconds / 3600 * 1.25 + storage * .1)
            self.assertEqual(forecast["available_before_protected_reserve_usd"], 195.)
            self.assertLess(forecast["total_cost_usd"] + 30 + 25, 250)
        selected = {"head_ft": 200, "lora_head_ft": 500}
        inventory = workload_inventory("five_shot", selected)
        self.assertEqual(inventory["adaptation_updates_by_method"], {"head_ft": 7200, "lora_head_ft": 9000})
        self.assertEqual(inventory["checkpoint_writes"], 129 + 168 + 6 * (4 + 8))
        forecast = _forecast_calibration_workload(timings, pricing, budget, 30., scope="five_shot", selected_updates=selected)
        self.assertAlmostEqual(forecast["adaptation_hours"], (7200 + 9000) / 20 / 3600)

    def test_full_length_training_evaluation_and_scheduled_recovery(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as temp, \
                patch("src.mae_encoder.MAEImageEncoder.from_checkpoint", side_effect=fixture_encoder):
            root = Path(temp)
            protocol, manifest = build_fixture(root)
            self.assertEqual(manifest["execution_mode"], "synthetic")
            self.assertEqual(sum(len(v) for v in manifest["demo_shards"].values()), 500)
            source = training.train_source(root / "protocol.json", root / "features", root / "source",
                updates=2000, device="cpu", batch_size=64, microbatch_size=8, run_mode="synthetic")
            self.assertEqual(source["updates"], 2000)
            source_payload = load(root / "source" / "source_seed42.pt")
            self.assertTrue(source_payload["training_complete"])
            self.assertFalse(source_payload["scientific_complete"])
            self.assertEqual(len(source_payload["curve"]), 20)
            # A completed source snapshot also recovers as exact original bytes,
            # retaining its original canonical ledger even from a relocated copy.
            recovered_source = root / "relocated-source"
            training.train_source(root / "protocol.json", root / "features", recovered_source,
                updates=2000, device="cpu", batch_size=64, microbatch_size=8, run_mode="synthetic",
                resume_checkpoint=root / "source" / "source_seed42.last.pt")
            self.assertEqual(training._sha(recovered_source / "source_seed42.pt"),
                             training._sha(root / "source" / "source_seed42.pt"))
            from raid_experiments.fewshot_report import canonical_study_root
            self.assertEqual(canonical_study_root({42: recovered_source / "source_seed42.pt"}), (root / "source").resolve())
            self.assertEqual((recovered_source / "source_seed42.complete.json").read_bytes(),
                             (root / "source" / "source_seed42.complete.json").read_bytes())

            for method in LEARNED:
                saved = training._atomic_save
                def capture(payload, path):
                    saved(payload, path)
                    if Path(path).name == f"{method}.last.pt" and payload["step"] in (200, 500, 1000):
                        archive = root / "continuations" / f"{method}-{payload['step']}.pt"
                        archive.parent.mkdir(exist_ok=True)
                        archive.write_bytes(Path(path).read_bytes())
                with patch.object(training, "_atomic_save", side_effect=capture):
                    result = adaptation(root, root / "full", method)
                self.assertEqual(result["updates"], 1000)
                self.assertEqual(result["status"], "synthetic")
                self.assertFalse(result["condition_training_complete"])
                self.assertEqual([p["step"] for p in result["curve"]], [200, 500, 1000])
                if method == "lora_head_ft":
                    self.assertTrue(result["frozen_backbone_unchanged"])
                    self.assertTrue(result["lora_parameters_changed"])
                    self.assertGreater(result["max_lora_gradient_norm"], 0)

            # Paired methods consume the same permutations despite live RGB loading.
            head_last = load(condition(root / "full") / "head_ft.last.pt")
            lora_last = load(condition(root / "full") / "lora_head_ft.last.pt")
            for key in ("order", "cursor", "order_rng_state"):
                self.assertTrue(training._same_checkpoint_value(head_last[key], lora_last[key]), key)
            methods = (*CONTROLS, *LEARNED)
            for step in (200, 500, 1000):
                report = evaluate(root, root / "full", methods, step)
                self.assertEqual(set(report["methods"]), set(methods))
                self.assertEqual(report["status"], "synthetic")
                for method, record in report["prediction_artifacts"].items():
                    payload = load(condition(root / "full") / record["path"])
                    self.assertEqual(payload["prediction_format"], 4)
                    self.assertEqual(payload["execution_mode"], "synthetic")
                    self.assertEqual(payload["transition_ids"], report["query_ids"])
                    self.assertTrue(torch.isfinite(payload["predictions"]).all())
                    self.assertEqual(payload["selected_update"], step if method in LEARNED else None)

            for method in LEARNED:
                reference = load(condition(root / "full") / f"{method}.last.pt")
                for step in (200, 500, 1000):
                    with self.subTest(method=method, snapshot=step):
                        out = root / f"resume-{method}-{step}"
                        continuation = root / "continuations" / f"{method}-{step}.pt"
                        adaptation(root, out, method, resume_checkpoint=continuation)
                        final = load(condition(out) / f"{method}.last.pt") if step < 1000 else load(continuation)
                        for key in ("state_dict", "lora_state_dict", "optimizer", "order", "cursor",
                                    "order_rng_state", "torch_rng_state", "numpy_rng_state", "python_rng_state", "cuda_rng_state"):
                            self.assertTrue(training._same_checkpoint_value(reference.get(key), final.get(key)), key)
                        for snapshot in (200, 500, 1000):
                            name = f"{method}.step{snapshot}.pt"
                            self.assertEqual(training._sha(condition(out) / name), training._sha(condition(root / "full") / name), name)
                        report = evaluate(root, out, (method,), 1000)
                        prediction = load(condition(out) / report["prediction_artifacts"][method]["path"])
                        original = load(condition(root / "full") / f"{method}_predictions_head1000_lora1000.pt")
                        torch.testing.assert_close(prediction["predictions"], original["predictions"], rtol=0, atol=0)
                        # Completed recovery and evaluation may be retried without replacing evidence.
                        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in condition(out).glob("*.step*.pt")}
                        adaptation(root, out, method, resume_checkpoint=continuation)
                        evaluate(root, out, (method,), 1000)
                        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in condition(out).glob("*.step*.pt")})
                        conflict = condition(out) / f"{method}.step200.pt"
                        conflict.write_bytes(b"conflicting evidence")
                        with self.assertRaisesRegex(ValueError, "existing snapshot differs"):
                            adaptation(root, out, method, resume_checkpoint=continuation)

            # Signals are delivered by the OS in fresh interpreters. Resume each
            # interruption in another directory; every saved snapshot is recovered.
            for method in LEARNED:
                resume = None
                for step, sig in ((200, signal.SIGINT), (500, signal.SIGTERM), (1000, signal.SIGINT)):
                    out = root / f"signal-{method}-{step}"
                    command = [sys.executable, __file__, str(root), str(out), method, str(step), str(int(sig)), str(resume) if resume else ""]
                    child = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ, "PYTHONPATH": ".", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                        capture_output=True, text=True, timeout=240)
                    self.assertEqual(child.returncode, 0, child.stdout[-2000:] + child.stderr[-2000:])
                    current = load(condition(out) / f"{method}.last.pt")
                    expected = load(root / "continuations" / f"{method}-{step}.pt")
                    for key in ("state_dict", "lora_state_dict", "optimizer", "order", "cursor", "order_rng_state",
                                "torch_rng_state", "numpy_rng_state", "python_rng_state", "cuda_rng_state", "snapshot_archive_bytes"):
                        self.assertTrue(training._same_checkpoint_value(current.get(key), expected.get(key)), f"{method}/{step}/{key}")
                    if step < 1000:
                        self.assertEqual(json.loads((condition(out) / f"{method}.json").read_text())["status"], "interrupted")
                    resume = condition(out) / f"{method}.last.pt"

            # Completing the full update schedule does not promote synthetic data.
            with self.assertRaisesRegex(ValueError, "scientific"):
                freeze_development(root / "protocol.json", root / "full", root / "source" / "source_seed42.pt", None,
                                   root / "development_freeze.json")
            with self.assertRaisesRegex(ValueError, "scientific feature cache"):
                training.evaluate_task(root / "protocol.json", root / "features", root / "source" / "source_seed42.pt",
                    root / "full", task=DEVELOPMENT_TASKS[0], budget=5, seed=42, methods=CONTROLS, device="cpu")
            summary = report_fewshot(root / "full", root / "protocol.json")
            self.assertEqual(summary["valid_evaluation_artifacts"], 0)
            self.assertTrue(summary["missing_conditions"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    signal_worker(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
                  Path(sys.argv[6]) if sys.argv[6] else None)
