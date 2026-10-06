import json
import tempfile
import unittest
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn

from raid_experiments.config import enumerate_expansion, load_config
from raid_experiments.models import LowRankLinear, RAIDDecoder, attach_mae_lora
from raid_experiments.retrieval import TransitionMemory
from raid_experiments.splits import load_and_validate
from raid_experiments.evaluation import action_metrics
from raid_experiments.data import FeatureTransitions
from raid_experiments.splits import _digest
from raid_experiments.simulator import _state
from raid_experiments.predictors import Cosmos3EdgeFuturePredictor, FuturePrediction
from raid_experiments.report import build_report
from raid_experiments.splits import select_training_ids
from raid_experiments.raw_training import RawImageTransitions, _collate as _raw_collate
from raid_experiments.splits import demo_uid, create_split
from raid_experiments import cloud


class TestPilot(unittest.TestCase):
    def test_config(self):
        cfg = load_config("configs/pilot_100.yaml")
        self.assertEqual(cfg.data.train_per_task, 10)
        self.assertEqual(cfg.model.lora_rank, 8)
        self.assertEqual(len(cfg.digest()), 16)

    def test_retrieval_excludes_demo(self):
        memory = TransitionMemory(2, 1)
        memory.add(torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.tensor([4.]), 7)
        memory.add(torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.tensor([9.]), 8)
        result = memory.retrieve(torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]), query_demo_ids=torch.tensor([7]))
        self.assertEqual(int(result.indices[0, 0]), 1)
        self.assertTrue(bool(result.valid[0, 0]))

    def test_empty_retrieval_is_finite(self):
        model = RAIDDecoder(feature_dim=2, action_dim=1, hidden_dim=4, bottleneck_dim=3)
        current = torch.randn(1, 2)
        future = torch.randn(1, 2)
        retrieved = torch.zeros(1, 2, 1)
        pred, _ = model(current, future, retrieved, torch.zeros(1, 2, dtype=torch.bool))
        self.assertTrue(torch.isfinite(pred).all())

    def test_metrics_include_task_macro_and_physical_units(self):
        metrics = action_metrics([[1.0, 2.0], [3.0, 4.0]], [[0.0, 2.0], [1.0, 5.0]], mean=[0, 0], std=[2, 3], task_ids=["a", "b"])
        self.assertIn("task_macro_normalized_mse", metrics)
        self.assertIn("unnormalized_per_action_mse", metrics)

    def test_legacy_future_feature_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature = root / "features"; feature.mkdir()
            torch.save({"task_name": "task_a", "horizon": 3, "demo_lengths": [2, 2],
                        "feat_t": torch.zeros(4, 2), "feat_next": torch.ones(4, 2),
                        "actions_raw": torch.tensor([[0.], [0.], [10.], [10.]]),
                        "future_representation": "recorded_rgb"}, feature / "task_a_features.pt")
            split = {"format": 1, "seed": 1, "tasks": [{"task": "task_a", "train": [0, 1], "validation": [], "test": []}]}
            split["digest"] = _digest(split)
            split_path = root / "split.json"; split_path.write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError, "horizon-one recorded-frame"):
                FeatureTransitions(feature, split_path, "train", train_per_task=1, horizon=3)
            # Even a spoofed format marker cannot make the removed combined
            # cache schema an active format-4 input.
            combined_v4 = root / "combined-v4.pt"
            torch.save({"format": 4, "feat_t": torch.zeros(1, 768),
                        "feat_next": torch.zeros(1, 768), "actions": torch.zeros(1, 7)}, combined_v4)
            from src.data_libero import CachedFeatureDataset
            with self.assertRaisesRegex(ValueError, "combined legacy feature caches are rejected"):
                CachedFeatureDataset(combined_v4)

    def test_validation_reuses_training_normalizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); feature = root / "features"; feature.mkdir()
            torch.save({"task_name": "task_a", "horizon": 1, "demo_lengths": [2, 2],
                        "feat_t": torch.zeros(4, 2), "feat_next": torch.ones(4, 2),
                        "actions_raw": torch.tensor([[0.], [0.], [100.], [100.]])}, feature / "task_a_features.pt")
            split = {"format": 1, "seed": 1, "tasks": [{"task": "task_a", "train": [0], "validation": [1], "test": []}]}
            split["digest"] = _digest(split); path = root / "split.json"; path.write_text(json.dumps(split))
            train = FeatureTransitions(feature, path, "train", train_per_task=1, horizon=1)
            val = FeatureTransitions(feature, path, "validation", train_per_task=1, horizon=1, norm_mean=train.norm_mean, norm_std=train.norm_std)
            self.assertAlmostEqual(float(val.norm_mean[0]), float(train.norm_mean[0]), places=5)

    def test_raw_and_cached_paths_share_ids_and_constant_normalizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); feature = root / "features"; feature.mkdir()
            tasks = []
            for task_name in ("task_a", "task_b"):
                path = root / f"{task_name}.hdf5"
                with h5py.File(path, "w") as handle:
                    for demo in (2, 5):
                        group = handle.create_group(f"data/demo_{demo}")
                        group.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32))
                        group.create_group("obs").create_dataset("agentview_rgb", data=np.zeros((4, 8, 8, 3), dtype=np.uint8))
                torch.save({"task_name": task_name, "horizon": 1, "demo_ids": [2, 5], "demo_lengths": [3, 3],
                            "feat_t": torch.zeros(6, 2), "feat_next": torch.ones(6, 2),
                            "actions_raw": torch.zeros(6, 7)}, feature / f"{task_name}_features.pt")
                tasks.append({"task": task_name, "path": str(path), "train": [2], "validation": [5], "test": []})
            split = {"format": 1, "seed": 1, "horizon": 1, "tasks": tasks}
            split["digest"] = _digest(split)
            split_path = root / "split.json"; split_path.write_text(json.dumps(split))
            cfg = load_config("configs/pilot_100.yaml")
            cfg = replace(cfg, horizon=1, data=replace(cfg.data, dataset_dir=str(root), feature_dir=str(feature), split_path=str(split_path), tasks=2, train_per_task=1))
            cached = FeatureTransitions(feature, split_path, "train", train_per_task=1, horizon=1)
            raw = RawImageTransitions(cfg, "train")
            self.assertEqual(sorted(x.demo_id for x in cached.items), sorted(x["demo_id"] for x in raw.items))
            self.assertEqual(float(cached.norm_std[0]), 1.0)
            self.assertEqual(float(raw.std[0]), 1.0)

    def test_expansion_is_explicitly_enumerated(self):
        jobs = enumerate_expansion("configs/expansion_study.yaml")
        self.assertEqual(len(jobs), 60)
        self.assertEqual({x["budget"] for x in jobs}, {2, 5, 10, 20, 40})
        self.assertEqual({x["seed"] for x in jobs}, {42, 43, 44})

    def test_budget_deadline_uses_reserved_duration(self):
        cfg = load_config("configs/pilot_100.yaml")
        cfg = replace(cfg, cloud=replace(cfg.cloud, budget_usd=28.0, reserve_usd=22.0, terminate_after_minutes=180, launch_enabled=True))
        gpu = {"displayName": "A40", "gpuId": "A40", "securePricePerHr": 0.49}
        calls = []
        def fake_call(*args, **kwargs):
            calls.append(args)
            return json.dumps({"id": "pod-1"})
        with patch.object(cloud, "choose_gpu", return_value=gpu), patch.object(cloud, "account", return_value={"clientBalance": 100}), patch.object(cloud, "ensure_network_volume", return_value={"id": "volume-1"}), patch.object(cloud, "_verify_capacity", return_value={"vcpus": 8, "host_ram_gb": 32}), patch.object(cloud, "_call", side_effect=fake_call):
            cloud.create_pod(cfg)
        create_args = next(args for args in calls if "pod" in args and "create" in args)
        deadline = datetime.fromisoformat(create_args[create_args.index("--terminate-after") + 1].replace("Z", "+00:00"))
        self.assertLess((deadline - datetime.now(timezone.utc)).total_seconds(), 65 * 60)

    def test_collection_rejects_stale_summary_without_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "summary.json").write_text("{}")
            with patch.object(cloud, "_ssh_info", return_value={"host": "example.invalid"}), patch.object(cloud.subprocess, "run"):
                with self.assertRaises(RuntimeError):
                    cloud.collect_remote_results("pod-1", "/remote", root)

    def test_cleanup_refuses_unowned_pods_and_volumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "collection_manifest.json"
            manifest.write_text('{"format":1,"experiment_id":"raid-study","files":[]}')
            receipt = root / "collection_receipt.json"
            receipt.write_text(json.dumps({"verified": True, "experiment_id": "raid-study",
                "manifest": str(manifest), "manifest_sha256": __import__("hashlib").sha256(
                    manifest.read_bytes()).hexdigest()}))
            with patch.object(cloud, "list_pods", return_value=[{"id": "other-pod", "name": "other-study"}]), \
                 patch.object(cloud, "list_network_volumes", return_value=[{"id": "other-volume", "name": "other-study"}]), \
                 patch.object(cloud, "_call") as provider_call:
                with self.assertRaisesRegex(ValueError, "another experiment"):
                    cloud.delete_pod("other-pod", "raid-study")
                with self.assertRaisesRegex(ValueError, "another experiment"):
                    cloud.delete_network_volume("other-volume", "raid-study", receipt_path=receipt)
                provider_call.assert_not_called()

    def test_libero_state_mapping_is_not_zero(self):
        state = _state({"robot0_eef_pos": [1, 2, 3], "robot0_eef_quat": [0, 0, 0, 1], "robot0_gripper_qpos": [.1, .2]})
        self.assertTrue(torch.isfinite(torch.from_numpy(state)).all())
        self.assertAlmostEqual(float(state[0]), 1.0)

    def test_cosmos_rejects_action_conditioned_future(self):
        predictor = Cosmos3EdgeFuturePredictor(lambda **_: FuturePrediction(torch.zeros(1, 4), 1.0, "x", True, {}))
        with self.assertRaises(ValueError):
            predictor.predict(torch.zeros(1, 3, 8, 8))

    def test_training_subset_is_seeded_and_nested(self):
        task = {"task": "task_a", "train": list(range(10))}
        low = set(select_training_ids(task, 2, 42)); high = set(select_training_ids(task, 5, 42))
        self.assertTrue(low.issubset(high))
        self.assertNotEqual(low, set(select_training_ids(task, 2, 43)))

    def test_report_lists_unrun_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "run.json").write_text(json.dumps({"method_registry": ["direct", "raid", "lora", "lora_raid"]}))
            report = build_report(root)
            self.assertEqual({row["condition"] for row in report["conditions"]}, {"direct", "raid", "lora", "lora_raid"})

    def test_lora_zero_initial_update(self):
        linear = nn.Linear(4, 3)
        wrapped = LowRankLinear(linear, rank=2, alpha=4)
        x = torch.randn(2, 4)
        torch.testing.assert_close(wrapped(x), linear(x))
        self.assertTrue(any(p.requires_grad for p in wrapped.parameters()))

    def test_mae_targets(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__(); self.qkv = nn.Linear(8, 8); self.proj = nn.Linear(8, 8)
        class Block(nn.Module):
            def __init__(self): super().__init__(); self.attn = Attention()
        class FakeMAE(nn.Module):
            def __init__(self): super().__init__(); self.blocks = nn.ModuleList([Block() for _ in range(12)])
        model = FakeMAE()
        targets = attach_mae_lora(model, (10, 11), 8, 16, 0.0)
        self.assertEqual(len(targets), 4)
        # Each of the four target projections has LoRA A and B parameters.
        self.assertEqual(sum(p.requires_grad for p in model.parameters()), 8)

    def test_raw_collate_preserves_transition_uid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = object.__new__(RawImageTransitions)
            raw.mean = np.zeros(7, dtype=np.float32); raw.std = np.ones(7, dtype=np.float32)
            raw.items = [{"image_t": np.zeros((4, 4, 3), dtype=np.uint8), "image_next": np.zeros((4, 4, 3), dtype=np.uint8),
                          "action_raw": np.zeros(7, dtype=np.float32), "task": "task_a", "demo_id": demo_uid("task_a", 2),
                          "timestep": 0, "transition_uid": "task_a:2:0"}]
            batch = raw[0]
            self.assertEqual(_raw_collate([batch])["transition_uid"], ["task_a:2:0"])

    def test_split_preserves_noncontiguous_demo_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / "task_a.hdf5"
            with h5py.File(path, "w") as handle:
                for demo in (2, 5, 9):
                    group = handle.create_group(f"data/demo_{demo}")
                    group.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32))
                    group.create_group("obs").create_dataset("agentview_rgb", data=np.zeros((4, 4, 4, 3), dtype=np.uint8))
            split = create_split([path], root / "split.json", expected_demos_per_task=3, validation_per_task=1, test_per_task=1)
            assigned = set(split["tasks"][0]["train"] + split["tasks"][0]["validation"] + split["tasks"][0]["test"])
            self.assertEqual(assigned, {2, 5, 9})

    def test_gripper_accuracy_uses_physical_units(self):
        metrics = action_metrics([[0.0] * 6 + [-1.0]], [[0.0] * 6 + [-3.0]], mean=[0] * 6 + [0.8], std=[1] * 6 + [0.6])
        self.assertEqual(metrics["gripper_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
