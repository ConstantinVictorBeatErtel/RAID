"""Regression checks for the new LIBERO-Goal protocol manifest."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from raid_experiments.fewshot import (
    FINAL_TEST_TASKS, DEVELOPMENT_TASKS, OFFICIAL_GOAL_TASKS, PROTOCOL,
    SOURCE_TASKS, create_goal_protocol, split_goal_task_names, validate_goal_protocol,
)
from raid_experiments.models import LowRankLinear, attach_mae_lora
from src.mae_encoder import MAEImageEncoder, MAE_IMPLEMENTATION, MAE_PREPROCESSING
from src.data_libero import LazyLiberoTransitionDataset
from raid_experiments.fewshot_config import load_fewshot_config


class FewShotProtocolTests(unittest.TestCase):
    def test_manifest_digest_is_stable_after_json_key_normalization(self):
        from raid_experiments.fewshot import _digest
        record = {"demo_lengths": {2: 17, 10: 23, 100: 31}, "path": "/ignored/location"}
        roundtrip = json.loads(json.dumps(record))
        self.assertEqual(_digest(record), _digest(roundtrip))

    def test_mae_encoder_returns_deterministic_normalized_cls(self):
        from functools import partial
        import timm
        mae = timm.models.vision_transformer.VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, num_classes=0, global_pool="",
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        encoder = MAEImageEncoder(mae, device="cpu")
        sample = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)
        first = encoder.encode_frames(sample)
        second = encoder.encode_frames(sample)
        self.assertEqual(tuple(first.shape), (1, 768))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(any(parameter.requires_grad for parameter in encoder.parameters()))

    def test_scientific_checkpoint_validator_rejects_nonfinite_and_incomplete_artifacts(self):
        from raid_experiments.fewshot_train import _validate_adaptation_checkpoint, _code_snapshot_digest, _config_digest, _scientific_config
        from raid_experiments.models import DirectDecoder
        identity={"method":"head_ft","task":"fixture","budget":5,"seed":42,
            "support_ids":[1,2,3,4,5],"protocol_digest":"p","source_checkpoint_sha256":"s",
            "source_feature_digest":"f","feature_digest":"g","code_snapshot_sha256":_code_snapshot_digest(),"selected_step":200}
        condition_config={"execution_mode":"scientific","method":"head_ft","task":"fixture",
            "budget":5,"seed":42,"updates":1000,"source_checkpoint_sha256":"s",
            "source_feature_digest":"f","mae_checkpoint_sha256":None,
            **{k:v for k,v in _scientific_config()["adaptation"].items()
               if k in {"batch_size","microbatch_size","head_lr","lora_lr","weight_decay","grad_clip",
                        "lora_rank","lora_alpha","lora_dropout"}},
            "lora_blocks":_scientific_config()["adaptation"]["target_blocks"]}
        payload={"checkpoint_format":4,"execution_mode":"scientific",**identity,
            "study_config_sha256":_config_digest(_scientific_config()),"source_feature_digest":"f",
            "condition_config":condition_config,"config_digest":_config_digest(condition_config),
            "architecture":{"feature_dim":768,"action_dim":7,"hidden_dim":512,"bottleneck_dim":256},
            "planned_updates":1000,"optimizer_updates":200,"condition_total_updates":1000,
            "condition_training_complete":True,"state_dict":DirectDecoder(768,7).state_dict()}
        _validate_adaptation_checkpoint(payload,identity,scientific=True)
        bad=dict(payload);bad["state_dict"]={key:value.clone() for key,value in payload["state_dict"].items()}
        next(iter(bad["state_dict"].values())).fill_(float("nan"))
        with self.assertRaisesRegex(ValueError,"non-finite head"):
            _validate_adaptation_checkpoint(bad,identity,scientific=True)
        old=dict(payload,checkpoint_format=2)
        with self.assertRaisesRegex(ValueError,"identity, version"):
            _validate_adaptation_checkpoint(old,identity,scientific=True)
        lora=dict(payload,method="lora_head_ft",adapter_config={"rank":8,"alpha":16,"dropout":0.,
            "blocks":[10,11],"projections":["qkv","proj"]},
            lora_state_dict={"blocks.10.attn.qkv.lora_B":torch.ones(1)},
            initial_lora_state_dict={"blocks.10.attn.qkv.lora_B":torch.zeros(1)})
        lora["condition_config"] = dict(condition_config, method="lora_head_ft")
        lora["config_digest"] = _config_digest(lora["condition_config"])
        lora_identity=dict(identity,method="lora_head_ft")
        with self.assertRaisesRegex(ValueError,"LoRA checkpoint"):
            _validate_adaptation_checkpoint(lora,lora_identity,scientific=True)

    def test_memory_chunking_matches_full_matrix_and_calibration_fails_closed(self):
        import tempfile
        from raid_experiments.retrieval import TransitionMemory
        from raid_experiments.fewshot_assets import calibrate_study
        from raid_experiments.fewshot_calibration import collect_calibration
        memory = TransitionMemory(2, 1, "cpu")
        current = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        future = torch.tensor([[0., 1.], [1., 0.], [1., -1.]])
        actions = torch.tensor([[1.], [2.], [3.]])
        memory.extend(current, future, actions, torch.tensor([10, 20, 30]))
        qcur = torch.tensor([[1., 0.], [0., 1.]])
        qnext = torch.tensor([[0., 1.], [1., 0.]])
        dense = memory.retrieve(qcur, qnext, k=2)
        blocked = memory.retrieve(qcur, qnext, k=2, query_chunk=1, bank_chunk=1)
        torch.testing.assert_close(blocked.similarities, dense.similarities)
        self.assertTrue(torch.equal(blocked.indices, dense.indices))
        exact=memory.retrieve(qcur[:1],qnext[:1],k=1)
        self.assertEqual(int(exact.indices[0,0]),0)
        self.assertAlmostEqual(float(exact.similarities[0,0]),1.0,places=6)

        config, _ = load_fewshot_config()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"smoke_real.json").write_text(json.dumps({
                "status":"passed","device":"cuda","cuda_verified":True,"stage_timings":{}}))
            feature=root/"features"; feature.mkdir()
            (feature/"manifest.json").write_text(json.dumps({"protocol_digest":"fixture",
                "mae_checkpoint_sha256":"mae",
                "identity_sha256":"cacheid",
                "measurements":{"device":"cuda","feature_cache_wall_seconds":1.0}}))
            pricing=root/"pricing.json";pricing.write_text(json.dumps({"verified":True,"source_url":"fixture",
                "gpu_hourly_usd":350.0,"storage_gb_month_usd":0.1}))
            cpu_calibration=root/"cpu_calibration"
            diagnostic=collect_calibration("missing-protocol","missing-cache","missing-mae",
                                           cpu_calibration,device="cpu")
            self.assertEqual(diagnostic["status"],"blocked")
            self.assertFalse(diagnostic["completion_receipt_written"])
            self.assertFalse((cpu_calibration/"calibration_completion.json").exists())
            blocked_report=calibrate_study(root,config,feature_dir=feature,pricing_file=pricing)
            self.assertEqual(blocked_report["status"],"blocked")
            self.assertIsNone(blocked_report["forecast"])
            bad_budget = dict(config, planning_budget_usd=251.0)
            rejected_budget = calibrate_study(root,bad_budget,feature_dir=feature,pricing_file=pricing)
            self.assertEqual(rejected_budget["status"],"blocked")
            self.assertIsNone(rejected_budget["forecast"])
            rejected_spend = calibrate_study(root,config,feature_dir=feature,pricing_file=pricing,
                                             spent_usd=float("nan"))
            self.assertEqual(rejected_spend["status"],"blocked")
            self.assertIsNone(rejected_spend["forecast"])

            from raid_experiments.fewshot_assets import _forecast_calibration_workload
            timings={key:1.0 for key in ("source_training_20_updates_seconds",
                "head_adaptation_20_updates_seconds","lora_adaptation_20_updates_seconds",
                "feature_cache_wall_seconds","development_baseline_bundle_seconds",
                "development_candidate_bundle_seconds","confirmation_bundle_seconds",
                "retrieval_bank_build_seconds","retention_pass_seconds",
                "checkpoint_io_bundle_seconds","recovery_seconds_per_adaptation",
                "peak_memory_bytes","total_artifact_storage_bytes")}
            budget={"planning_budget_usd":250.,"protected_reserve_usd":25.,"workload_margin":.25}
            prices={"gpu_hourly_usd":1.,"storage_gb_month_usd":.1}
            all_scope=_forecast_calibration_workload(timings,prices,budget,0.,scope="all")
            self.assertAlmostEqual(all_scope["source_training_hours"],100*3/3600)
            self.assertEqual(all_scope["condition_matrix"]["development_snapshot_evaluations"],36)
            self.assertEqual(all_scope["condition_matrix"]["confirmation_conditions"],18)
            self.assertEqual(all_scope["condition_matrix"]["canonical_evaluation_conditions"],24)
            self.assertEqual(all_scope["condition_matrix"]["adaptation_runs_per_method"],24)
            self.assertEqual(all_scope["condition_matrix"]["development_retention_passes"],42)
            self.assertEqual(all_scope["condition_matrix"]["confirmation_retention_passes"],54)
            self.assertEqual(all_scope["condition_matrix"]["retention_passes"],96)
            self.assertEqual(all_scope["condition_matrix"]["retrieval_bank_builds"],48)
            self.assertAlmostEqual(all_scope["workload_hours_with_margin"],
                (300 + 2400 + 1 + 60 + 48 + 96 + 51 + 48) / 3600 * 1.25)
            spent_scope=_forecast_calibration_workload(timings,prices,budget,30.,scope="all")
            self.assertEqual(spent_scope["available_before_protected_reserve_usd"],195.)
            self.assertLess(spent_scope["total_cost_usd"],195.)
            five_scope=_forecast_calibration_workload(timings,prices,budget,0.,scope="five_shot")
            self.assertEqual(five_scope["condition_matrix"]["confirmation_conditions"],6)
            self.assertEqual(five_scope["condition_matrix"]["canonical_evaluation_conditions"],12)
            self.assertEqual(five_scope["condition_matrix"]["adaptation_runs_per_method"],12)
            self.assertEqual(five_scope["condition_matrix"]["development_retention_passes"],42)
            self.assertEqual(five_scope["condition_matrix"]["confirmation_retention_passes"],18)
            self.assertEqual(five_scope["condition_matrix"]["retention_passes"],60)
            self.assertEqual(five_scope["condition_matrix"]["retrieval_bank_builds"],24)
            self.assertAlmostEqual(five_scope["workload_hours_with_margin"],
                (300 + 1200 + 1 + 48 + 24 + 60 + 27 + 24) / 3600 * 1.25)

    def test_hand_calculated_demo_macro_error_and_paired_bootstrap(self):
        from raid_experiments.fewshot_report import _mse_from_predictions, _paired_bootstrap
        data={"predictions":torch.tensor([[1.]*7,[2.]*7,[3.]*7]),
            "targets":torch.zeros(3,7),"transition_ids":["t:10:0","t:10:1","t:11:0"]}
        self.assertAlmostEqual(_mse_from_predictions(data),(2.5+9.0)/2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); records={}; protocol={"task_roles":{"development":list(DEVELOPMENT_TASKS)}}
            for task in DEVELOPMENT_TASKS:
                base=root/task/"5shot"/"seed42";base.mkdir(parents=True)
                ids=[f"{task}:1:0",f"{task}:1:1",f"{task}:2:0"]
                targets=torch.zeros(3,7)
                left=torch.zeros_like(targets); right=torch.ones_like(targets)
                artifacts={}
                for method,pred in (("left",left),("right",right)):
                    path=base/f"{method}.pt"
                    torch.save({"predictions":pred,"targets":targets,"transition_ids":ids},path)
                    artifacts[method]={"path":path.name}
                records[(task,5,42)]={"prediction_artifacts":artifacts}
            result=_paired_bootstrap(root,records,protocol,"development",5,"left","right",draws=100)
            self.assertEqual(result["estimate"],-1.0)
            self.assertEqual(result["lower_95"],-1.0)
            self.assertEqual(result["upper_95"],-1.0)

    def test_active_config_loads_and_rejects_paid_launch(self):
        config, digest = load_fewshot_config()
        self.assertEqual(config["protocol"], PROTOCOL)
        self.assertEqual(len(digest), 64)
        self.assertIs(config["cloud"]["launch_enabled"], False)
        with tempfile.TemporaryDirectory() as temp:
            changed = dict(config)
            changed["cloud"] = {"launch_enabled": True}
            path = Path(temp) / "unsafe.yaml"
            import yaml
            path.write_text(yaml.safe_dump(changed))
            with self.assertRaisesRegex(ValueError, "paid launch disabled"):
                load_fewshot_config(path)

    def _dataset(self, root: Path, *, unequal_lengths: bool = False) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        for index, task in enumerate(OFFICIAL_GOAL_TASKS):
            task_dir = root / "libero_goal"
            task_dir.mkdir(parents=True, exist_ok=True)
            path = task_dir / f"{task}_demo.hdf5"
            with h5py.File(path, "w") as handle:
                for demo_id in range(100, 150):
                    group = handle.create_group(f"data/demo_{demo_id}")
                    action_count = 2 + demo_id % 3 if unequal_lengths else 2
                    group.create_dataset("actions", data=np.full((action_count, 7), demo_id, dtype=np.float32))
                    group.create_group("obs").create_dataset(
                        "agentview_rgb", data=np.stack([np.full((4, 4, 3), demo_id + frame, dtype=np.uint8)
                                                         for frame in range(action_count + 1 if unequal_lengths else 2)])
                    )
        return root

    def test_task_assignment_is_fixed_and_partitions_official_suite(self):
        roles = split_goal_task_names()
        self.assertEqual(tuple(roles["source"]), SOURCE_TASKS)
        self.assertEqual(tuple(roles["development"]), DEVELOPMENT_TASKS)
        self.assertEqual(tuple(roles["final_test"]), FINAL_TEST_TASKS)
        flat = roles["source"] + roles["development"] + roles["final_test"]
        self.assertEqual(len(flat), 10)
        self.assertEqual(set(flat), set(OFFICIAL_GOAL_TASKS))

    def test_lora_updates_only_adapters_and_roundtrips(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__(); self.qkv = nn.Linear(8, 8); self.proj = nn.Linear(8, 8)
        class Block(nn.Module):
            def __init__(self):
                super().__init__(); self.attn = Attention()
        class FakeMAE(nn.Module):
            def __init__(self):
                super().__init__(); self.blocks = nn.ModuleList([Block() for _ in range(12)])
            def forward(self, x):
                for block in self.blocks:
                    x = block.attn.proj(block.attn.qkv(x))
                return x, None

        torch.manual_seed(5); model = FakeMAE()
        names = attach_mae_lora(model, (10, 11), 2, 4, 0.0)
        self.assertEqual(len(names), 4)
        base_before = {n: p.detach().clone() for n, p in model.named_parameters() if ".base." in n}
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 8)
        adapter_before = {n: p.detach().clone() for n, p in trainable}
        optimizer = torch.optim.SGD([p for _, p in trainable], lr=0.1)
        x = torch.randn(3, 8); target = torch.randn(3, 8)
        for _ in range(3):
            optimizer.zero_grad(); loss = (model(x)[0] - target).pow(2).mean(); loss.backward(); optimizer.step()
        self.assertTrue(any(not torch.equal(p.detach(), adapter_before[n]) for n, p in trainable))
        for name, old in base_before.items():
            torch.testing.assert_close(dict(model.named_parameters())[name], old)
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        clone = FakeMAE(); attach_mae_lora(clone, (10, 11), 2, 4, 0.0); clone.load_state_dict(saved)
        torch.testing.assert_close(model(x)[0], clone(x)[0])

    def test_lora_initialization_is_seed_reproducible(self):
        layer = nn.Linear(4, 4)
        torch.manual_seed(123); first = LowRankLinear(layer, rank=2)
        torch.manual_seed(123); second = LowRankLinear(layer, rank=2)
        torch.testing.assert_close(first.lora_A, second.lora_A, rtol=0, atol=0)
        self.assertEqual(first.lora_A.device, layer.weight.device)
        self.assertEqual(first.lora_A.dtype, layer.weight.dtype)

    def test_lazy_hdf5_reader_preserves_demo_boundaries_and_action_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = self._dataset(root / "data")
            path = data / "libero_goal" / f"{OFFICIAL_GOAL_TASKS[0]}_demo.hdf5"
            dataset = LazyLiberoTransitionDataset(path, {"action_mean": torch.zeros(7), "action_std": torch.ones(7)})
            self.assertEqual(len(dataset), 50)
            first = dataset[0]; second = dataset[1]
            self.assertEqual((first["demo_id"], first["timestep"]), (100, 0))
            self.assertEqual((second["demo_id"], second["timestep"]), (101, 0))
            self.assertEqual(float(first["action_raw"][0]), 100.0)
            self.assertEqual(int(first["image_t"][0, 0, 0]), 100)
            self.assertEqual(int(first["image_next"][0, 0, 0]), 101)

    def test_lightweight_mae_lora_image_path_backpropagates_only_into_adapters(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__(); self.qkv = nn.Linear(8, 8); self.proj = nn.Linear(8, 8)
        class Block(nn.Module):
            def __init__(self):
                super().__init__(); self.attn = Attention()
        class TinyMAE(nn.Module):
            def __init__(self):
                super().__init__(); self.input = nn.Linear(3, 8); self.blocks = nn.ModuleList([Block() for _ in range(12)])
                self.output = nn.Linear(8, 768)
            def forward(self, pixels):
                state = self.input(pixels.mean(dim=(2, 3)))
                for block in self.blocks:
                    state = state + block.attn.proj(block.attn.qkv(state))
                return self.output(state), None

        torch.manual_seed(9)
        encoder = MAEImageEncoder(TinyMAE(), "cpu")
        attach_mae_lora(encoder.mae, (10, 11), 2, 4, 0.0)
        base_before = {k: p.detach().clone() for k, p in encoder.mae.state_dict().items() if "lora_" not in k}
        adapters = [p for p in encoder.mae.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(adapters, lr=0.05)
        images = torch.randint(0, 256, (2, 24, 24, 3), dtype=torch.uint8)
        first = encoder.encode_frames(images)
        optimizer.zero_grad(); first.pow(2).mean().backward(); optimizer.step()
        second = encoder.encode_frames(images)
        self.assertFalse(torch.equal(first, second))
        actual = encoder.mae.state_dict()
        for key, value in base_before.items():
            wrapped = key.replace(".qkv.", ".qkv.base.").replace(".proj.", ".proj.base.")
            self.assertTrue(torch.equal(value, actual[wrapped if wrapped in actual else key]), key)
        self.assertTrue(all(not p.requires_grad for k,p in encoder.mae.named_parameters() if "lora_" not in k))
        self.assertEqual(sum(p.requires_grad for p in encoder.mae.parameters()), 8)

    def test_manifest_has_disjoint_roles_real_ids_and_nested_support(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = self._dataset(root / "data")
            manifest_path = root / "protocol.json"
            manifest = create_goal_protocol(data, manifest_path)
            self.assertEqual(manifest["protocol"], PROTOCOL)
            self.assertEqual(len(manifest["digest"]), 64)
            source = manifest["tasks"][SOURCE_TASKS[0]]["source"]
            self.assertEqual([len(source[k]) for k in ("source_train", "source_validation", "source_retention")], [40, 5, 5])
            target = manifest["tasks"][DEVELOPMENT_TASKS[0]]
            self.assertEqual(set(target["query"]) | set(target["support_pool"]), set(range(100, 150)))
            self.assertFalse(set(target["query"]) & set(target["support_pool"]))
            self.assertTrue(set(target["support_budgets"]["2"]) <= set(target["support_budgets"]["5"]))
            self.assertTrue(set(target["support_budgets"]["5"]) <= set(target["support_budgets"]["10"]))
            self.assertEqual(validate_goal_protocol(manifest_path)["digest"], manifest["digest"])

    def test_manifest_refuses_source_mutation_and_assignment_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = self._dataset(root / "data")
            manifest_path = root / "protocol.json"
            create_goal_protocol(data, manifest_path)
            first = next(data.rglob("*.hdf5"))
            with first.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                validate_goal_protocol(manifest_path)
            with self.assertRaises(ValueError):
                create_goal_protocol(data, root / "other.json", expected_demos=49)

    def test_digest_valid_but_protocol_invalid_horizon_and_duplicate_support_fail(self):
        from raid_experiments.fewshot import _digest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = self._dataset(root / "data")
            path = root / "protocol.json"; original = create_goal_protocol(data, path)
            invalid = json.loads(json.dumps(original)); invalid["horizon"] = 3; invalid["digest"] = _digest(invalid)
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "constants"):
                validate_goal_protocol(path)
            invalid = json.loads(json.dumps(original)); task = DEVELOPMENT_TASKS[0]
            selected = invalid["tasks"][task]["support_budgets_by_seed"]["42"]["2"]
            selected[1] = selected[0]; invalid["digest"] = _digest(invalid)
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "support selections"):
                validate_goal_protocol(path)
            invalid = json.loads(json.dumps(original)); source_task = SOURCE_TASKS[0]
            source = invalid["tasks"][source_task]["source"]
            source["source_train"][0], source["source_validation"][0] = source["source_validation"][0], source["source_train"][0]
            invalid["digest"] = _digest(invalid); path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "deterministic protocol allocation"):
                validate_goal_protocol(path)

    def test_development_freeze_rejects_mismatched_metrics_and_is_idempotent(self):
        from raid_experiments.fewshot_report import freeze_development, register_confirmation, METHODS as REPORT_METHODS
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data=self._dataset(root/"data"); protocol_path=root/"protocol.json"
            protocol=create_goal_protocol(data,protocol_path); output=root/"runs"
            source=root/"source42.pt"
            from raid_experiments.models import DirectDecoder
            from raid_experiments.fewshot_train import _code_snapshot_digest, _config_digest, _scientific_config
            source_config={"execution_mode":"scientific","seed":42,"updates":2000,"hidden":512,
                "bottleneck":256,"batch":64,"microbatch":8,"lr":0.0003,"weight_decay":0.0001}
            torch.save({"checkpoint_format":4,"study_config_sha256":_config_digest(_scientific_config()),
                "code_snapshot_sha256":_code_snapshot_digest(),"execution_mode":"scientific","scientific_complete":True,
                "training_updates":2000,"planned_updates":2000,"training_complete":True,
                "protocol":PROTOCOL,"protocol_digest":protocol["digest"],"seed":42,
                "feature_digest":("f"*64),"source_feature_digest":("f"*64),
                "source_config":source_config,"config_digest":_config_digest(source_config),
                "study_root":str(root.resolve()),"best_step":2000,"best_validation_mse":0.5,
                "curve":[{"step":2000,"validation_mse":0.5}],
                "architecture":{"feature_dim":768,"action_dim":7,"hidden_dim":512,"bottleneck_dim":256},
                "state_dict":DirectDecoder(768,7).state_dict(),"normalizer_mean":torch.zeros(7),
                "normalizer_std":torch.ones(7)},source)
            source_hash=hashlib.sha256(source.read_bytes()).hexdigest(); mean=torch.zeros(7); std=torch.ones(7)
            source_receipt={"evidence_format":4,"status":"complete","checkpoint_sha256":source_hash,
                "protocol_digest":protocol["digest"],"seed":42,"source_feature_digest":"f"*64,
                "code_snapshot_sha256":_code_snapshot_digest(),"config_digest":_config_digest(source_config),
                "planned_updates":2000,"training_updates":2000,"best_step":2000,
                "best_validation_mse":0.5,"execution_mode":"scientific"}
            (root/"source_seed42.complete.json").write_text(json.dumps(source_receipt))
            norm_sha=hashlib.sha256(mean.numpy().tobytes()+std.numpy().tobytes()).hexdigest()
            controls=set(REPORT_METHODS)-{"head_ft","lora_head_ft"}
            for task in DEVELOPMENT_TASKS:
                spec=protocol["tasks"][task]; query=[f"{task}:{d}:0" for d in spec["query"]]
                targets=torch.tensor([[int(tid.rsplit(":",2)[-2])]*7 for tid in query],dtype=torch.float32)
                for budget in (2,5,10):
                    base=output/task/f"{budget}shot"/"seed42";base.mkdir(parents=True)
                    supports=list(map(int,spec["support_budgets_by_seed"]["42"][str(budget)]))
                    def write_report(name, methods):
                        if name != "evaluation_baselines":
                            methods={method:score for method,score in methods.items()
                                     if method in ("head_ft","lora_head_ft")}
                        artifacts={}; scores={}
                        for method,score in methods.items():
                            pred=targets+float(score)**0.5
                            selected_update=None;adaptation_sha=None
                            if method in ("head_ft","lora_head_ft"):
                                step=int(name.rsplit("head",1)[-1] if method=="head_ft" else name.rsplit("lora",1)[-1])
                                fixed=_scientific_config()["adaptation"]
                                condition_config={"execution_mode":"scientific","method":method,"task":task,
                                    "budget":budget,"seed":42,"updates":1000,
                                    "source_checkpoint_sha256":source_hash,"source_feature_digest":"f"*64,
                                    "mae_checkpoint_sha256":None,"batch_size":fixed["batch_size"],
                                    "microbatch_size":fixed["microbatch_size"],"head_lr":fixed["head_lr"],
                                    "lora_lr":fixed["lora_lr"],"weight_decay":fixed["weight_decay"],
                                    "grad_clip":fixed["grad_clip"],"lora_rank":fixed["lora_rank"],
                                    "lora_alpha":fixed["lora_alpha"],"lora_dropout":fixed["lora_dropout"],
                                    "lora_blocks":fixed["target_blocks"]}
                                checkpoint={"checkpoint_format":4,"execution_mode":"scientific",
                                    "protocol_digest":protocol["digest"],"method":method,"task":task,
                                    "budget":budget,"seed":42,"selected_step":step,"source_checkpoint_sha256":source_hash,
                                    "support_ids":supports,
                                    "source_feature_digest":("f"*64),"feature_digest":("g"*64),"code_snapshot_sha256":_code_snapshot_digest(),
                                    "study_config_sha256":_config_digest(_scientific_config()),
                                    "condition_config":condition_config,
                                    "config_digest":_config_digest(condition_config),
                                    "planned_updates":1000,"optimizer_updates":step,
                                    "condition_training_complete":True,"condition_total_updates":1000,
                                    "architecture":{"feature_dim":768,"action_dim":7,"hidden_dim":512,"bottleneck_dim":256},
                                    "state_dict":DirectDecoder(768,7).state_dict()}
                                if method == "lora_head_ft":
                                    checkpoint["adapter_config"]={"rank":8,"alpha":16,"dropout":0.0,
                                        "blocks":[10,11],"projections":["qkv","proj"]}
                                    adapter={}
                                    for block in (10,11):
                                        for projection,outdim in (("qkv",2304),("proj",768)):
                                            adapter[f"blocks.{block}.attn.{projection}.lora_A"]=torch.zeros(8,768)
                                            adapter[f"blocks.{block}.attn.{projection}.lora_B"]=torch.ones(outdim,8)
                                    checkpoint["lora_state_dict"]=adapter
                                    checkpoint["initial_lora_state_dict"]={key:torch.zeros_like(value) for key,value in adapter.items()}
                                checkpoint_path=base/f"{method}.step{step}.pt";torch.save(checkpoint,checkpoint_path)
                                adaptation_sha=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest();selected_update=step
                            payload={"prediction_format":4,"execution_mode":"scientific","predictions":pred,"targets":targets,"transition_ids":query,
                                "protocol_digest":protocol["digest"],"source_checkpoint_sha256":source_hash,
                                "source_feature_digest":"f"*64,"feature_digest":"g"*64,
                                "code_snapshot_sha256":_code_snapshot_digest(),
                                "study_config_sha256":_config_digest(_scientific_config()),
                                "support_ids":supports,"normalizer_mean":mean,"normalizer_std":std,
                                "method":method,"task":task,"budget":budget,"seed":42,
                                "selected_update":selected_update,"adaptation_artifact_sha256":adaptation_sha}
                            file=base/f"{name}_{method}.pt";torch.save(payload,file)
                            artifacts[method]={"path":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
                            scores[method]={"hierarchical_source_normalized_mse":score}
                            if selected_update is not None:
                                scores[method]["selected_update"]=selected_update
                                scores[method]["adaptation_config_digest"]=_config_digest(condition_config)
                        selected={m:scores[m]["selected_update"] for m in scores if m in ("head_ft","lora_head_ft")}
                        eval_config={"protocol_digest":protocol["digest"],"task":task,"budget":budget,
                            "seed":42,"methods":sorted(methods),"selected_updates":selected,
                            "source_checkpoint_sha256":source_hash,"source_feature_digest":"f"*64,
                            "feature_digest":"g"*64,"code_snapshot_sha256":_code_snapshot_digest()}
                        eval_digest=_config_digest(eval_config)
                        for method in methods:
                            # Bind each prediction to the exact evaluation call.
                            payload_path=base/f"{name}_{method}.pt"
                            payload=torch.load(payload_path,map_location="cpu",weights_only=False)
                            payload["evaluation_config"]=eval_config
                            payload["evaluation_config_sha256"]=eval_digest
                            torch.save(payload,payload_path)
                            artifacts[method]["sha256"]=hashlib.sha256(payload_path.read_bytes()).hexdigest()
                        item={"evidence_format":4,"status":"complete","execution_mode":"scientific",
                            "protocol_digest":protocol["digest"],"task":task,
                            "role":"development","budget":budget,"seed":42,"source_checkpoint_sha256":source_hash,
                            "normalizer_sha256":norm_sha,"source_feature_digest":("f"*64),"feature_digest":("g"*64),
                            "code_snapshot_sha256":_code_snapshot_digest(),"support_ids":supports,"query_ids":query,
                            "selected_updates":selected,
                            "evaluation_config_sha256":eval_digest,
                            "prediction_config_sha256":{method:eval_digest for method in methods},
                            "methods":scores,"prediction_artifacts":artifacts}
                        if name != "evaluation_baselines":
                            baseline_item=json.loads((base/"evaluation_baselines.json").read_text())
                            item["baseline_evidence"]={"evaluation_sha256":hashlib.sha256(
                                (base/"evaluation_baselines.json").read_bytes()).hexdigest(),
                                "prediction_artifact_sha256":{m:baseline_item["prediction_artifacts"][m]["sha256"]
                                    for m in sorted(controls)}}
                        (base/f"{name}.json").write_text(json.dumps(item))
                    write_report("evaluation_baselines",{m:.5 for m in controls})
                    for step in (200,500,1000):
                        write_report(f"evaluation_head{step}",{**{m:.5 for m in controls},"head_ft":.1 if step==500 else .2})
                        write_report(f"evaluation_lora{step}",{**{m:.5 for m in controls},"lora_head_ft":.1 if step==1000 else .3})
                    for method in ("head_ft", "lora_head_ft"):
                        fixed=_scientific_config()["adaptation"]
                        condition_config={"execution_mode":"scientific","method":method,"task":task,
                            "budget":budget,"seed":42,"updates":1000,"source_checkpoint_sha256":source_hash,
                            "source_feature_digest":"f"*64,"mae_checkpoint_sha256":None,
                            "batch_size":fixed["batch_size"],"microbatch_size":fixed["microbatch_size"],
                            "head_lr":fixed["head_lr"],"lora_lr":fixed["lora_lr"],
                            "weight_decay":fixed["weight_decay"],"grad_clip":fixed["grad_clip"],
                            "lora_rank":fixed["lora_rank"],"lora_alpha":fixed["lora_alpha"],
                            "lora_dropout":fixed["lora_dropout"],"lora_blocks":fixed["target_blocks"]}
                        marker={"evidence_format":4,"status":"complete","method":method,"task":task,
                            "budget":budget,"seed":42,"protocol_digest":protocol["digest"],
                            "source_checkpoint_sha256":source_hash,"source_feature_digest":("f"*64),"feature_digest":("g"*64),
                            "code_snapshot_sha256":_code_snapshot_digest(),"planned_updates":1000,
                            "config_digest":_config_digest(condition_config),
                            "snapshot_sha256":{f"{method}.step{step}.pt":hashlib.sha256(
                                (base/f"{method}.step{step}.pt").read_bytes()).hexdigest() for step in (200,500,1000)}}
                        (base/f"{method}.complete.json").write_text(json.dumps(marker))
                        (base/f"{method}.json").write_text(json.dumps({"status":"complete",
                            "execution_mode":"scientific","method":method,"task":task,"budget":budget,
                            "seed":42,"protocol_digest":protocol["digest"],"source_checkpoint_sha256":source_hash,
                            "source_feature_digest":("f"*64),"feature_digest":("g"*64),"code_snapshot_sha256":_code_snapshot_digest(),
                            "planned_updates":1000,"updates":1000,"config_digest":_config_digest(condition_config)}))
            freeze_path=output/"development_freeze.json"
            freeze=freeze_development(protocol_path,output,source,None,freeze_path)
            self.assertEqual(freeze["selected_updates"],{"head_ft":500,"lora_head_ft":1000})
            self.assertEqual(freeze_development(protocol_path,output,source,None,freeze_path),freeze)
            source_paths={42:source}
            base_source=torch.load(source,map_location="cpu",weights_only=False)
            for seed in (43,44):
                source_payload=dict(base_source);source_payload["seed"]=seed
                source_payload["source_config"]=dict(base_source["source_config"],seed=seed)
                source_payload["config_digest"]=_config_digest(source_payload["source_config"])
                path=root/f"source{seed}.pt";torch.save(source_payload,path);source_paths[seed]=path
                receipt=dict(source_receipt,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=seed,
                    config_digest=source_payload["config_digest"])
                (root/f"source_seed{seed}.complete.json").write_text(json.dumps(receipt))
            confirmation=register_confirmation(protocol_path,freeze_path,source_paths,
                                                root/"confirmation.json",scope="five_shot")
            self.assertEqual(confirmation["confirmation_scope"]["support_budgets"],[5])
            self.assertEqual(confirmation["study_root"],str(root.resolve()))
            self.assertEqual(register_confirmation(protocol_path,freeze_path,source_paths,
                root/"confirmation.json",scope="five_shot"),confirmation)
            self.assertEqual(register_confirmation(protocol_path,freeze_path,source_paths,
                root/"relocated"/"confirmation.json",scope="five_shot"),confirmation)
            (root/"final_query_access.json").write_text(json.dumps({"protocol_digest":protocol["digest"],
                "evidence_format":4,"conditions":[{"task":"already-opened"}]}))
            self.assertEqual(register_confirmation(protocol_path,freeze_path,source_paths,
                root/"confirmation.json",scope="five_shot"),confirmation)
            self.assertEqual(register_confirmation(protocol_path,freeze_path,source_paths,
                root/"relocated-after-access"/"confirmation.json",scope="five_shot"),confirmation)
            with self.assertRaisesRegex(FileExistsError,"confirmation registry"):
                register_confirmation(protocol_path,freeze_path,source_paths,
                    root/"changed-confirmation.json",scope="all")
            bad_sources=dict(source_paths);bad_sources[44]=source_paths[43]
            with self.assertRaisesRegex(ValueError,"seed 44"):
                register_confirmation(protocol_path,freeze_path,bad_sources,root/"bad-confirmation.json")

            from raid_experiments.fewshot_report import report_fewshot
            canonical=output/DEVELOPMENT_TASKS[0]/"2shot"/"seed42"/"evaluation.json"
            item=json.loads(canonical.read_text());item["execution_mode"]="calibration";item["evidence_format"]=2
            canonical.write_text(json.dumps(item))
            report=report_fewshot(output,protocol_path)
            self.assertEqual(report["valid_evaluation_artifacts"],5)
            self.assertTrue(any("unsupported evidence format" in row["error"] for row in report["invalid_or_failed_artifacts"]))
            item["execution_mode"]="scientific";item["evidence_format"]=3;item["status"]="calibration"
            canonical.write_text(json.dumps(item))
            report=report_fewshot(output,protocol_path)
            self.assertEqual(report["valid_evaluation_artifacts"],5)
            self.assertTrue(any("calibration" in row["error"] for row in report["invalid_or_failed_artifacts"]))

            # Corrupt one candidate's score while preserving its prediction file hash.
            candidate=output/DEVELOPMENT_TASKS[0]/"2shot"/"seed42"/"evaluation_head200.json"
            item=json.loads(candidate.read_text());item["methods"]["head_ft"]["hierarchical_source_normalized_mse"]=999
            candidate.write_text(json.dumps(item))
            with self.assertRaisesRegex(ValueError,"reported metric disagrees"):
                freeze_development(protocol_path,output,source,None,root/"bad-freeze.json")
            item=json.loads(candidate.read_text());item["methods"]["head_ft"]["hierarchical_source_normalized_mse"]=.2
            prediction_record=item["prediction_artifacts"]["head_ft"]
            prediction_path=candidate.parent/prediction_record["path"]
            prediction=torch.load(prediction_path,map_location="cpu",weights_only=False)
            prediction["targets"]=prediction["targets"]+1
            torch.save(prediction,prediction_path)
            prediction_record["sha256"]=hashlib.sha256(prediction_path.read_bytes()).hexdigest()
            candidate.write_text(json.dumps(item))
            with self.assertRaisesRegex(ValueError,"HDF5 evaluation labels"):
                freeze_development(protocol_path,output,source,None,root/"false-target-freeze.json")

    def test_synthetic_source_adaptation_retrieval_and_reporting_path(self):
        from raid_experiments.fewshot_train import train_source, adapt_target, evaluate_task, RawSelectedRows

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = self._dataset(root / "data", unequal_lengths=True)
            protocol_path = root / "protocol.json"
            protocol = create_goal_protocol(data, protocol_path)
            self.assertGreater(len(set(protocol["tasks"][DEVELOPMENT_TASKS[0]]["transitions"].values())), 1)
            import hashlib
            from types import SimpleNamespace
            from unittest.mock import patch
            from raid_experiments.fewshot import LIBERO_COMMIT
            from src.cache_mae_features import cache_protocol_tasks

            class FixtureEncoder:
                def encode_frames(self, images):
                    pixels = images.float()
                    if images.dtype == torch.uint8:
                        pixels = pixels / 255.0
                    level = pixels.mean(dim=(1, 2, 3))
                    basis = torch.linspace(0.1, 1.0, 768, dtype=torch.float32).expand(len(images), -1).clone()
                    basis[:, 0] += level
                    return torch.nn.functional.normalize(basis, dim=-1)

            fixture_mae = root / "fixture-mae.pth"
            fixture_mae.write_bytes(b"synthetic fixture encoder; never scientific")
            fixture_repo = root / "pinned-libero-checkout"
            fixture_repo.mkdir()

            def fixture_git(command, **kwargs):
                output = LIBERO_COMMIT if command[-2:] == ["rev-parse", "HEAD"] else ""
                return SimpleNamespace(stdout=output)

            features = root / "features"
            with patch("src.cache_mae_features.subprocess.run", side_effect=fixture_git):
                cache_manifest = cache_protocol_tasks(protocol_path, data, features, fixture_mae,
                    fixture_repo, batch_size=8, execution_mode="synthetic",
                    encoder_factory=lambda _checkpoint, _device: FixtureEncoder())
            shard_index = cache_manifest["demo_shards"]
            identity = {key: cache_manifest[key] for key in ("execution_mode", "horizon",
                "future_representation", "mae_checkpoint_sha256", "mae_implementation",
                "preprocessing", "preprocessing_version", "action_convention", "feature_dim",
                "libero_commit")}
            identity_sha = cache_manifest["identity_sha256"]
            original_mae_sha = cache_manifest["mae_checkpoint_sha256"]
            from raid_experiments.fewshot_train import _load_cache_map
            with self.assertRaisesRegex(ValueError,"scientific execution requires a scientific feature cache"):
                _load_cache_map(features,protocol,run_mode="scientific")
            cache_manifest["mae_checkpoint_sha256"] = "changed-encoder"
            (features / "manifest.json").write_text(json.dumps(cache_manifest))
            with self.assertRaisesRegex(ValueError,"identity digest"):
                _load_cache_map(features,protocol)
            changed_identity=dict(identity);changed_identity["mae_checkpoint_sha256"]="changed-encoder"
            cache_manifest["mae_checkpoint_sha256"]="changed-encoder"
            cache_manifest["identity_sha256"]=hashlib.sha256(json.dumps(changed_identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            (features / "manifest.json").write_text(json.dumps(cache_manifest))
            with self.assertRaisesRegex(ValueError,"shard identity"):
                _load_cache_map(features,protocol)
            cache_manifest["mae_checkpoint_sha256"] = original_mae_sha
            cache_manifest["identity_sha256"] = identity_sha
            (features / "manifest.json").write_text(json.dumps(cache_manifest))
            from raid_experiments.fewshot_assets import audit_complete_feature_cache
            complete_audit=audit_complete_feature_cache(features,protocol)
            self.assertEqual(complete_audit["checked_shards"],500)
            self.assertEqual(complete_audit["query_shards"],80)
            from raid_experiments.fewshot_train import _load_cache_map
            digest_task=DEVELOPMENT_TASKS[0]
            source_cache,source_digest=_load_cache_map(features,protocol)
            _,condition_digest=_load_cache_map(features,protocol,target_task=digest_task,budget=2,seed=42)
            self.assertEqual(source_cache[digest_task]["_source_feature_digest"],source_digest)
            self.assertNotEqual(source_digest,condition_digest)
            # Query shards are physically label-free, so training cannot load sentinel labels.
            query_demo = str(protocol["tasks"][FINAL_TEST_TASKS[0]]["query"][0])
            query_record = shard_index[FINAL_TEST_TASKS[0]][query_demo]
            query_shard = torch.load(features / query_record["path"], map_location="cpu", weights_only=False)
            self.assertNotIn("actions_raw", query_shard)

            source_dir = root / "source"
            source_timing={"warmup_updates":1,"measured_update_seconds":[]}
            trained = train_source(protocol_path, features, source_dir, seed=42, updates=2,
                                   batch_size=8, microbatch_size=4, device="cpu", run_mode="synthetic",
                                   timing_profile=source_timing)
            self.assertEqual(trained["status"], "synthetic")
            self.assertEqual(len(source_timing["measured_update_seconds"]),1)
            self.assertGreater(source_timing["measured_update_seconds"][0],0)
            source_ckpt = source_dir / "source_seed42.pt"
            source_snapshot_bytes=source_ckpt.read_bytes()
            with self.assertRaisesRegex(FileExistsError,"immutable snapshot"):
                train_source(protocol_path,features,source_dir,seed=42,updates=2,hidden=256,
                    batch_size=8,microbatch_size=4,device="cpu",run_mode="synthetic")
            self.assertEqual(source_ckpt.read_bytes(),source_snapshot_bytes)
            with self.assertRaisesRegex(ValueError, "completed 2,000-update scientific"):
                from raid_experiments.fewshot_train import _load_cache_map, _resolve_device, _load_source
                _load_source(source_ckpt, protocol, _load_cache_map(features, protocol)[0],
                             _resolve_device("cpu"), run_mode="scientific")
            self.assertTrue(source_ckpt.is_file())
            # Exercise the actual Python signal handlers at saved optimizer
            # boundaries, rather than only passing stop_after_updates.
            import os, signal, subprocess, sys, time
            repo=Path(__file__).resolve().parents[1]
            def signal_at_checkpoint(command, args, checkpoint_path, signum):
                process=subprocess.Popen([sys.executable,"-c",command,*map(str,args)],cwd=repo,
                    env={**os.environ,"PYTHONPATH":str(repo)},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                deadline=time.time()+90
                try:
                    while time.time()<deadline and process.poll() is None:
                        if checkpoint_path.is_file():
                            payload=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
                            if int(payload.get("step",payload.get("step",0))) >= 100:
                                os.kill(process.pid,signum)
                                stdout,stderr=process.communicate(timeout=30)
                                self.assertEqual(process.returncode,0,stderr[-2000:])
                                return payload
                        time.sleep(.02)
                    stdout,stderr=process.communicate(timeout=5)
                    self.fail(f"signal-boundary worker exited before checkpoint: {stdout[-1000:]} {stderr[-1000:]}")
                finally:
                    if process.poll() is None:
                        process.kill();process.wait(timeout=5)
            signal_source=root/"signal-source";signal_source.mkdir()
            task=DEVELOPMENT_TASKS[0]
            source_command=("import sys; from raid_experiments.fewshot_train import train_source; "
                "train_source(sys.argv[1],sys.argv[2],sys.argv[3],seed=42,updates=2000,"
                "batch_size=8,microbatch_size=4,device='cpu',run_mode='synthetic')")
            signal_at_checkpoint(source_command,[protocol_path,features,signal_source],
                signal_source/"source_seed42.last.pt",signal.SIGINT)
            self.assertEqual(json.loads((signal_source/"source_seed42.json").read_text())["status"],"interrupted")
            signal_adapt=root/"signal-adaptation";signal_adapt.mkdir()
            adapt_command=("import sys; from raid_experiments.fewshot_train import adapt_target; "
                "adapt_target(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],task=sys.argv[5],"
                "method='head_ft',budget=2,seed=42,updates=500,batch_size=8,microbatch_size=4,"
                "device='cpu',run_mode='synthetic')")
            adapt_last=signal_adapt/task/"2shot"/"seed42"/"head_ft.last.pt"
            process=subprocess.Popen([sys.executable,"-c",adapt_command,str(protocol_path),str(features),
                str(source_ckpt),str(signal_adapt),task],cwd=repo,
                env={**os.environ,"PYTHONPATH":str(repo)},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            deadline=time.time()+90
            try:
                while time.time()<deadline and process.poll() is None:
                    if adapt_last.is_file():
                        payload=torch.load(adapt_last,map_location="cpu",weights_only=False)
                        if int(payload.get("step",0))>=100:
                            os.kill(process.pid,signal.SIGTERM)
                            stdout,stderr=process.communicate(timeout=30)
                            self.assertEqual(process.returncode,0,stderr[-2000:])
                            break
                    time.sleep(.02)
                else:
                    stdout,stderr=process.communicate(timeout=5)
                    self.fail(f"adaptation signal worker exited before checkpoint: {stdout[-1000:]} {stderr[-1000:]}")
            finally:
                if process.poll() is None:
                    process.kill();process.wait(timeout=5)
            self.assertEqual(json.loads((signal_adapt/task/"2shot"/"seed42"/"head_ft.json").read_text())["status"],"interrupted")
            source_state = torch.load(source_ckpt, map_location="cpu", weights_only=False)["state_dict"]
            self.assertEqual((source_dir / "action_normalizer.pt").is_file(), True)
            interrupted = root / "interrupted"
            train_source(protocol_path, features, interrupted, seed=42, updates=2, stop_after_updates=1,
                         batch_size=8, microbatch_size=4, device="cpu", run_mode="synthetic")
            train_source(protocol_path, features, interrupted, seed=42, updates=2,
                         batch_size=8, microbatch_size=4, device="cpu",
                         run_mode="synthetic",
                         resume_checkpoint=interrupted / "source_seed42.last.pt")
            full_state = torch.load(source_ckpt, map_location="cpu", weights_only=False)["state_dict"]
            resumed_state = torch.load(interrupted / "source_seed42.pt", map_location="cpu", weights_only=False)["state_dict"]
            for key in full_state:
                torch.testing.assert_close(full_state[key], resumed_state[key], rtol=0, atol=0)
            fresh_resume = root / "fresh-resume"
            train_source(protocol_path, features, fresh_resume, seed=42, updates=2,
                         batch_size=8, microbatch_size=4, device="cpu",
                         run_mode="synthetic",
                         resume_checkpoint=interrupted / "source_seed42.last.pt")
            self.assertTrue((fresh_resume / "source_seed42.pt").is_file())
            task = DEVELOPMENT_TASKS[0]
            norm = torch.load(source_dir / "action_normalizer.pt", map_location="cpu", weights_only=False)
            from unittest.mock import patch
            source_payload_for_gate=torch.load(source_ckpt,map_location="cpu",weights_only=False)
            with patch("raid_experiments.fewshot_train._load_source",return_value=source_payload_for_gate), \
                 patch("raid_experiments.fewshot_train._load_cache_map",return_value=({},"g"*64)), \
                 patch("raid_experiments.fewshot_train._query_rows_from_hdf5",
                       side_effect=AssertionError("query labels were opened before condition validation")):
                with self.assertRaisesRegex(FileNotFoundError,"adaptation checkpoint"):
                    evaluate_task(protocol_path,features,source_ckpt,root/"invalid-condition",
                        task=task,budget=2,seed=42,methods=("head_ft",),
                        selected_updates={"head_ft":200},device="cpu",run_mode="scientific")
            target_support = protocol["tasks"][task]["support_budgets_by_seed"]["42"]["2"]
            raw_support = RawSelectedRows(protocol["tasks"][task]["path"], task, target_support,
                                          norm["mean"], norm["std"])
            expected_support_transitions = sum(int(protocol["tasks"][task]["transitions"][str(demo)])
                                               for demo in target_support)
            self.assertEqual(len(raw_support), expected_support_transitions)
            self.assertEqual({row["transition_id"] for row in raw_support},
                             {f"{task}:{demo}:{step}" for demo in target_support
                              for step in range(int(protocol["tasks"][task]["transitions"][str(demo)]))})
            normalizer_path = source_dir / "action_normalizer.pt"
            normalizer_copy = normalizer_path.read_bytes()
            normalizer_payload = torch.load(normalizer_path, map_location="cpu", weights_only=False)
            normalizer_payload["mean"] = normalizer_payload["mean"] + 1
            torch.save(normalizer_payload, normalizer_path)
            with self.assertRaisesRegex(ValueError, "normalizer sidecar"):
                adapt_target(protocol_path, features, source_ckpt, root / "tamper", task=task,
                             method="head_ft", budget=2, seed=42, updates=2, device="cpu",
                             batch_size=8, microbatch_size=4, run_mode="synthetic")
            normalizer_path.write_bytes(normalizer_copy)
            head_timing={"warmup_updates":1,"measured_update_seconds":[]}
            adapted = adapt_target(protocol_path, features, source_ckpt, root / "runs",
                                   task=task, method="head_ft", budget=2, seed=42,
                                   updates=2, device="cpu", batch_size=8, microbatch_size=4,
                                   run_mode="synthetic",timing_profile=head_timing)
            self.assertEqual(adapted["selected_step"], 2)
            self.assertEqual(len(head_timing["measured_update_seconds"]),1)
            resumed_root = root / "resumed-runs"
            adapt_target(protocol_path, features, source_ckpt, resumed_root, task=task,
                         method="head_ft", budget=2, seed=42, updates=2, stop_after_updates=1,
                         device="cpu", batch_size=8, microbatch_size=4, run_mode="synthetic")
            adapt_target(protocol_path, features, source_ckpt, resumed_root, task=task,
                         method="head_ft", budget=2, seed=42, updates=2,
                         device="cpu", batch_size=8, microbatch_size=4,
                         run_mode="synthetic",
                         resume_checkpoint=resumed_root / task / "2shot" / "seed42" / "head_ft.last.pt")
            full_adapt = torch.load(root / "runs" / task / "2shot" / "seed42" / "head_ft.selected.pt", map_location="cpu", weights_only=False)
            resumed_adapt = torch.load(resumed_root / task / "2shot" / "seed42" / "head_ft.selected.pt", map_location="cpu", weights_only=False)
            for key in full_adapt["state_dict"]:
                torch.testing.assert_close(full_adapt["state_dict"][key], resumed_adapt["state_dict"][key], rtol=0, atol=0)
            from raid_experiments.fewshot_train import _same_checkpoint_value
            full_last=torch.load(root/"runs"/task/"2shot"/"seed42"/"head_ft.last.pt",map_location="cpu",weights_only=False)
            resumed_last=torch.load(resumed_root/task/"2shot"/"seed42"/"head_ft.last.pt",map_location="cpu",weights_only=False)
            for key in ("optimizer","order","cursor","order_rng_state","torch_rng_state",
                        "numpy_rng_state","python_rng_state","cuda_rng_state"):
                self.assertTrue(_same_checkpoint_value(full_last[key],resumed_last[key]),key)
            full_snapshot=root/"runs"/task/"2shot"/"seed42"/"head_ft.step2.pt"
            resumed_snapshot=resumed_root/task/"2shot"/"seed42"/"head_ft.step2.pt"
            self.assertEqual(full_snapshot.read_bytes(),resumed_snapshot.read_bytes())
            fresh_adapt = root / "fresh-adaptation-resume"
            adapt_target(protocol_path, features, source_ckpt, fresh_adapt, task=task,
                         method="head_ft", budget=2, seed=42, updates=2, device="cpu",
                         batch_size=8, microbatch_size=4,
                         run_mode="synthetic",
                         resume_checkpoint=root / "runs" / task / "2shot" / "seed42" / "head_ft.last.pt")
            self.assertTrue((fresh_adapt / task / "2shot" / "seed42" / "head_ft.selected.pt").is_file())
            off_schedule = root / "off-schedule"
            adapt_target(protocol_path, features, source_ckpt, off_schedule, task=task,
                         method="head_ft", budget=2, seed=42, updates=500, stop_after_updates=300,
                         device="cpu", batch_size=8, microbatch_size=4, run_mode="synthetic")
            off_checkpoint = off_schedule / task / "2shot" / "seed42" / "head_ft.last.pt"
            off_payload = torch.load(off_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(off_payload["step"], 300)
            self.assertIn("200", off_payload["snapshot_archive"])
            recovered = root / "off-schedule-recovered"
            adapt_target(protocol_path, features, source_ckpt, recovered, task=task,
                         method="head_ft", budget=2, seed=42, updates=500, stop_after_updates=300,
                         device="cpu", batch_size=8, microbatch_size=4, run_mode="synthetic",
                         resume_checkpoint=off_checkpoint)
            self.assertTrue((recovered / task / "2shot" / "seed42" / "head_ft.step200.pt").is_file())
            result = evaluate_task(protocol_path, features, source_ckpt, root / "runs",
                                   task=task, budget=2, seed=42, device="cpu",
                                   methods=("frozen_source", "target_knn", "head_ft", "support_mean",
                                            "transition_1nn", "current_weighted_knn", "source_support_knn"),
                                   selected_updates={"head_ft": 2}, run_mode="synthetic")
            self.assertEqual(set(result["methods"]), {"frozen_source", "target_knn", "head_ft", "support_mean",
                                                         "transition_1nn", "current_weighted_knn", "source_support_knn"})
            self.assertTrue(result["methods"]["head_ft"]["source_retention"]["target_adapter_active"])
            self.assertTrue((root / "runs" / task / "2shot" / "seed42" / "head_ft_predictions_head2_lorana.pt").is_file())
            source43 = root / "source43"
            train_source(protocol_path, features, source43, seed=43, updates=2,
                         batch_size=8, microbatch_size=4, device="cpu", run_mode="synthetic")
            paired = evaluate_task(protocol_path, features, source43 / "source_seed43.pt", root / "seed43-runs",
                task=task, budget=2, seed=43, device="cpu", methods=("frozen_source", "target_knn"),
                run_mode="synthetic")
            expected_support43 = list(map(int, protocol["tasks"][task]["support_budgets_by_seed"]["43"]["2"]))
            saved_prediction = torch.load(root / "seed43-runs" / task / "2shot" / "seed43" /
                "target_knn_predictions.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved_prediction["support_ids"], expected_support43)
            self.assertEqual(paired["seed"], 43)
            from raid_experiments.fewshot_report import _freeze_digest
            final_task=FINAL_TEST_TASKS[0]; source_hash=hashlib.sha256(source_ckpt.read_bytes()).hexdigest()
            lock={"confirmation_format":4,"protocol_digest":protocol["digest"],"source_checkpoints":{"42":source_hash},
                  "study_root":str(source_dir.resolve()),
                  "selected_updates":{"head_ft":200,"lora_head_ft":1000},
                  "confirmation_scope":{"support_budgets":[5]},"confirmation_registered":True}
            lock["confirmation_digest"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            confirmation_path=root/"confirmation.json";confirmation_path.write_text(json.dumps(lock))
            final_adapt=adapt_target(protocol_path,features,source_ckpt,root/"final-runs",task=final_task,
                method="head_ft",budget=5,seed=42,updates=1000,device="cpu",batch_size=8,
                microbatch_size=4,freeze_file=confirmation_path,run_mode="synthetic")
            self.assertEqual(final_adapt["updates"],200)
            self.assertEqual(final_adapt["selected_step"],200)
            final_eval = evaluate_task(protocol_path, features, source_ckpt, root / "final-runs",
                task=final_task,budget=5,seed=42,device="cpu",methods=("frozen_source","head_ft"),
                run_mode="synthetic",confirmation_file=confirmation_path)
            self.assertEqual(final_eval["selected_updates"],lock["selected_updates"])
            access=json.loads((source_dir/"final_query_access.json").read_text())
            self.assertEqual(access["conditions"],[{"task":final_task,"budget":5,"seed":42,
                "source_checkpoint_sha256":source_hash}])


if __name__ == "__main__":
    unittest.main()
