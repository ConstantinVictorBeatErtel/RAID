"""Configuration loading and validation for unattended RAID runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _reject_unknown(raw: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} configuration fields: {', '.join(unknown)}")


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in {".json", ".jsonl"}:
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised in cloud setup
        raise RuntimeError("PyYAML is required to read YAML experiment configs") from exc
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


@dataclass(frozen=True)
class DataConfig:
    dataset_dir: str = "data/libero_spatial/libero_spatial"
    feature_dir: str = "data/libero_spatial/features"
    split_path: str = "artifacts/splits/libero_spatial_seed20260920.json"
    tasks: int = 10
    demonstrations_per_task: int = 50
    validation_per_task: int = 5
    test_per_task: int = 5
    train_per_task: int = 10
    split_seed: int = 20260920
    train_seed: int = 42
    image_size: int = 128


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int = 384
    action_dim: int = 7
    hidden_dim: int = 512
    bottleneck_dim: int = 256
    decoder_dropout: float = 0.1
    retrieval_k: int = 3
    retrieval_temperature: float = 0.1
    prior_dropout: float = 0.4
    prior_noise_std: float = 0.05
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_blocks: tuple[int, int] = (10, 11)
    lora_targets: tuple[str, str] = ("qkv", "proj")


@dataclass(frozen=True)
class TrainingConfig:
    updates: int = 2000
    batch_size: int = 64
    microbatch_size: int = 8
    decoder_lr: float = 3e-4
    lora_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    validate_every: int = 100
    num_workers: int = 4
    device: str = "cuda"


@dataclass(frozen=True)
class RolloutConfig:
    smoke_episodes_per_task: int = 1
    episodes_per_task: int = 10
    settle_steps: int = 5
    max_steps: int = 600
    deterministic: bool = True
    use_predicted_future: bool = True


@dataclass(frozen=True)
class CloudConfig:
    provider: str = "runpod"
    budget_usd: float = 100.0
    max_gpu_hourly_usd: float = 0.80
    preferred_gpu: str = "A40"
    fallback_gpu: str = "RTX 4090"
    container_disk_gb: int = 20
    volume_gb: int = 100
    min_vcpus: int = 8
    min_host_ram_gb: int = 32
    terminate_after_minutes: int = 180
    max_retries: int = 1
    launch_enabled: bool = False
    reserve_usd: float = 15.0


@dataclass(frozen=True)
class PredictorConfig:
    profile: str = "recorded_rgb"
    source: str = "recorded_frames"
    checkpoint_revision: str = "none"
    generation_fps: int = 20
    generation_frames: int = 3
    allow_action_conditioned_future: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str = "libero_spatial"
    libero_plus_subset: str = "disabled"
    robocasa_task_group: str = "atomic_seen"
    control_initial_states: str = "official_paired"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "raid_pilot_100"
    protocol: str = "observed_and_predicted"
    horizon: int = 3
    libero_commit: str = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    output_dir: str = "artifacts/raid_pilot"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def validate(self) -> None:
        if self.cloud.budget_usd <= 0:
            raise ValueError("cloud budget must be positive")
        if self.cloud.max_gpu_hourly_usd <= 0:
            raise ValueError("max_gpu_hourly_usd must be positive")
        if self.data.validation_per_task + self.data.test_per_task >= self.data.demonstrations_per_task:
            raise ValueError("validation and test holdouts leave no training pool")
        if self.data.train_per_task > self.data.demonstrations_per_task - self.data.validation_per_task - self.data.test_per_task:
            raise ValueError("train_per_task exceeds the available training pool")
        if self.training.batch_size % self.training.microbatch_size != 0:
            raise ValueError("batch_size must be divisible by microbatch_size")
        if self.model.retrieval_k < 1:
            raise ValueError("retrieval_k must be at least one")
        if self.horizon not in (1, 3):
            raise ValueError("supported horizons are 1 and 3")
        if self.cloud.provider != "runpod":
            raise ValueError("the unattended pilot currently supports only Runpod")
        if self.predictor.profile != "recorded_rgb":
            raise ValueError("legacy future-prediction profiles are disabled; use recorded RGB frames")
        if self.benchmark.name not in {"libero_spatial", "libero_plus", "robocasa365"}:
            raise ValueError("unsupported benchmark")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _tuple(value: Any, default: tuple) -> tuple:
    if value is None:
        return default
    return tuple(value)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = _read_mapping(Path(path))
    _reject_unknown(raw, {"name", "protocol", "horizon", "libero_commit", "output_dir", "data", "model", "training", "rollout", "cloud", "predictor", "benchmark"}, "top-level")
    data_raw = raw.get("data", {}); _reject_unknown(data_raw, set(DataConfig.__dataclass_fields__), "data")
    data = DataConfig(**data_raw)
    model_raw = dict(raw.get("model", {}))
    _reject_unknown(model_raw, set(ModelConfig.__dataclass_fields__), "model")
    model_raw["lora_target_blocks"] = _tuple(model_raw.get("lora_target_blocks"), (10, 11))
    model_raw["lora_targets"] = _tuple(model_raw.get("lora_targets"), ("qkv", "proj"))
    model = ModelConfig(**model_raw)
    for section, cls in (("training", TrainingConfig), ("rollout", RolloutConfig), ("cloud", CloudConfig), ("predictor", PredictorConfig), ("benchmark", BenchmarkConfig)):
        _reject_unknown(raw.get(section, {}), set(cls.__dataclass_fields__), section)
    training = TrainingConfig(**raw.get("training", {}))
    rollout = RolloutConfig(**raw.get("rollout", {}))
    cloud = CloudConfig(**raw.get("cloud", {}))
    predictor = PredictorConfig(**raw.get("predictor", {}))
    benchmark = BenchmarkConfig(**raw.get("benchmark", {}))
    config = ExperimentConfig(
        name=raw.get("name", "raid_pilot_100"),
        protocol=raw.get("protocol", "observed_and_predicted"),
        horizon=int(raw.get("horizon", 3)),
        libero_commit=raw.get("libero_commit", ExperimentConfig.libero_commit),
        output_dir=raw.get("output_dir", "artifacts/raid_pilot"),
        data=data,
        model=model,
        training=training,
        rollout=rollout,
        cloud=cloud,
        predictor=predictor,
        benchmark=benchmark,
    )
    config.validate()
    return config


@dataclass(frozen=True)
class ExpansionStudy:
    name: str
    budgets_per_task: tuple[int, ...]
    seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    dataset: str
    predictors: tuple[str, ...]
    lookup_controls: tuple[str, ...]


def load_expansion_study(path: str | Path) -> ExpansionStudy:
    raw = _read_mapping(Path(path))
    _reject_unknown(raw, {"name", "budgets_per_task", "seeds", "conditions", "lookup_controls", "dataset", "additional_benchmarks", "predictors", "native_policy_reference", "split_seed", "validation_per_task", "test_per_task", "notes"}, "expansion study")
    return ExpansionStudy(name=str(raw["name"]), budgets_per_task=tuple(map(int, raw["budgets_per_task"])), seeds=tuple(map(int, raw["seeds"])), conditions=tuple(raw["conditions"]), dataset=str(raw["dataset"]), predictors=tuple(raw.get("predictors", ("recorded_rgb",))), lookup_controls=tuple(raw.get("lookup_controls", ())))


def enumerate_expansion(path: str | Path) -> list[dict[str, Any]]:
    study = load_expansion_study(path)
    return [{"budget": budget, "seed": seed, "condition": condition, "predictor": predictor, "dataset": study.dataset}
            for predictor in study.predictors for seed in study.seeds for budget in study.budgets_per_task for condition in study.conditions]
