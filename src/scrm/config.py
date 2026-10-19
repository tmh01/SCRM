from __future__ import annotations

"""模型与实验配置；路径、数据集名称不进入模型定义。"""

from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    embedding_dim: int = 32
    hidden_dim: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.10
    batch_size: int = 64
    max_epochs: int = 120
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    pretrain_max_epochs: int = 120
    pretrain_patience: int = 15
    pretrain_learning_rate: float = 1e-3
    finetune_max_epochs: int = 120
    finetune_patience: int = 15
    finetune_learning_rate: float = 3e-4
    min_prefix_len: int = 1
    inner_val_ratio: float = 0.20
    memory_batch_size: int = 512
    memory_max_cases: int = 20
    gate_risk_floor: float = 0.05
    gate_power: float = 2.0
    gate_folds: int = 3
    gate_grid_step: float = 0.05
    gate_regularization: float = 0.01
    gate_correction_ratio_cap: float = 1.5
    min_activity_prior_cases: int = 2
    source_weight_grid_step: float = 0.1
    bootstrap_reps: int = 20_000
    permutation_reps: int = 50_000
    alpha: float = 0.05

    def training_config(self, quick: bool = False):
        if not quick:
            return self
        return replace(self, max_epochs=1, patience=1, pretrain_max_epochs=1,
                       pretrain_patience=1, finetune_max_epochs=1, finetune_patience=1, batch_size=32)


def load_settings(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("配置文件必须是 YAML 对象")
    settings.setdefault("training", {})
    settings.setdefault("K", [5, 10, 20, 40])
    settings.setdefault("seeds", [42, 2024, 3407])
    if not settings.get("groups"):
        raise ValueError("配置缺少 groups 日志分组")
    ExperimentConfig(**settings["training"])
    return settings
