# 案例前缀与少样本实验单元的数据结构。
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class PrefixSample:
    case_id: int
    business_key: str
    process_id: int
    activity_names: list[str]
    time_features: np.ndarray
    prefix_len: int
    elapsed_hours: float
    delta_hours: float
    remaining_hours: float
    case_duration_hours: float
    current_time: pd.Timestamp

@dataclass
class FullTrace:
    case_id: int
    activity_names: list[str]
    timestamps: list[pd.Timestamp]

@dataclass
class ExperimentBundle:
    domain: str
    target: str
    k: int
    seed: int
    support_ids: list[int]
    inner_train_ids: list[int]
    inner_val_ids: list[int]
    test_ids: list[int]
    support_samples: list
    inner_train_samples: list
    inner_val_samples: list
    test_samples: list
    traces: dict[int, FullTrace]
    manifest_path: Path
    manifest_sha256: str
    split_path: Path
