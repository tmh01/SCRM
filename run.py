from __future__ import annotations

"""PyCharm 入口：修改配置路径后点击运行，无需填写命令行参数。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from scrm.experiments.runner import run_experiment

# 正式实验改为实际数据配置，并关闭快速检查。
CONFIG_PATH = ROOT / "configs" / "demo.yaml"
QUICK_CHECK = True
DEVICE = "auto"
RESUME = True


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    run_experiment(CONFIG_PATH, root=ROOT, quick=QUICK_CHECK, device=DEVICE, resume=RESUME)
