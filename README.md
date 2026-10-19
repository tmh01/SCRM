# SCRM

Key implementation of the SCRM Self-Anchor framework for few-shot cross-log remaining-time prediction.

```text
src/scrm/data/          Event logs, case splits and prefix construction
src/scrm/selection/     Six-view signatures and Learned-LOTO source selection
src/scrm/training/      Transformer pretraining and target adaptation
src/scrm/models/        Support diagnostics, risk calibration and final prediction
src/scrm/experiments/   Main-framework experiment runner
configs/               Dataset paths and experiment settings
```

Install `requirements.txt`, then open `run.py` in PyCharm and click Run. Its default configuration uses synthetic logs for a short pipeline check, not paper results. For real logs, change `CONFIG_PATH` to a dataset configuration and set `QUICK_CHECK = False`.

Only framework code is included. Baselines, raw data, checkpoints and experiment outputs are not distributed. Dataset paths and optional column mappings are configured in YAML.

The original experiment's LOTO convention is preserved: the evaluation log is excluded as a meta-target, but can remain a candidate source for other auxiliary meta-targets. No subsequent selector optimization is included.
