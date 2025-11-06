"""src/main.py
High-level orchestrator.  Reads a *run* id from the CLI, applies mode-specific
modifications (trial / full) and launches *src.train* as a subprocess with the
appropriate Hydra overrides.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

# -----------------------------------------------------------------------------
# Hydra entry-point -----------------------------------------------------------
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):  # pragma: no-cover
    if cfg.get("run", None) is None:
        raise ValueError("CLI override run=<run_id> is required.")
    if cfg.get("mode", None) not in {"trial", "full"}:
        raise ValueError("mode=trial | mode=full is required.")

    overrides: List[str] = [
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]

    if cfg.mode == "trial":  # lightweight execution
        overrides.extend([
            "wandb.mode=disabled",
            "optuna.n_trials=0",
            "training.epochs=1",
            "training.max_train_batches=2",
            "training.max_val_batches=2",
        ])
    elif cfg.mode == "full":
        overrides.append("wandb.mode=online")

    # convert overrides to "key=value" CLI args accepted by Hydra
    cmd = [sys.executable, "-u", "-m", "src.train", *overrides]
    print("[Main] Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=get_original_cwd())


if __name__ == "__main__":  # pragma: no-cover
    main()