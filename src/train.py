"""src/train.py
Single-run training executor with Hydra, Optuna and exhaustive WandB logging.
This script is intended to be launched by ``src.main`` but can also be invoked
stand-alone exactly as described in the README / work-flow documentation.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import optuna
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from optuna.trial import Trial
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.model import GLoRAWrapper, build_model, extract_lora_ranks
from src.preprocess import GLUEDataModule

CACHE_DIR = ".cache/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Utility helpers -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Simple top-1 accuracy."""
    return (preds == labels).float().mean().item()


def _apply_optuna_params(cfg: DictConfig, params: Dict[str, Any]) -> DictConfig:
    """Inject Optuna-sampled *params* (dot keys) into a **copy** of *cfg*."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    new_cfg = OmegaConf.create(cfg_dict)
    for dotted_key, value in params.items():
        node = new_cfg
        *parents, leaf = dotted_key.split(".")
        for p in parents:
            if p not in node or node[p] is None:
                node[p] = {}
            node = node[p]
        node[leaf] = value
    return new_cfg


def _save_cfg(cfg: DictConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)


# -----------------------------------------------------------------------------
# Single training routine -----------------------------------------------------
# -----------------------------------------------------------------------------

def _train_one_run(cfg: DictConfig, trial: Optional[Trial] = None) -> float:
    """Train **one** run.  If *trial* is supplied we assume Optuna hyper-parameter
    optimisation and therefore **disable** WandB logging for that run to avoid
    cluttering the dashboard with intermediate trials.

    Returns the best validation accuracy reached by the model.
    """

    # ------------------------------------------------------------------
    # Optuna parameter sampling (if active) -----------------------------
    # ------------------------------------------------------------------
    if trial is not None and cfg.get("optuna", None):
        sampled: Dict[str, Any] = {}
        for key, space in cfg.optuna.get("search_space", {}).items():
            stype = space.get("type")
            if stype == "loguniform":
                sampled[key] = trial.suggest_float(key, space["low"], space["high"], log=True)
            elif stype == "uniform":
                sampled[key] = trial.suggest_float(key, space["low"], space["high"], log=False)
            elif stype == "int":
                sampled[key] = trial.suggest_int(key, space["low"], space["high"], log=False)
            elif stype == "categorical":
                sampled[key] = trial.suggest_categorical(key, space["choices"])
            else:
                raise ValueError(f"Unsupported search-space type '{stype}' for key '{key}'")
        cfg = _apply_optuna_params(cfg, sampled)

    # ------------------------------------------------------------------
    # WandB initialisation ---------------------------------------------
    # ------------------------------------------------------------------
    wandb_run: Optional[wandb.wandb_run.Run] = None
    wandb_enabled = trial is None and cfg.wandb.mode != "disabled"
    if wandb_enabled:
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print(f"[WandB] URL -> {wandb_run.url}")

    # ------------------------------------------------------------------
    # Reproducibility ---------------------------------------------------
    # ------------------------------------------------------------------
    if cfg.get("seed", None) is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    # ------------------------------------------------------------------
    # Directories -------------------------------------------------------
    # ------------------------------------------------------------------
    run_dir = Path(cfg.results_dir) / cfg.run.run_id
    _save_cfg(cfg, run_dir)  # save *immediately* for debugging

    # ------------------------------------------------------------------
    # Data --------------------------------------------------------------
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone, cache_dir=CACHE_DIR)
    dm = GLUEDataModule(cfg, tokenizer)
    train_loader, val_loader = dm.train_loader, dm.val_loader
    num_labels = dm.num_labels

    # ------------------------------------------------------------------
    # Model -------------------------------------------------------------
    # ------------------------------------------------------------------
    model_or_wrapper = build_model(cfg, num_labels)
    if isinstance(model_or_wrapper, GLoRAWrapper):
        model = model_or_wrapper.model
        glora = model_or_wrapper
    else:
        model = model_or_wrapper
        glora = None
    model.to(DEVICE)

    # ------------------------------------------------------------------
    # Optimiser & LR-schedule ------------------------------------------
    # ------------------------------------------------------------------
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = AdamW(param_groups, lr=cfg.training.learning_rate)

    updates_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
    total_updates = updates_per_epoch * cfg.training.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_updates,
    )

    # ------------------------------------------------------------------
    # Training loop -----------------------------------------------------
    # ------------------------------------------------------------------
    best_val_acc = -1.0
    global_step = 0
    start_time = time.time()

    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0

        for step, batch in enumerate(train_loader):
            if cfg.training.max_train_batches is not None and step >= cfg.training.max_train_batches:
                break

            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / cfg.training.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * cfg.training.gradient_accumulation_steps

            preds = torch.argmax(outputs.logits.detach(), dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            seen += preds.size(0)

            # gradient accumulation ‑> step
            if (step + 1) % cfg.training.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optim.step()
                scheduler.step()
                optim.zero_grad()
                global_step += 1

                if glora is not None:
                    glora.step_callback()

                if wandb_run is not None:
                    wandb.log(
                        {
                            "train_loss": running_loss / (seen / preds.size(0)),
                            "train_acc": correct / seen if seen else 0.0,
                            "train_lr": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

        # ---------------- Validation ----------------
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for vstep, batch in enumerate(val_loader):
                if cfg.training.max_val_batches is not None and vstep >= cfg.training.max_val_batches:
                    break
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                val_correct += (preds == batch["labels"]).sum().item()
                val_total += preds.size(0)
        val_acc = val_correct / val_total if val_total else 0.0
        val_loss /= max(1, vstep + 1)

        if wandb_run is not None:
            wandb.log({"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch}, step=global_step)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # lightweight ckpt (state_dict)
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            # full fp32 model (useful for inference scripts)
            torch.save(model.to("cpu"), run_dir / "best_model_full.pt")
            model.to(DEVICE)

    # ------------------------------------------------------------------
    # Post-training bookkeeping ----------------------------------------
    # ------------------------------------------------------------------
    wall_clock = time.time() - start_time
    final_ranks = extract_lora_ranks(model)
    with open(run_dir / "final_ranks.json", "w") as f:
        json.dump(final_ranks, f, indent=2)

    if wandb_run is not None:
        wandb_run.summary["best_val_acc"] = best_val_acc
        wandb_run.summary["wall_clock_train_seconds"] = wall_clock
        wandb_run.summary["final_ranks"] = final_ranks
        wandb_run.finish()

    _save_cfg(cfg, run_dir)  # save final cfg (may include tuned params)
    return best_val_acc


# -----------------------------------------------------------------------------
# Hydra entry-point -----------------------------------------------------------
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:  # pragma: no-cover
    """Top-level Hydra entry.  Handles Optuna search followed by final training
    with the best set of hyper-parameters (if optimisation is requested).
    """

    # ------------------------------------------------------------------
    # Merge *run*-specific YAML into cfg -------------------------------
    # ------------------------------------------------------------------
    if cfg.get("run", None) is None:
        raise ValueError("Argument run=<run_id> is required.")
    run_yaml = Path(get_original_cwd()) / "config" / "run" / f"{cfg.run}.yaml"
    if not run_yaml.exists():
        raise FileNotFoundError(f"Run config file not found: {run_yaml}")
    run_cfg = OmegaConf.load(run_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"run": run_cfg}))  # nest under cfg.run
    # flatten some convenience aliases
    cfg.run.run_id = run_cfg["run_id"]

    # ------------------------------------------------------------------
    # Trial / full mode handling ---------------------------------------
    # ------------------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.run.training.epochs = 1  # type: ignore[attr-defined]
        cfg.run.training.max_train_batches = 2  # type: ignore[attr-defined]
        cfg.run.training.max_val_batches = 2  # type: ignore[attr-defined]
        cfg.run.optuna.n_trials = 0  # type: ignore[attr-defined]
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode={cfg.mode}. Use 'trial' or 'full'.")

    # propagate some high-level fields so that downstream code does not need to
    # dive into cfg.run.* areas
    merged_cfg = OmegaConf.merge(cfg, cfg.run)

    # ------------------------------------------------------------------
    # Optuna? -----------------------------------------------------------
    # ------------------------------------------------------------------
    if merged_cfg.optuna.n_trials and merged_cfg.optuna.n_trials > 0:
        study = optuna.create_study(direction=merged_cfg.optuna.direction)

        def _objective(trial: Trial):
            cfg_copy = OmegaConf.create(OmegaConf.to_container(merged_cfg, resolve=True))
            return _train_one_run(cfg_copy, trial)

        study.optimize(_objective, n_trials=merged_cfg.optuna.n_trials)
        print("[Optuna] Best parameters:", study.best_trial.params)
        tuned_cfg = _apply_optuna_params(merged_cfg, study.best_trial.params)
        best_val = _train_one_run(tuned_cfg)
    else:
        best_val = _train_one_run(merged_cfg)

    print(f"[Train] Run {merged_cfg.run.run_id} finished – best val acc = {best_val:.4f}")


if __name__ == "__main__":  # pragma: no-cover
    _hydra_main()