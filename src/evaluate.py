"""src/evaluate.py
Independent evaluation / analysis script.
Fetches run information from WandB, saves raw metrics and generates figures /
aggregated comparison visualisations.

Usage:
    uv run python -m src.evaluate results_dir=path run_ids='["run-1", "run-2"]'
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from omegaconf import OmegaConf

# -----------------------------------------------------------------------------
# Helper functions ------------------------------------------------------------
# -----------------------------------------------------------------------------

def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _learning_curve_figure(df, run_id: str, out_dir: Path):
    """Generate train / val accuracy learning curve."""
    plt.figure(figsize=(6, 4))
    if "train_acc" in df:
        sns.lineplot(x=df.index, y=df["train_acc"], label="train_acc")
    if "val_acc" in df:
        sns.lineplot(x=df.index, y=df["val_acc"], label="val_acc")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title(f"Learning curve – {run_id}")
    plt.legend()
    plt.tight_layout()
    fname = f"{run_id}_learning_curve.pdf"
    path = out_dir / fname
    plt.savefig(path)
    plt.close()
    return path


def _bar_chart(aggregated: List[dict], metric: str, out_dir: Path):
    plt.figure(figsize=(6, 4))
    run_ids = [d["run_id"] for d in aggregated]
    values = [d[metric] for d in aggregated]
    sns.barplot(x=run_ids, y=values)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.ylabel(metric)
    plt.title(f"Comparison – {metric}")
    plt.tight_layout()
    fname = f"comparison_{metric}_bar_chart.pdf"
    path = out_dir / fname
    plt.savefig(path)
    plt.close()
    return path


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():  # pragma: no-cover
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help="JSON string list of run IDs")
    args = parser.parse_args()

    run_ids: List[str] = json.loads(args.run_ids)
    results_root = Path(args.results_dir)

    # load wandb credentials from repo config (entity / project)
    repo_cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    repo_cfg = OmegaConf.load(repo_cfg_path)
    entity, project = repo_cfg.wandb.entity, repo_cfg.wandb.project

    api = wandb.Api()
    per_run_agg: List[dict] = []
    generated_files: List[Path] = []

    for run_id in run_ids:
        print(f"[Eval] Processing {run_id} …")
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history(keys=["train_acc", "val_acc", "train_loss", "val_loss"], pandas=True)
        summary = run.summary._json_dict
        config = dict(run.config)

        # export metrics JSON
        run_dir = results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_json_path = run_dir / "metrics.json"
        _write_json({"summary": summary, "config": config, "history": history.to_dict(orient="list")}, metrics_json_path)
        generated_files.append(metrics_json_path)

        # figures -------------------------------------------------------
        fig_path = _learning_curve_figure(history, run_id, run_dir)
        generated_files.append(fig_path)

        # aggregated dict for later comparison -------------------------
        per_run_agg.append({
            "run_id": run_id,
            "best_val_acc": summary.get("best_val_acc", float("nan")),
            "wall_clock_train_seconds": summary.get("wall_clock_train_seconds", float("nan")),
        })

    # ------------------------------------------------------------------
    # Aggregated cross-run analysis ------------------------------------
    # ------------------------------------------------------------------
    comp_dir = results_root / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    aggregated_path = comp_dir / "aggregated_metrics.json"
    _write_json(per_run_agg, aggregated_path)
    generated_files.append(aggregated_path)

    # improvement rate vs first run
    if len(per_run_agg) >= 2:
        baseline = per_run_agg[0]["best_val_acc"]
        for d in per_run_agg[1:]:
            d["improvement_vs_baseline"] = (d["best_val_acc"] - baseline) / baseline if baseline else float("nan")
        _write_json(per_run_agg, aggregated_path)  # overwrite with new fields

    # bar chart for best_val_acc --------------------------------------
    bar_path = _bar_chart(per_run_agg, "best_val_acc", comp_dir)
    generated_files.append(bar_path)

    # print generated file paths --------------------------------------
    print("[Eval] Generated files:")
    for p in generated_files:
        print(p.resolve())


if __name__ == "__main__":
    main()