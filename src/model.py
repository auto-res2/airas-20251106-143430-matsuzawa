"""src/model.py
Model construction utilities – handles plain backbone, LoRA, AdaLoRA and the
proposed GLoRA wrappers.
"""
from __future__ import annotations

import warnings
from typing import Any

import torch
from peft import AdaLoraConfig, LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

CACHE_DIR = ".cache/"

# -----------------------------------------------------------------------------
# Proposed GLoRA wrapper -------------------------------------------------------
# -----------------------------------------------------------------------------

class GLoRAWrapper:
    """Wrap a HF model with gradient-norm aware rank allocation for LoRA."""

    def __init__(self, hf_model: torch.nn.Module, r_init: int, total_rank: int, beta: float, warmup_steps: int):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:  # pragma: no-cover
            raise RuntimeError("peft>=0.6 required for GLoRA") from e

        self.model = get_peft_model(hf_model, LoraConfig(r=r_init, bias="none"))
        self.total_rank = total_rank
        self.beta = beta
        self.warmup_steps = warmup_steps
        self._step = 0

        # gather LoRA modules that possess lora_A / lora_B parameters
        self.lora_modules = [m for m in self.model.modules() if hasattr(m, "lora_A") and hasattr(m, "lora_B")]
        self.g_ema = torch.zeros(len(self.lora_modules))

        # register hooks for gradient norms
        for idx, m in enumerate(self.lora_modules):
            # lora_B is a dict in newer PEFT versions
            if hasattr(m.lora_B, "default"):
                m.lora_B.default.weight.register_hook(self._make_hook(idx))
            else:
                m.lora_B.weight.register_hook(self._make_hook(idx))

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    # ------------------------------------------------------------------
    def _make_hook(self, idx: int):
        def _hook(grad):
            gn = grad.norm(p="fro")
            self.g_ema[idx] = self.beta * self.g_ema[idx] + (1 - self.beta) * gn.detach()
        return _hook

    # ------------------------------------------------------------------
    # External API ------------------------------------------------------
    # ------------------------------------------------------------------
    def step_callback(self):
        """To be called **once** per optimisation step **after** gradient update.
        Performs rank re-allocation exactly at *warmup_steps*.
        """
        self._step += 1
        if self._step != self.warmup_steps:
            return

        scores = self.g_ema / self.g_ema.sum()
        new_ranks = torch.round(scores * self.total_rank).int().tolist()
        # budget correction (due to rounding)
        diff = self.total_rank - sum(new_ranks)
        for i in range(abs(diff)):
            new_ranks[i % len(new_ranks)] += 1 if diff > 0 else -1

        # rebuild layers with new rank allocations
        for new_r, mod in zip(new_ranks, self.lora_modules):
            # Handle different PEFT versions
            lora_A_weight = mod.lora_A.default.weight if hasattr(mod.lora_A, "default") else mod.lora_A.weight
            lora_B_weight = mod.lora_B.default.weight if hasattr(mod.lora_B, "default") else mod.lora_B.weight

            old_r = lora_A_weight.size(0)
            if new_r == old_r:
                continue
            keep = min(old_r, new_r)
            with torch.no_grad():
                A_old = lora_A_weight.data[:keep].clone()
                B_old = lora_B_weight.data[:keep].clone()
                mod.update_layer(r=new_r)  # provided by peft.LoraLayer
                # Update references after update_layer
                lora_A_weight = mod.lora_A.default.weight if hasattr(mod.lora_A, "default") else mod.lora_A.weight
                lora_B_weight = mod.lora_B.default.weight if hasattr(mod.lora_B, "default") else mod.lora_B.weight
                lora_A_weight.data[:keep] = A_old
                lora_B_weight.data[:keep] = B_old
        print(f"[GLoRA] New per-layer ranks: {new_ranks}")


# -----------------------------------------------------------------------------
# Build helpers ----------------------------------------------------------------
# -----------------------------------------------------------------------------


def build_model(cfg, num_labels: int):
    """Return *model* or wrapper according to *cfg.model.peft.type*.
    The caller is responsible for sending the returned object / underlying model
    to the right device.
    """
    backbone_name = cfg.model.backbone
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        backbone_name, num_labels=num_labels, cache_dir=CACHE_DIR
    )

    peft_cfg = cfg.model.get("peft", {})
    peft_type = peft_cfg.get("type", "none")

    if peft_type == "none":
        return hf_model
    elif peft_type == "lora":
        lora_cfg = LoraConfig(r=peft_cfg.get("init_rank", 8), bias="none")
        return get_peft_model(hf_model, lora_cfg)
    elif peft_type == "adalora":
        ada_cfg = AdaLoraConfig(
            init_r=peft_cfg.get("init_rank", 8),
            target_r=peft_cfg.get("total_rank_budget", 128),
            beta1=cfg.adalora.beta1,
            beta2=cfg.adalora.beta2,
            orth_reg_weight=cfg.adalora.orth_reg_weight,
            tinit=cfg.adalora.adaptation_start_step,
        )
        return get_peft_model(hf_model, ada_cfg)
    elif peft_type == "glora":  # proposed method
        return GLoRAWrapper(
            hf_model,
            r_init=peft_cfg.get("init_rank", 8),
            total_rank=cfg.glo_ra.total_rank_budget,
            beta=cfg.glo_ra.beta_ema,
            warmup_steps=cfg.glo_ra.warmup_steps,
        )
    else:
        raise ValueError(f"Unsupported peft type: {peft_type}")


# -----------------------------------------------------------------------------
# Miscellaneous utilities ------------------------------------------------------
# -----------------------------------------------------------------------------

def extract_lora_ranks(model: torch.nn.Module):
    """Return a ``dict[layer_name -> rank]`` for every LoRA layer in *model*."""
    ranks = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            # Handle different PEFT versions
            lora_A_weight = module.lora_A.default.weight if hasattr(module.lora_A, "default") else module.lora_A.weight
            ranks[name] = lora_A_weight.size(0)
    return ranks