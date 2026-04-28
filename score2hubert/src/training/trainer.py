"""
Training loop for Score2HuBERT.

Features:
- Mixed precision (FP16) training with GradScaler
- Gradient clipping
- Exponential Moving Average (EMA) of model weights
- Warmup + cosine annealing LR schedule
- TensorBoard logging
- Checkpoint save/resume
- Validation with sample generation
- Multi-phase training support (pretrain → finetune)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import time
import json
import copy
from pathlib import Path
from typing import Optional, Dict
from collections import defaultdict
import logging

from ..model.score2hubert import Score2HuBERT
from ..data.dataset import Score2HuBERTDataset
from ..data.collate import collate_fn
from .losses import Score2HuBERTLoss

logger = logging.getLogger(__name__)


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model: nn.Module):
        """Apply EMA weights (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original weights (after evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def get_scheduler(optimizer, config: dict):
    """Create learning rate scheduler based on config."""
    sched_cfg = config["training"]["scheduler"]
    sched_type = sched_cfg["type"]
    warmup_steps = sched_cfg["warmup_steps"]
    total_steps = sched_cfg.get("total_steps", config["training"]["max_steps"])
    min_lr = sched_cfg.get("min_lr", 1e-6)
    base_lr = config["training"]["optimizer"]["lr"]

    if sched_type == "warmup_cosine":
        import math

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cosine)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched_type == "warmup_linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_lr / base_lr, 1 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")


class Trainer:
    """
    Score2HuBERT training manager.

    Handles the complete training lifecycle:
    - Dataset loading and batching
    - Training loop with mixed precision
    - Validation and metric logging
    - Checkpointing and EMA
    - Resume from checkpoint
    """

    def __init__(
        self,
        config: dict,
        resume_from: Optional[str] = None,
        init_from: Optional[str] = None,
    ):
        """
        resume_from: full restart of an existing training run (loads model +
                     optimizer + scheduler + step + EMA). Same architecture.
        init_from:   warm-start from a different model — loads model weights and
                     EMA shadow only (with strict=False so new modules can be added),
                     keeps fresh optimizer / scheduler / step. Used when switching
                     training mode (e.g. flow → deterministic).
        """
        if resume_from and init_from:
            raise ValueError("Pass either --resume or --init-from, not both")

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_step = 0
        # Track best by recon_l1 (deterministic content metric), NOT total val loss.
        # See project_score2hubert_decisions: cos sim was 0.76 in Plan A while audio
        # had no lyrics — total loss dominated by flow MSE has the same blind spot.
        self.best_val_l1 = float("inf")

        self.mode = config["model"].get("mode", "flow")

        # Setup directories
        self.ckpt_dir = Path(config["paths"]["checkpoint_dir"])
        self.log_dir = Path(config["paths"]["log_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Build model
        self.model = Score2HuBERT(config).to(self.device)
        param_counts = self.model.count_parameters()
        logger.info(f"Model parameters: {param_counts}")
        logger.info(f"Training mode: {self.mode}")

        # Freeze flow_decoder in deterministic mode — it's dead weight there,
        # kept in the state dict only for future hybrid training.
        if self.mode == "deterministic":
            for p in self.model.flow_decoder.parameters():
                p.requires_grad = False
            n_frozen = sum(p.numel() for p in self.model.flow_decoder.parameters())
            logger.info(f"Froze flow_decoder ({n_frozen/1e6:.1f}M params) in det mode")

        # Config-driven selective unfreezing: freeze everything, then unfreeze
        # only the specified modules. Used for targeted fine-tuning (e.g., f0 only).
        unfreeze_only = self.config["training"].get("unfreeze_only")
        if unfreeze_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for name, p in self.model.named_parameters():
                if any(key in name for key in unfreeze_only):
                    p.requires_grad = True
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            logger.info(f"Selective unfreeze {unfreeze_only}: "
                        f"training {n_train/1e6:.2f}M, frozen {n_frozen/1e6:.1f}M")

        # In discrete mode, freeze everything except discrete_head.
        if self.mode == "discrete":
            for name, p in self.model.named_parameters():
                if "discrete_head" not in name:
                    p.requires_grad = False
            # Freeze codebook inside discrete_head (it's K-means centroids)
            if hasattr(self.model, "discrete_head"):
                self.model.discrete_head.codebook.weight.requires_grad = False
            n_frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Discrete mode: froze {n_frozen/1e6:.1f}M params, "
                        f"training discrete_head only ({n_train/1e6:.2f}M params)")

        # In hybrid mode, freeze EVERYTHING except flow_refiner.
        # Det head's output is used as conditioning (detached in forward()),
        # so it must not receive gradients.
        if self.mode == "hybrid":
            for name, p in self.model.named_parameters():
                if "flow_refiner" not in name:
                    p.requires_grad = False
            n_frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            n_refiner = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Hybrid mode: froze {n_frozen/1e6:.1f}M params, "
                        f"training flow_refiner only ({n_refiner/1e6:.2f}M params)")

        # Loss
        self.loss_fn = Score2HuBERTLoss(config)

        # Optimizer — only trainable params (excludes frozen flow_decoder in det mode)
        opt_cfg = config["training"]["optimizer"]
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable params: {n_trainable/1e6:.2f}M")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=opt_cfg["lr"],
            betas=tuple(opt_cfg["betas"]),
            weight_decay=opt_cfg["weight_decay"],
            eps=opt_cfg["eps"],
        )

        # Scheduler
        self.scheduler = get_scheduler(self.optimizer, config)

        # Mixed precision
        self.use_fp16 = config["training"].get("fp16", True) and self.device == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_fp16)

        # EMA
        ema_cfg = config["training"].get("ema", {})
        self.use_ema = ema_cfg.get("enabled", True)
        if self.use_ema:
            self.ema = EMA(self.model, decay=ema_cfg.get("decay", 0.9999))

        # Gradient accumulation
        self.accumulate_grad = config["training"].get("accumulate_grad", 1)

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
        except ImportError:
            logger.warning("TensorBoard not available, logging to console only")
            self.writer = None

        # Resume / init
        if resume_from:
            self._load_checkpoint(resume_from)
        elif init_from:
            self._init_from_checkpoint(init_from)

    def _build_dataloaders(self) -> tuple:
        """Create training and validation dataloaders."""
        data_cfg = self.config["data"]
        train_cfg = self.config["training"]

        train_dataset = Score2HuBERTDataset(
            processed_dir=data_cfg["processed_dir"],
            split="train",
            max_frames=data_cfg["max_frames"],
            min_frames=data_cfg["min_frames"],
            augmentation=data_cfg.get("augmentation"),
        )

        val_dataset = Score2HuBERTDataset(
            processed_dir=data_cfg["processed_dir"],
            split="val",
            max_frames=data_cfg["max_frames"],
            min_frames=data_cfg["min_frames"],
        )

        num_workers = train_cfg["num_workers"]
        loader_kwargs = dict(
            batch_size=train_cfg["batch_size"],
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        # Worker-only options (DataLoader rejects these when num_workers=0)
        if num_workers > 0:
            if train_cfg.get("persistent_workers", False):
                loader_kwargs["persistent_workers"] = True
            if "prefetch_factor" in train_cfg:
                loader_kwargs["prefetch_factor"] = train_cfg["prefetch_factor"]

        # Language-balanced training: hard-oversample minority languages so each
        # language has roughly equal representation. Duplicates metadata entries
        # (not files) → DataLoader uses regular shuffle → fast sequential I/O.
        balance_cfg = train_cfg.get("language_balance")
        if balance_cfg and balance_cfg.get("enabled", False):
            lang_groups = {}
            for m in train_dataset.samples:
                lang = m.get("language", "zh")
                lang_groups.setdefault(lang, []).append(m)
            max_count = max(len(v) for v in lang_groups.values())
            balanced = []
            for lang, group in lang_groups.items():
                repeat = max(1, round(max_count / len(group)))
                balanced.extend(group * repeat)
            orig_len = len(train_dataset.samples)
            train_dataset.samples = balanced
            lang_summary = {lang: len(g) for lang, g in lang_groups.items()}
            logger.info(f"Language balance: {lang_summary} → "
                        f"oversampled to {len(balanced)} entries (from {orig_len})")

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )

        logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        return train_loader, val_loader

    def train(self):
        """Main training loop."""
        train_loader, val_loader = self._build_dataloaders()
        train_cfg = self.config["training"]
        max_steps = train_cfg["max_steps"]
        val_every = train_cfg["val_every"]
        save_every = train_cfg["save_every"]
        log_every = train_cfg["log_every"]
        grad_clip = train_cfg.get("grad_clip", 1.0)

        self.model.train()
        data_iter = iter(train_loader)
        running_metrics = defaultdict(float)
        log_count = 0

        logger.info(f"Starting training from step {self.global_step}/{max_steps}")

        while self.global_step < max_steps:
            # Get batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            with autocast("cuda", enabled=self.use_fp16):
                # Build variance targets dict
                var_targets = {}
                if "energy" in batch:
                    var_targets["energy"] = batch["energy"]
                if "breathiness" in batch:
                    var_targets["breathiness"] = batch["breathiness"]

                output = self.model(
                    phonemes=batch["phonemes"],
                    pitch=batch["pitch"],
                    duration=batch["duration"],
                    singing_ratio=batch["singing_ratio"],
                    hubert_target=batch["hubert"],
                    f0_target=batch["f0"],
                    phone_mask=batch["phone_mask"],
                    frame_mask=batch["frame_mask"],
                    lang_id=batch["lang_id"],
                    variance_targets=var_targets if var_targets else None,
                    tech=batch.get("tech"),
                    hubert_tokens=batch.get("hubert_tokens"),
                )

                loss = output["loss"] / self.accumulate_grad

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient step
            if (self.global_step + 1) % self.accumulate_grad == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

                if self.use_ema:
                    self.ema.update(self.model)

            self.global_step += 1

            # Accumulate metrics
            metrics = output.get("metrics", {})
            metrics["total_loss"] = output["loss"].item()
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            for k, v in metrics.items():
                running_metrics[k] += v
            log_count += 1

            # Log
            if self.global_step % log_every == 0:
                avg_metrics = {k: v / log_count for k, v in running_metrics.items()}
                self._log_metrics(avg_metrics, "train")
                log_parts = [
                    f"Step {self.global_step}/{max_steps}",
                    f"loss={avg_metrics.get('total_loss', 0):.4f}",
                    f"flow={avg_metrics.get('flow_loss', 0):.4f}",
                    f"f0={avg_metrics.get('f0_loss', 0):.4f}",
                ]
                if "recon_l1" in avg_metrics:
                    log_parts.append(
                        f"recon_l1={avg_metrics['recon_l1']:.4f}"
                    )
                    log_parts.append(
                        f"recon_cos={avg_metrics.get('recon_cos_sim', 0):.3f}"
                    )
                log_parts.append(f"lr={avg_metrics.get('lr', 0):.2e}")
                logger.info(" | ".join(log_parts))
                running_metrics.clear()
                log_count = 0

            # Validation
            if self.global_step % val_every == 0:
                val_metrics = self._validate(val_loader)
                self._log_metrics(val_metrics, "val")
                val_loss = val_metrics.get("total_loss", float("inf"))
                val_l1 = val_metrics.get("recon_l1", float("inf"))
                val_cos = val_metrics.get("recon_cos_sim", 0.0)
                logger.info(
                    f"Validation | loss={val_loss:.4f} | l1={val_l1:.4f} | "
                    f"cos={val_cos:.3f} | "
                    f"f0_mae={val_metrics.get('f0_mae_cents', 0):.1f} cents"
                )

                if val_l1 < self.best_val_l1:
                    self.best_val_l1 = val_l1
                    self._save_checkpoint("best")

                self.model.train()

            # Save
            if self.global_step % save_every == 0:
                self._save_checkpoint(f"step_{self.global_step}")

        # Final save
        self._save_checkpoint("final")
        logger.info("Training complete!")

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> dict:
        """Run validation epoch."""
        self.model.eval()

        if self.use_ema:
            self.ema.apply(self.model)

        all_metrics = defaultdict(float)
        n_batches = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            var_targets = {}
            if "energy" in batch:
                var_targets["energy"] = batch["energy"]
            if "breathiness" in batch:
                var_targets["breathiness"] = batch["breathiness"]

            with autocast("cuda", enabled=self.use_fp16):
                output = self.model(
                    phonemes=batch["phonemes"],
                    pitch=batch["pitch"],
                    duration=batch["duration"],
                    singing_ratio=batch["singing_ratio"],
                    hubert_target=batch["hubert"],
                    f0_target=batch["f0"],
                    phone_mask=batch["phone_mask"],
                    frame_mask=batch["frame_mask"],
                    lang_id=batch["lang_id"],
                    variance_targets=var_targets if var_targets else None,
                    tech=batch.get("tech"),
                    hubert_tokens=batch.get("hubert_tokens"),
                )

            metrics = output.get("metrics", {})
            metrics["total_loss"] = output["loss"].item()
            for k, v in metrics.items():
                all_metrics[k] += v
            n_batches += 1

        if self.use_ema:
            self.ema.restore(self.model)

        return {k: v / max(1, n_batches) for k, v in all_metrics.items()}

    def _log_metrics(self, metrics: dict, prefix: str):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        ckpt = {
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_val_l1": self.best_val_l1,
            "config": self.config,
        }
        if self.use_ema:
            ckpt["ema_shadow"] = self.ema.shadow

        path = self.ckpt_dir / f"{name}.pt"
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")

        # Manage checkpoint count
        if name.startswith("step_"):
            save_top_k = self.config["training"].get("save_top_k", 5)
            step_ckpts = sorted(
                self.ckpt_dir.glob("step_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            while len(step_ckpts) > save_top_k:
                oldest = step_ckpts.pop(0)
                oldest.unlink()

    def _init_from_checkpoint(self, path: str):
        """
        Warm-start: load model weights and EMA shadow only, with strict=False.
        Optimizer / scheduler / step / best metric all start fresh.

        Used when switching architectures or training modes (e.g. flow → det).
        New modules (e.g. det_head when initializing from a flow-mode checkpoint)
        keep their fresh PyTorch initialization.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        missing, unexpected = self.model.load_state_dict(
            ckpt["model_state"], strict=False
        )
        # In det mode initialized from a flow ckpt, det_head.* will be missing — expected.
        # Anything else missing is a real concern.
        expected_missing_prefixes = ["det_head.", "flow_refiner.", "discrete_head."]
        unexpected_missing = [
            k for k in missing
            if not any(k.startswith(p) for p in expected_missing_prefixes)
        ]
        if unexpected_missing:
            logger.warning(
                f"Init checkpoint missing {len(unexpected_missing)} unexpected keys, "
                f"first 5: {unexpected_missing[:5]}"
            )
        if unexpected:
            logger.warning(
                f"Init checkpoint had {len(unexpected)} unexpected keys (ignored), "
                f"first 5: {unexpected[:5]}"
            )

        n_loaded = len(ckpt["model_state"]) - len(unexpected)
        logger.info(
            f"Init from {path} (step {ckpt.get('global_step', '?')}): "
            f"loaded {n_loaded} keys, {len(missing)} fresh "
            f"({len([k for k in missing if k.startswith('det_head.')])} in det_head)"
        )

        # EMA shadow: merge instead of replace, so fresh modules (det_head) keep
        # their freshly-initialized shadow values that match the live params.
        if self.use_ema and "ema_shadow" in ckpt:
            n_merged = 0
            for name, val in ckpt["ema_shadow"].items():
                if name in self.ema.shadow:
                    self.ema.shadow[name] = val.to(self.device)
                    n_merged += 1
            logger.info(
                f"Merged EMA shadow: {n_merged}/{len(self.ema.shadow)} params "
                f"loaded from checkpoint, rest stay at fresh init"
            )

        # Fresh optimizer, scheduler, step, best — DO NOT load these.

    def _load_checkpoint(self, path: str):
        """Resume from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])

        # IMPORTANT: scheduler.load_state_dict() restores `base_lrs` from the
        # saved checkpoint, which silently OVERRIDES any lr change in the new
        # config. Re-apply the new config's base lr so that resume + new lr works.
        new_base_lr = self.config["training"]["optimizer"]["lr"]
        old_base_lrs = list(self.scheduler.base_lrs)
        if any(abs(b - new_base_lr) > 1e-12 for b in old_base_lrs):
            self.scheduler.base_lrs = [new_base_lr for _ in self.scheduler.base_lrs]
            for group in self.optimizer.param_groups:
                group["initial_lr"] = new_base_lr
            logger.info(
                f"Overriding scheduler base_lr {old_base_lrs} -> {new_base_lr} "
                f"(new config lr takes precedence on resume)"
            )

        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step = ckpt["global_step"]
        # Resume best metric — falls back to inf if loading an old checkpoint
        # that tracked total_loss (old key). First val will set the new baseline.
        self.best_val_l1 = ckpt.get("best_val_l1", float("inf"))

        if self.use_ema and "ema_shadow" in ckpt:
            self.ema.shadow = ckpt["ema_shadow"]

        logger.info(f"Resumed from {path} at step {self.global_step}")
