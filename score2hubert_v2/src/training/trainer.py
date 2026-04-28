"""Score2HuBERT v2 trainer.

Single-GPU training loop with FP16, cosine LR schedule, per-language val metrics.
"""

import os
import sys
import time
import math
import yaml
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

# cudnn.benchmark off — our inputs have variable seq lengths per batch, benchmark mode
# would re-probe for every new shape and slow things down.

from src.model.score2hubert import Score2HuBERT
from src.data.dataset import Score2HuBERTDataset
from src.data.collate import collate_fn

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config_path: str):
        with open(config_path, encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_logging()
        self._build_model()
        self._build_data()
        self._build_optimizer()

        self.global_step = 0
        self.best_val_loss = float('inf')

        # Loss weights
        lw = self.cfg['loss']
        self.w_hub_l1 = lw['hubert_l1']
        self.w_hub_cos = lw['hubert_cos']
        self.w_f0 = lw['f0_l1']
        self.w_voicing = lw['voicing_bce']
        self.w_hub_consist = lw.get('hubert_consist', 0.0)  # Run 8: anti-"长尾口胡"; Run 9: 0.005 + d<N exempt
        self.w_hub_consist_min_dur = lw.get('hubert_consist_min_dur', 0)  # Run 9: phonemes <N frames exempt from consist smoothing
        self.w_pos_aux = lw.get('pos_aux', 0.0)              # Run 8: force pos load-bear

        self.scaler = GradScaler(enabled=self.cfg['training']['fp16'])

        ckpt_dir = Path(self.cfg['checkpoint']['dir'])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = ckpt_dir

    def _setup_logging(self):
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'train.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout),
            ]
        )

    def _build_model(self):
        mc = self.cfg['model']
        self.model = Score2HuBERT(
            n_phones=mc['n_phones'],
            hidden_dim=mc['hidden_dim'],
            hubert_dim=mc['hubert_dim'],
            n_enc_layers=mc['n_enc_layers'],
            n_heads=mc['n_heads'],
            enc_kernel=mc['enc_kernel'],
            n_det_blocks=mc['n_det_blocks'],
            det_kernel=mc['det_kernel'],
            dropout=mc['dropout'],
        ).to(self.device)

        params = self.model.count_params()
        logger.info(f"Model params: {params['total']:,} ({params['total']/1e6:.1f}M)")
        for k, v in params.items():
            if k != 'total':
                logger.info(f"  {k}: {v:,}")

    @staticmethod
    def _make_entry_filter(cfg_data):
        """Build entry filter from config. Always rejects gtsinger entries (dropped 2026-04-13)."""
        filters = cfg_data.get('entry_filters', {})
        singing_only = filters.get('singing_only', False)

        def _filter(entry):
            # Hard reject: GTSinger is permanently dropped (see project_score2hubert_gtsinger_dropped.md)
            if entry.get('dataset', '') == 'gtsinger':
                return False
            if singing_only:
                if entry.get('singing_ratio', 1.0) <= 0:
                    return False
            return True

        return _filter

    def _build_data(self):
        dc = self.cfg['data']
        tc = self.cfg['training']

        entry_filter = self._make_entry_filter(dc)

        self.train_ds = Score2HuBERTDataset(
            data_dirs=dc['train_dirs'],
            split='train',
            max_frames=dc['max_frames'],
            entry_filter=entry_filter,
        )
        self.val_ds = Score2HuBERTDataset(
            data_dirs=dc['val_dirs'],
            split='test',
            max_frames=dc['max_frames'],
            entry_filter=entry_filter,
        )

        logger.info(f"Train: {len(self.train_ds)} samples")
        logger.info(f"Val: {len(self.val_ds)} samples")

        # Per-language weighted sampling (optional, configured via data.lang_weights)
        lang_weights = dc.get('lang_weights', None)
        sampler = None
        if lang_weights:
            from torch.utils.data import WeightedRandomSampler
            from collections import Counter
            weights = []
            for entry in self.train_ds.entries:
                lang = entry.get('language', 'zh')
                weights.append(float(lang_weights.get(lang, 1.0)))
            # Log effective per-language share
            total_w = sum(weights)
            lang_w_sum = Counter()
            for w, e in zip(weights, self.train_ds.entries):
                lang_w_sum[e.get('language', 'zh')] += w
            logger.info(f"Weighted sampling — effective batch composition:")
            for lang, w in sorted(lang_w_sum.items()):
                logger.info(f"  {lang}: {w/total_w*100:.1f}%")
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        n_workers = dc['num_workers']
        self.train_loader = DataLoader(
            self.train_ds, batch_size=tc['batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=n_workers, collate_fn=collate_fn,
            persistent_workers=dc.get('persistent_workers', True) and n_workers > 0,
            pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=tc['batch_size'], shuffle=False,
            num_workers=min(4, dc['num_workers']), collate_fn=collate_fn,
            persistent_workers=False, pin_memory=True,
        )

    def _build_optimizer(self):
        tc = self.cfg['training']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc['lr'],
            weight_decay=tc['weight_decay'],
            betas=(0.9, 0.98),
        )

        # Cosine schedule with warmup
        warmup = tc['warmup_steps']
        max_steps = tc['max_steps']

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # SP token id in unified IPA vocab (data/processed/<ds>/vocab.yaml). Stable.
    SP_ID = 3

    def _build_frame_phoneme(self, batch, T_frame):
        """Expand per-phoneme tokens to per-frame tokens via duration. [B, T_frame] long.

        Pad frames (beyond expanded length) get -1.
        """
        phonemes = batch['phonemes'].to(self.device)       # [B, T_phone]
        durations = batch['duration'].to(self.device)      # [B, T_phone]
        phone_mask = batch['phone_mask'].to(self.device)   # [B, T_phone] True=pad
        B = phonemes.shape[0]
        frame_ph = torch.full((B, T_frame), -1, dtype=torch.long, device=self.device)
        for b in range(B):
            valid_p = ~phone_mask[b]
            ph_b = phonemes[b][valid_p]
            d_b = durations[b][valid_p]
            if d_b.sum() == 0:
                continue
            expanded = torch.repeat_interleave(ph_b, d_b)
            L = min(T_frame, int(expanded.shape[0]))
            frame_ph[b, :L] = expanded[:L]
        return frame_ph

    def _build_frame_pos_dur(self, batch, T_frame):
        """Returns (dur_per_frame, pos_in_phone, pos_in_note) all [B, T_frame] float.

        dur_per_frame: duration (in frames) of the phoneme this frame belongs to.
        pos_in_phone:  0-1 position within the phoneme.
        pos_in_note:   0-1 position within the musical note (consecutive same-MIDI
                       phonemes = same note).
        """
        phonemes = batch['phonemes'].to(self.device)
        durations = batch['duration'].to(self.device)
        pitch = batch['pitch'].to(self.device)
        phone_mask = batch['phone_mask'].to(self.device)
        B = phonemes.shape[0]
        dur_pf = torch.ones(B, T_frame, dtype=torch.float, device=self.device)
        pip = torch.zeros(B, T_frame, dtype=torch.float, device=self.device)
        pin = torch.zeros(B, T_frame, dtype=torch.float, device=self.device)

        for b in range(B):
            valid_p = ~phone_mask[b]
            ph_b = phonemes[b][valid_p]
            d_b = durations[b][valid_p].clamp(min=0).long()
            pt_b = pitch[b][valid_p]
            if d_b.sum() == 0:
                continue
            T_phone_v = d_b.shape[0]
            total_d = int(d_b.sum().item())
            L = min(T_frame, total_d)

            frame_to_phone = torch.repeat_interleave(
                torch.arange(T_phone_v, device=self.device), d_b
            )[:L]
            dpf = d_b[frame_to_phone].float().clamp(min=1.0)
            dur_pf[b, :L] = dpf

            cum_durs = torch.cumsum(d_b, dim=0)
            phone_start = cum_durs - d_b
            global_idx = torch.arange(L, device=self.device)
            local_offset = (global_idx - phone_start[frame_to_phone]).float()
            pip[b, :L] = local_offset / dpf

            if T_phone_v > 1:
                change = (pt_b[1:] != pt_b[:-1]).long()
                note_idx = torch.cat([
                    torch.zeros(1, device=self.device, dtype=torch.long),
                    change.cumsum(0),
                ])
            else:
                note_idx = torch.zeros(1, device=self.device, dtype=torch.long)
            num_notes = int(note_idx.max().item()) + 1
            note_durs = torch.zeros(num_notes, device=self.device, dtype=torch.long)
            note_durs.scatter_add_(0, note_idx, d_b)
            note_cumsum = torch.cumsum(note_durs, 0)
            note_starts = note_cumsum - note_durs
            frame_note_idx = note_idx[frame_to_phone]
            note_start_f = note_starts[frame_note_idx].float()
            note_len_f = note_durs[frame_note_idx].float().clamp(min=1.0)
            pin[b, :L] = (global_idx.float() - note_start_f) / note_len_f

        return dur_pf, pip, pin

    def compute_loss(self, batch, output):
        """Compute all loss components. Returns (total_loss, loss_dict)."""
        frame_mask = batch['frame_mask'].to(self.device)  # [B, T] True=pad
        valid_frames = ~frame_mask  # True=valid

        # Frame-level phoneme expansion (used for SP=0 + frame consistency)
        T_frame = batch['hubert'].shape[1]
        frame_ph = self._build_frame_phoneme(batch, T_frame)
        sp_frame_mask = (frame_ph == self.SP_ID)

        # Frame-level dur + pos targets (for sustained weight + pos_aux)
        dur_pf, pos_phone_gt, pos_note_gt = self._build_frame_pos_dur(batch, T_frame)

        # Run 9: sustained 1.5x weight removed — was over-biasing long notes vs short consonants.
        # All valid frames equal weight.

        # HuBERT L1 — include SP frames; target=0 drives pred → silence
        hubert_target = batch['hubert'].to(self.device).clone()
        hubert_target[sp_frame_mask] = 0.0
        hubert_pred = output['hubert_pred']
        l1_pf = (hubert_pred - hubert_target).abs().mean(dim=-1)  # [B, T]
        w_l1 = valid_frames.float()
        hub_l1 = (l1_pf * w_l1).sum() / w_l1.sum().clamp(min=1)

        # HuBERT cos — exclude SP
        non_sp_valid = valid_frames & ~sp_frame_mask
        cos_sim = F.cosine_similarity(hubert_pred, hubert_target, dim=-1)  # [B, T]
        w_cos = non_sp_valid.float()
        hub_cos = ((1.0 - cos_sim) * w_cos).sum() / w_cos.sum().clamp(min=1)

        # Frame-consistency (anti-"长尾口胡"): L1 on adjacent pred diff, same phoneme only
        ph_t = frame_ph[:, :-1]
        ph_t1 = frame_ph[:, 1:]
        same_ph = (ph_t == ph_t1) & (ph_t >= 0) & (ph_t != self.SP_ID)
        same_ph = same_ph & valid_frames[:, :-1] & valid_frames[:, 1:]
        if self.w_hub_consist_min_dur > 0:
            # Run 9: phonemes with duration < min_dur frames exempt from consist — protects short consonants
            same_ph = same_ph & (dur_pf[:, :-1] >= self.w_hub_consist_min_dur)
        if same_ph.any():
            diff = (hubert_pred[:, 1:] - hubert_pred[:, :-1]).abs().mean(dim=-1)
            hub_consist = diff[same_ph].mean()
        else:
            hub_consist = hub_l1.new_zeros(())

        # Aux pos prediction — force encoder/LR to keep pos signal decodable
        pos_pred = output['pos_pred']  # [B, T, 2]
        pos_tgt = torch.stack([pos_phone_gt, pos_note_gt], dim=-1)  # [B, T, 2]
        pos_pf = (pos_pred - pos_tgt).abs().mean(dim=-1)  # [B, T]
        w_pos = non_sp_valid.float()
        pos_aux = (pos_pf * w_pos).sum() / w_pos.sum().clamp(min=1)

        # F0 L1 (log-scale, voiced + non-rest frames, with outlier-frame downweighting)
        f0_target = batch['f0'].to(self.device)
        f0_pred = output['f0_pred']
        note_pitch_frame = output['note_pitch_frame']  # [B, T_frame] long
        # Non-rest + voiced
        voiced_mask = (f0_target > 30.0) & valid_frames & (note_pitch_frame > 0)
        if voiced_mask.any():
            # Deviation in cents vs MIDI note
            # 0 MIDI note maps to 0 Hz — clamp for numerical safety (masked out anyway)
            midi_hz = 440.0 * (2.0 ** ((note_pitch_frame.float() - 69.0) / 12.0))
            midi_hz = midi_hz * (note_pitch_frame > 0).float()
            f0_safe = f0_target.clamp(min=30.0)
            midi_safe = midi_hz.clamp(min=30.0)
            dev_cents = 1200.0 * torch.log2(f0_safe / midi_safe)  # [B, T]

            # Outlier weight: down-weight frames where |dev| > 200c (portamento / label errors)
            outlier_weight = torch.where(
                dev_cents.abs() > 200.0,
                torch.full_like(dev_cents, 0.1),
                torch.ones_like(dev_cents),
            )
            weight = outlier_weight * voiced_mask.float()  # [B, T]
            log_f0_pred = torch.log1p(f0_pred)
            log_f0_target = torch.log1p(f0_target)
            f0_l1 = ((log_f0_pred - log_f0_target).abs() * weight).sum() / (weight.sum() + 1e-8)
        else:
            f0_l1 = torch.tensor(0.0, device=self.device)

        # Voicing BCE (with logits for autocast safety)
        voicing_target = (f0_target > 30.0).float()
        voicing_logit = output['voicing_logit']
        voicing_bce = F.binary_cross_entropy_with_logits(
            voicing_logit[valid_frames], voicing_target[valid_frames]
        )

        total = (self.w_hub_l1 * hub_l1 + self.w_hub_cos * hub_cos +
                 self.w_f0 * f0_l1 + self.w_voicing * voicing_bce +
                 self.w_hub_consist * hub_consist +
                 self.w_pos_aux * pos_aux)

        loss_dict = {
            'total': total.item(),
            'hub_l1': hub_l1.item(),
            'hub_cos': hub_cos.item(),
            'hub_consist': hub_consist.item(),
            'pos_aux': pos_aux.item(),
            'f0_l1': f0_l1.item(),
            'voicing': voicing_bce.item(),
        }
        return total, loss_dict

    @torch.no_grad()
    def _per_sample_metrics(self, batch, output):
        """Per-sample metrics for language breakdown. Returns dict of [B] tensors (CPU floats).

        SP-frame handling matches compute_loss (Run 8): GT HuBERT[sp_frame_mask]=0 for L1
        (so reported hub_l1 includes the cost of "be silent at rest"), and SP frames are
        EXCLUDED from cos (cos vs zero is undefined).

        F0 metrics filter frames to (f0_target > 30 & note_pitch > 0 & valid) — matching
        compute_loss's voiced_mask. Without note_pitch>0 check, rest-note frames with
        voicing carryover contaminate the metric (model predicts 0 at rest by design,
        but target has non-zero f0, yielding huge cent error that is not a model failure).
        """
        frame_mask = batch['frame_mask'].to(self.device)
        valid = (~frame_mask).float()  # [B, T]
        sample_valid = valid.sum(dim=-1).clamp(min=1)  # [B]

        # SP=0 force (matches compute_loss)
        T_frame = batch['hubert'].shape[1]
        frame_ph = self._build_frame_phoneme(batch, T_frame)
        sp_frame_mask = (frame_ph == self.SP_ID)
        non_sp_valid = ((~frame_mask) & ~sp_frame_mask).float()
        sample_non_sp_valid = non_sp_valid.sum(dim=-1).clamp(min=1)

        # HuBERT L1 per sample (SP target=0)
        hubert_target = batch['hubert'].to(self.device).clone()
        hubert_target[sp_frame_mask] = 0.0
        hubert_pred = output['hubert_pred']
        l1_per_frame = (hubert_pred - hubert_target).abs().mean(dim=-1)  # [B, T]
        hub_l1 = (l1_per_frame * valid).sum(dim=-1) / sample_valid

        # HuBERT cos per sample — non-SP frames only
        cos_sim = F.cosine_similarity(hubert_pred, hubert_target, dim=-1)  # [B, T]
        hub_cos = ((1.0 - cos_sim) * non_sp_valid).sum(dim=-1) / sample_non_sp_valid

        # F0 metrics: voiced + non-rest mask (matches compute_loss's filter)
        f0_target = batch['f0'].to(self.device)
        f0_pred = output['f0_pred']
        note_pitch_frame = output['note_pitch_frame']
        voiced = ((f0_target > 30.0) & (note_pitch_frame > 0) & valid.bool()).float()
        sample_voiced = voiced.sum(dim=-1).clamp(min=1)

        # F0 MAE in cents per sample
        cents_abs = (1200.0 * torch.log2(
            f0_pred.clamp(min=1e-3) / f0_target.clamp(min=1e-3)
        )).abs()  # [B, T]
        f0_cents = (cents_abs * voiced).sum(dim=-1) / sample_voiced

        # F0 L1 on log-space (unfiltered-weight, consistent with training)
        log_f0_diff = (torch.log1p(f0_pred) - torch.log1p(f0_target)).abs()
        f0_l1 = (log_f0_diff * voiced).sum(dim=-1) / sample_voiced

        # Voicing BCE per sample
        voicing_target = (f0_target > 30.0).float()
        voicing_logit = output['voicing_logit']
        voicing_per_frame = F.binary_cross_entropy_with_logits(
            voicing_logit, voicing_target, reduction='none'
        )
        voicing_bce = (voicing_per_frame * valid).sum(dim=-1) / sample_valid

        return {
            'hub_l1': hub_l1.detach().cpu(),
            'hub_cos': hub_cos.detach().cpu(),
            'f0_l1': f0_l1.detach().cpu(),
            'f0_cents': f0_cents.detach().cpu(),
            'voicing': voicing_bce.detach().cpu(),
        }

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        with autocast(enabled=self.cfg['training']['fp16']):
            output = self.model(
                phonemes=batch['phonemes'].to(self.device),
                pitch=batch['pitch'].to(self.device),
                durations=batch['duration'].to(self.device),
                phone_mask=batch['phone_mask'].to(self.device),
                f0_target=batch['f0'].to(self.device),
            )
            total_loss, loss_dict = self.compute_loss(batch, output)

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['training']['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss_dict

    @torch.no_grad()
    def _train_sample_per_lang_metrics(self, n_batches: int = 8):
        """Quick pass over `n_batches` train batches to compute per-lang metrics.

        Used for train-vs-val gap monitoring (overfit signal). Returns dict
        {lang: {metric: mean, n_samples: int}}.
        """
        self.model.eval()
        lang_sums: dict = {}
        lang_counts: dict = {}
        iter_ = iter(self.train_loader)
        for _ in range(n_batches):
            try:
                batch = next(iter_)
            except StopIteration:
                break
            with autocast(enabled=self.cfg['training']['fp16']):
                output = self.model(
                    phonemes=batch['phonemes'].to(self.device),
                    pitch=batch['pitch'].to(self.device),
                    durations=batch['duration'].to(self.device),
                    phone_mask=batch['phone_mask'].to(self.device),
                    f0_target=batch['f0'].to(self.device),
                )
                per_sample = self._per_sample_metrics(batch, output)
            for i, lang in enumerate(batch['languages']):
                if lang not in lang_sums:
                    lang_sums[lang] = {}
                    lang_counts[lang] = 0
                for mn, mt in per_sample.items():
                    lang_sums[lang][mn] = lang_sums[lang].get(mn, 0.0) + float(mt[i])
                lang_counts[lang] += 1
        out = {}
        for lang, s in lang_sums.items():
            n = max(lang_counts[lang], 1)
            out[lang] = {k: v / n for k, v in s.items()}
            out[lang]['n_samples'] = lang_counts[lang]
        return out

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_losses = {}
        n_batches = 0

        # Per-language accumulators: lang -> metric_name -> running sum
        lang_sums: dict = {}
        lang_counts: dict = {}

        for batch in self.val_loader:
            with autocast(enabled=self.cfg['training']['fp16']):
                output = self.model(
                    phonemes=batch['phonemes'].to(self.device),
                    pitch=batch['pitch'].to(self.device),
                    durations=batch['duration'].to(self.device),
                    phone_mask=batch['phone_mask'].to(self.device),
                    f0_target=batch['f0'].to(self.device),
                )
                _, loss_dict = self.compute_loss(batch, output)
                per_sample = self._per_sample_metrics(batch, output)

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v
            n_batches += 1

            for i, lang in enumerate(batch['languages']):
                if lang not in lang_sums:
                    lang_sums[lang] = {}
                    lang_counts[lang] = 0
                for metric_name, metric_tensor in per_sample.items():
                    lang_sums[lang][metric_name] = (
                        lang_sums[lang].get(metric_name, 0.0) + float(metric_tensor[i])
                    )
                lang_counts[lang] += 1

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        lang_avg = {}
        for lang, sums in lang_sums.items():
            n = max(lang_counts[lang], 1)
            lang_avg[lang] = {k: v / n for k, v in sums.items()}
            lang_avg[lang]['n_samples'] = lang_counts[lang]
        return avg_losses, lang_avg

    def save_checkpoint(self, tag='latest'):
        path = self.ckpt_dir / f'{tag}.pt'
        torch.save({
            'step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.cfg,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.global_step = ckpt['step']
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.info(f"Loaded checkpoint from step {self.global_step}")

    def train(self):
        tc = self.cfg['training']
        max_steps = tc['max_steps']

        logger.info(f"Starting training: {max_steps} steps, batch {tc['batch_size']}, "
                     f"lr {tc['lr']}, fp16={tc['fp16']}")

        # NOTE: no torch.manual_seed() here — setting a CUDA seed interacts with
        # cuDNN RNN autograd to trigger a slow deterministic path (~18x slower LSTM
        # backward) when gradient inputs are non-trivial (e.g. Kaiming-init LR features
        # feeding F0Predictor). Training is non-reproducible regardless (fp16 reductions
        # + cuDNN default + multi-worker DataLoader), so seeding gave no real guarantee.

        train_iter = iter(self.train_loader)
        t0 = time.time()
        running_loss = {}
        log_count = 0

        while self.global_step < max_steps:
            # Get next batch (re-create iterator on epoch boundary)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            try:
                loss_dict = self.train_step(batch)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.warning(f"OOM at step {self.global_step+1}, skipping batch")
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                raise
            self.global_step += 1

            # Accumulate for logging
            for k, v in loss_dict.items():
                running_loss[k] = running_loss.get(k, 0) + v
            log_count += 1

            # Log
            if self.global_step % tc['log_interval'] == 0:
                elapsed = time.time() - t0
                avg = {k: v / log_count for k, v in running_loss.items()}
                steps_per_sec = tc['log_interval'] / elapsed
                lr = self.scheduler.get_last_lr()[0]

                logger.info(
                    f"step {self.global_step:>6d} | "
                    f"loss {avg['total']:.4f} | "
                    f"hub_l1 {avg['hub_l1']:.4f} | "
                    f"hub_cos {avg['hub_cos']:.4f} | "
                    f"consist {avg.get('hub_consist', 0):.4f} | "
                    f"pos_aux {avg.get('pos_aux', 0):.4f} | "
                    f"f0 {avg['f0_l1']:.4f} | "
                    f"voicing {avg['voicing']:.4f} | "
                    f"lr {lr:.2e} | "
                    f"{steps_per_sec:.1f} step/s"
                )

                running_loss = {}
                log_count = 0
                t0 = time.time()

            # Validate
            if self.global_step % tc['val_interval'] == 0:
                # VRAM snapshot before validate (captures training peak)
                if torch.cuda.is_available():
                    train_peak = torch.cuda.max_memory_allocated() / (1024**3)
                    train_reserved = torch.cuda.memory_reserved() / (1024**3)
                    torch.cuda.reset_peak_memory_stats()
                else:
                    train_peak = train_reserved = 0.0

                val_losses, lang_avg = self.validate()
                # Train-sample per-lang (for overfit detection; gap > 0.10 = warn)
                train_avg = self._train_sample_per_lang_metrics(n_batches=8)
                logger.info(
                    f"  VAL step {self.global_step} | "
                    f"loss {val_losses['total']:.4f} | "
                    f"hub_l1 {val_losses['hub_l1']:.4f} | "
                    f"hub_cos {val_losses['hub_cos']:.4f} | "
                    f"f0 {val_losses['f0_l1']:.4f} | "
                    f"vram peak={train_peak:.2f}G res={train_reserved:.2f}G"
                )
                for lang in sorted(lang_avg):
                    v = lang_avg[lang]
                    t = train_avg.get(lang, {})
                    # val hub_cos is (1 - cos), so smaller = better. "cos_sim" = 1 - hub_cos.
                    val_cos = 1.0 - v['hub_cos']
                    train_cos = (1.0 - t['hub_cos']) if 'hub_cos' in t else float('nan')
                    gap = train_cos - val_cos if 'hub_cos' in t else float('nan')
                    gap_warn = '⚠️' if (not (gap != gap)) and gap > 0.10 else ''
                    logger.info(
                        f"    [{lang} n={v['n_samples']}] "
                        f"val_cos_sim {val_cos:.4f} | train_cos_sim {train_cos:.4f} | "
                        f"gap {gap:+.4f}{gap_warn} | "
                        f"hub_l1 {v['hub_l1']:.4f} | "
                        f"f0_mae {v['f0_cents']:.1f}c | "
                        f"voicing {v['voicing']:.4f}"
                    )

                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best')
                    logger.info(f"  New best val loss: {self.best_val_loss:.4f}")

            # Save checkpoint
            if self.global_step % tc['save_interval'] == 0:
                self.save_checkpoint(f'step_{self.global_step}')
                self.save_checkpoint('latest')

                # Clean old checkpoints
                keep = self.cfg['checkpoint']['keep_last']
                step_ckpts = sorted(self.ckpt_dir.glob('step_*.pt'),
                                    key=lambda p: int(p.stem.split('_')[1]))
                for old in step_ckpts[:-keep]:
                    old.unlink()

        # Final save
        self.save_checkpoint('final')
        self.save_checkpoint('latest')
        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
