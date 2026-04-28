"""
Score2HuBERT: Full model assembly.

Two output heads selected by `model.mode`:

  - 'deterministic' (default for new training): direct regression from
    [frame_hidden | f0_emb] -> HuBERT via DetHubertHead.
    Single forward pass, no ODE steps. Replaces flow matching after the
    Plan A finding that HuBERT space is ~80% deterministic given the score.

  - 'flow' (legacy, Plan A): OT-CFM flow matching via FlowMatchingDecoder
    with DiT blocks. Kept in code for the future hybrid (det warm-start +
    flow refiner on residual).

Pipeline:
    Score (phonemes, pitch, duration, singing_ratio)
        → ScoreEncoder (Conformer) → phone-level hidden [B, T_phone, 512]
        → LengthRegulator → frame-level hidden [B, T_frame, 512]
        → VarianceAdaptor → frame-level hidden + variance features
        → F0Predictor → F0 contour [B, T_frame]
        → DetHubertHead OR FlowMatchingDecoder → HuBERT features [B, T_frame, 768]

Training: teacher forcing with ground truth durations, F0, variances.
Inference: predicted durations (optional), predicted F0, then det/flow sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .encoder import ScoreEncoder
from .flow_matching import FlowMatchingDecoder
from .f0_predictor import F0Predictor
from .variance import VarianceAdaptor
from .modules import LengthRegulator, Conv1DStack
from .det_head import DetHubertHead


class Score2HuBERT(nn.Module):
    """
    Complete Score-to-HuBERT feature synthesis model.

    Converts musical scores (phonemes + pitch + duration + expressiveness)
    into HuBERT features + F0 that can be directly fed into RVC's decoder
    for voice conversion, bypassing the traditional audio intermediate.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        enc_cfg = config["model"]["encoder"]
        fm_cfg = config["model"]["flow_matching"]
        f0_cfg = config["model"]["f0_predictor"]
        var_cfg = config["model"]["variance"]

        d_model = enc_cfg["d_model"]
        d_hubert = fm_cfg["d_hubert"]

        # Mode: 'flow' (original Plan A) or 'deterministic' (Plan C — bypass flow,
        # direct regression). See project_score2hubert_decisions: HuBERT space
        # favors regression, flow matching's stochastic sampling fights the
        # deterministic target. Default 'flow' for backwards compatibility with
        # Plan A checkpoints; new training runs use 'deterministic'.
        self.mode = config["model"].get("mode", "flow")

        # Score encoder
        self.encoder = ScoreEncoder(
            phoneme_vocab_size=enc_cfg["phoneme_vocab_size"],
            d_model=d_model,
            n_layers=enc_cfg["n_layers"],
            n_heads=enc_cfg["n_heads"],
            d_ff=enc_cfg["d_ff"],
            conv_kernel_size=enc_cfg["conv_kernel_size"],
            dropout=enc_cfg["dropout"],
            max_pitch=enc_cfg["max_pitch"],
            pitch_embed_dim=enc_cfg.get("pitch_embed_dim", 256),
            use_relative_pos=enc_cfg.get("use_relative_pos", True),
        )

        # Length regulator
        self.length_regulator = LengthRegulator()

        # Duration predictor (optional — for when durations aren't given)
        dur_cfg = config["model"].get("duration_predictor", {})
        self.predict_duration = dur_cfg.get("enabled", False)
        if self.predict_duration:
            self.duration_predictor = Conv1DStack(
                in_dim=d_model,
                hidden_dim=dur_cfg.get("hidden_dim", 256),
                out_dim=1,
                n_layers=dur_cfg.get("n_layers", 3),
                kernel_size=dur_cfg.get("kernel_size", 3),
                dropout=dur_cfg.get("dropout", 0.1),
            )

        # Variance adaptor
        self.variance_adaptor = VarianceAdaptor(
            d_cond=d_model,
            energy_cfg=var_cfg.get("energy"),
            breathiness_cfg=var_cfg.get("breathiness"),
            voicing_cfg=var_cfg.get("voicing"),
        )

        # F0 predictor
        self.f0_predictor = F0Predictor(
            d_cond=d_model,
            hidden_dim=f0_cfg["hidden_dim"],
            conv_layers=f0_cfg["conv_layers"],
            conv_kernel=f0_cfg["conv_kernel"],
            lstm_layers=f0_cfg["lstm_layers"],
            dropout=f0_cfg["dropout"],
        )

        # Flow matching decoder (CFG drop is applied here in forward(), not inside)
        cfg_config = fm_cfg.get("cfg", {})
        self.cfg_drop_prob = cfg_config.get("drop_prob", 0.15)
        self.flow_decoder = FlowMatchingDecoder(
            d_hubert=d_hubert,
            d_cond=d_model,
            n_layers=fm_cfg["n_layers"],
            n_heads=fm_cfg["n_heads"],
            d_ff=fm_cfg["d_ff"],
            dropout=fm_cfg["dropout"],
            sigma_min=fm_cfg.get("sigma_min", 1e-4),
        )

        # F0 conditioning: inject F0 into the flow matching / det-head condition
        self.f0_cond_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Deterministic HuBERT head — replaces flow_decoder in 'deterministic' mode.
        # Always built (so checkpoints have a consistent shape across modes), but
        # only used in det mode. In flow mode it gets no gradient and can be ignored.
        det_cfg = config["model"].get("det_head", {})
        self.det_head = DetHubertHead(
            d_cond=d_model,
            d_hubert=d_hubert,
            d_hidden=det_cfg.get("d_hidden", 768),
            n_blocks=det_cfg.get("n_blocks", 6),
            kernel_size=det_cfg.get("kernel_size", 7),
            dropout=det_cfg.get("dropout", 0.1),
        )

        # Discrete HuBERT head — only used in 'discrete' mode. Replaces continuous
        # regression with K-means codebook token classification.
        if self.mode == "discrete":
            from .discrete_head import DiscreteHubertHead
            disc_cfg = config["model"].get("discrete_head", {})
            self.discrete_head = DiscreteHubertHead(
                d_cond=d_model,
                n_codes=disc_cfg.get("n_codes", 1024),
                d_hidden=disc_cfg.get("d_hidden", 512),
                n_blocks=disc_cfg.get("n_blocks", 4),
                kernel_size=disc_cfg.get("kernel_size", 7),
                dropout=disc_cfg.get("dropout", 0.1),
                codebook_path=disc_cfg.get("codebook_path", ""),
                d_hubert=d_hubert,
            )
            # Warm-start conv blocks from det_head if requested
            if disc_cfg.get("init_from_det_head", False):
                n_copied = self.discrete_head.init_from_det_head(self.det_head.state_dict())
                import logging
                logging.getLogger(__name__).info(
                    f"DiscreteHubertHead: copied {n_copied} weights from det_head")

        # Flow refiner — only used in 'hybrid' mode. Refines det_head's coarse
        # output by modeling the residual via OT-CFM.
        if self.mode == "hybrid":
            from .flow_refiner import FlowRefiner
            refiner_cfg = config["model"].get("flow_refiner", {})
            self.flow_refiner = FlowRefiner(
                d_hubert=d_hubert,
                d_cond=d_model + d_hubert + d_model,  # frame_hidden + hubert_coarse + f0_emb
                d_hidden=refiner_cfg.get("d_hidden", 384),
                n_blocks=refiner_cfg.get("n_blocks", 4),
                kernel_size=refiner_cfg.get("kernel_size", 7),
                dropout=refiner_cfg.get("dropout", 0.1),
                noise_scale=refiner_cfg.get("noise_scale", 1.0),
            )

    def forward(
        self,
        phonemes: torch.Tensor,          # [B, T_phone]
        pitch: torch.Tensor,             # [B, T_phone] MIDI
        duration: torch.Tensor,          # [B, T_phone] frames
        singing_ratio: torch.Tensor,     # [B] or [B, T_phone]
        hubert_target: torch.Tensor,     # [B, T_frame, 768]
        f0_target: torch.Tensor,         # [B, T_frame] Hz
        phone_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        lang_id: Optional[torch.Tensor] = None,
        variance_targets: Optional[Dict[str, torch.Tensor]] = None,
        tech: Optional[torch.Tensor] = None,  # [B, T_phone, n_techniques] Phase 3
        hubert_tokens: Optional[torch.Tensor] = None,  # [B, T_frame] long, Phase 5 discrete
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with teacher forcing.

        Returns dict with losses and metrics.
        """
        # 1. Encode score
        phone_hidden = self.encoder(
            phonemes, pitch, duration, singing_ratio,
            lang_id=lang_id, phone_mask=phone_mask, tech=tech,
        )  # [B, T_phone, d_model]

        # 2. Duration prediction loss (if enabled)
        losses = {}
        metrics = {}

        if self.predict_duration:
            dur_pred = self.duration_predictor(phone_hidden, phone_mask).squeeze(-1)
            dur_target_log = torch.log1p(duration.float())
            if phone_mask is not None:
                dur_loss = nn.functional.mse_loss(dur_pred[phone_mask], dur_target_log[phone_mask])
            else:
                dur_loss = nn.functional.mse_loss(dur_pred, dur_target_log)
            losses["duration"] = dur_loss
            metrics["dur_loss"] = dur_loss.item()

        # 3. Length regulation (expand to frame level using GT durations)
        T_frame = hubert_target.size(1)
        frame_hidden = self.length_regulator(phone_hidden, duration, max_len=T_frame)

        # 4. Variance adaptor
        frame_hidden, var_preds = self.variance_adaptor(
            frame_hidden,
            targets=variance_targets,
            mask=frame_mask,
        )

        if variance_targets:
            var_loss, var_metrics = self.variance_adaptor.compute_loss(
                var_preds, variance_targets, mask=frame_mask
            )
            losses["variance"] = var_loss
            metrics.update(var_metrics)

        # 5. F0 prediction
        # Expand note pitch to frame level
        note_pitch_frame = self._expand_to_frames(pitch, duration, T_frame)

        # Expand singing_ratio to frame level if needed
        if singing_ratio.dim() == 1:
            sr_frame = singing_ratio
        else:
            sr_frame = self._expand_to_frames(singing_ratio, duration, T_frame)

        # Derive GT voicing mask from F0 target for teacher forcing
        voicing_gt = (f0_target > self.f0_predictor.f0_min).float()

        f0_pred, voicing_pred, f0_details = self.f0_predictor(
            frame_hidden, note_pitch_frame, sr_frame, mask=frame_mask,
            voicing_target=voicing_gt,
        )

        f0_loss, f0_metrics = self.f0_predictor.compute_loss(
            f0_pred, f0_target, voicing_pred, mask=frame_mask, details=f0_details
        )
        losses["f0"] = f0_loss
        metrics.update(f0_metrics)

        # 6. Build per-frame conditions (phoneme + F0)
        f0_log = torch.log(f0_target.clamp(min=1.0) + 1e-6).unsqueeze(-1)
        f0_emb = self.f0_cond_proj(f0_log)
        phoneme_condition = frame_hidden  # already frame-level after length reg + variance

        B_, T_, D_ = hubert_target.shape

        # 7. Mode branch: deterministic head OR flow matching
        if self.mode == "deterministic":
            # 7-DET. Direct regression: det_head([frame_hidden | f0_emb]) -> HuBERT.
            # No CFG (no classifier-free guidance in deterministic regression).
            # No flow decoder forward pass (saves ~65M params worth of compute per step).
            hubert_pred = self.det_head(phoneme_condition, f0_emb, mask=frame_mask)

            if frame_mask is not None:
                m_e = frame_mask.unsqueeze(-1).float()
                det_l1 = (torch.abs(hubert_pred - hubert_target) * m_e).sum() / m_e.sum() / D_
                cos_sim = F.cosine_similarity(hubert_pred, hubert_target, dim=-1)
                m_f = frame_mask.float()
                det_cos_loss = ((1 - cos_sim) * m_f).sum() / m_f.sum()
            else:
                det_l1 = F.l1_loss(hubert_pred, hubert_target)
                det_cos_loss = (1 - F.cosine_similarity(hubert_pred, hubert_target, dim=-1)).mean()

            losses["det"] = det_l1 + 0.5 * det_cos_loss
            # Reuse 'recon_l1' / 'recon_cos_sim' metric names so the trainer's
            # checkpoint criterion (best by recon_l1) works without extra wiring.
            metrics["recon_l1"] = det_l1.item()
            metrics["recon_cos_sim"] = (1 - det_cos_loss).item()

        elif self.mode == "discrete":
            # 7-DISC. Classify each frame into a K-means codebook token.
            # Cross-entropy loss → no L1 averaging → sharp articulation.
            logits = self.discrete_head(phoneme_condition, f0_emb, mask=frame_mask)

            if hubert_tokens is None:
                raise ValueError("discrete mode requires hubert_tokens in batch")

            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.discrete_head.n_codes),
                hubert_tokens[:, :T_].reshape(-1),
                label_smoothing=0.1,
                ignore_index=-100,
            )
            losses["discrete"] = ce_loss

            # Metrics
            with torch.no_grad():
                pred_tokens = logits.argmax(dim=-1)  # [B, T]
                valid = hubert_tokens[:, :T_] >= 0
                if valid.sum() > 0:
                    correct = (pred_tokens == hubert_tokens[:, :T_]) & valid
                    metrics["token_accuracy"] = correct.sum().item() / valid.sum().item()
                    # Top-5
                    top5 = logits.topk(5, dim=-1).indices  # [B, T, 5]
                    gt_exp = hubert_tokens[:, :T_].unsqueeze(-1)  # [B, T, 1]
                    top5_hit = (top5 == gt_exp).any(dim=-1) & valid
                    metrics["top5_accuracy"] = top5_hit.sum().item() / valid.sum().item()
                else:
                    metrics["token_accuracy"] = 0.0
                    metrics["top5_accuracy"] = 0.0

                # Reconstruction metrics via codebook lookup (comparable to Phase 3/4)
                pred_features = self.discrete_head.tokens_to_features(pred_tokens)
                if frame_mask is not None:
                    m_e = frame_mask.unsqueeze(-1).float()
                    recon_l1 = (torch.abs(pred_features - hubert_target) * m_e).sum() / m_e.sum() / D_
                    cos_sim = F.cosine_similarity(pred_features, hubert_target, dim=-1)
                    m_f = frame_mask.float()
                    recon_cos = (cos_sim * m_f).sum() / m_f.sum()
                else:
                    recon_l1 = F.l1_loss(pred_features, hubert_target)
                    recon_cos = F.cosine_similarity(pred_features, hubert_target, dim=-1).mean()
                metrics["recon_l1"] = recon_l1.item()
                metrics["recon_cos_sim"] = recon_cos.item()

        elif self.mode == "hybrid":
            # 7-HYBRID. Det head produces coarse mean (DETACHED — refiner must
            # not backprop through det head). Flow refiner models the residual
            # between GT and coarse via OT-CFM.
            with torch.no_grad():
                hubert_coarse = self.det_head(phoneme_condition, f0_emb, mask=frame_mask)
            hubert_coarse = hubert_coarse.detach()

            residual_gt = hubert_target - hubert_coarse

            cond = torch.cat([
                phoneme_condition.detach(), hubert_coarse, f0_emb.detach()
            ], dim=-1)

            refiner_loss = self.flow_refiner(residual_gt, cond, mask=frame_mask)
            losses["flow_refiner"] = refiner_loss

            # Coarse metrics (always available, cheap)
            if frame_mask is not None:
                m_e = frame_mask.unsqueeze(-1).float()
                coarse_l1 = (torch.abs(hubert_coarse - hubert_target) * m_e).sum() / m_e.sum() / D_
                cos_sim = F.cosine_similarity(hubert_coarse, hubert_target, dim=-1)
                m_f = frame_mask.float()
                coarse_cos = (cos_sim * m_f).sum() / m_f.sum()
            else:
                coarse_l1 = F.l1_loss(hubert_coarse, hubert_target)
                coarse_cos = F.cosine_similarity(hubert_coarse, hubert_target, dim=-1).mean()
            metrics["recon_l1_coarse"] = coarse_l1.item()

            if not self.training:
                # Validation: sample refined HuBERT to measure actual quality
                residual_pred = self.flow_refiner.sample(cond, n_steps=4)
                hubert_refined = hubert_coarse + residual_pred
                if frame_mask is not None:
                    ref_l1 = (torch.abs(hubert_refined - hubert_target) * m_e).sum() / m_e.sum() / D_
                    ref_cos = F.cosine_similarity(hubert_refined, hubert_target, dim=-1)
                    ref_cos = (ref_cos * m_f).sum() / m_f.sum()
                else:
                    ref_l1 = F.l1_loss(hubert_refined, hubert_target)
                    ref_cos = F.cosine_similarity(hubert_refined, hubert_target, dim=-1).mean()
                metrics["recon_l1"] = ref_l1.item()
                metrics["recon_cos_sim"] = ref_cos.item()
            else:
                # Training: use coarse metrics as proxy (no expensive sampling)
                metrics["recon_l1"] = coarse_l1.item()
                metrics["recon_cos_sim"] = coarse_cos.item()

        else:
            # 7-FLOW. Original Plan A path. Kept for future hybrid (det warm-start
            # + flow refiner on residual). NOT trained in deterministic mode.
            #
            # SELECTIVE CFG: drop only the phoneme content, keep F0 in unconditional
            # path. F0 is a "hard" user-given condition that should always reach the
            # decoder; CFG should learn to amplify phoneme content (the part that
            # was being ignored in Run 1).
            if self.training and self.cfg_drop_prob > 0:
                B_drop = phoneme_condition.size(0)
                drop = torch.rand(B_drop, device=phoneme_condition.device) < self.cfg_drop_prob
                if drop.any():
                    phoneme_condition = phoneme_condition.clone()
                    phoneme_condition[drop] = 0.0

            flow_condition = phoneme_condition + f0_emb

            flow_loss, flow_metrics, pred_v, t_sampled, x_t = self.flow_decoder.compute_loss(
                hubert_target, flow_condition, mask=frame_mask
            )
            losses["flow"] = flow_loss
            metrics.update(flow_metrics)

            # 7b. Reconstruction loss: single Euler step from x_t to estimate x_1.
            t_view = t_sampled.view(B_, 1, 1)
            x_1_pred = x_t + pred_v * (1 - t_view)

            if frame_mask is not None:
                m_e = frame_mask.unsqueeze(-1).float()
                recon_l1 = (torch.abs(x_1_pred - hubert_target) * m_e).sum() / m_e.sum() / D_
                cos_sim = F.cosine_similarity(x_1_pred, hubert_target, dim=-1)
                m_f = frame_mask.float()
                recon_cos_loss = ((1 - cos_sim) * m_f).sum() / m_f.sum()
            else:
                recon_l1 = F.l1_loss(x_1_pred, hubert_target)
                recon_cos_loss = (1 - F.cosine_similarity(x_1_pred, hubert_target, dim=-1)).mean()

            losses["recon"] = recon_l1 + 0.5 * recon_cos_loss
            metrics["recon_l1"] = recon_l1.item()
            metrics["recon_cos_sim"] = (1 - recon_cos_loss).item()

        # 8. Aggregate total loss
        loss_cfg = self.config["training"]["loss"]

        if self.mode == "deterministic":
            total_loss = (
                loss_cfg.get("det", 1.0) * losses["det"]
                + loss_cfg["f0"] * losses["f0"]
            )
        elif self.mode == "discrete":
            total_loss = loss_cfg.get("discrete", 1.0) * losses["discrete"]
            if loss_cfg["f0"] > 0:
                total_loss = total_loss + loss_cfg["f0"] * losses["f0"]
        elif self.mode == "hybrid":
            total_loss = loss_cfg.get("flow_refiner", 1.0) * losses["flow_refiner"]
            if loss_cfg["f0"] > 0:
                total_loss = total_loss + loss_cfg["f0"] * losses["f0"]
        else:
            total_loss = (
                loss_cfg["flow_matching"] * losses["flow"]
                + loss_cfg["f0"] * losses["f0"]
            )
            if "recon" in losses:
                total_loss = total_loss + loss_cfg.get("reconstruction", 0.5) * losses["recon"]

        if "duration" in losses:
            total_loss = total_loss + loss_cfg.get("duration", 0.5) * losses["duration"]
        if "variance" in losses:
            var_weight = loss_cfg.get("energy", 0.1) + loss_cfg.get("breathiness", 0.1)
            total_loss = total_loss + var_weight * losses["variance"]

        # Multi-scale HuBERT loss (flow mode only — uses flow_condition)
        if self.mode != "deterministic":
            ms_cfg = loss_cfg.get("multiscale_hubert", {})
            if ms_cfg.get("enabled", False):
                ms_loss = self._multiscale_loss(
                    hubert_target, flow_condition, frame_mask,
                    scales=ms_cfg.get("scales", [1, 5, 10])
                )
                total_loss = total_loss + ms_cfg.get("weight", 0.3) * ms_loss
                metrics["ms_hubert_loss"] = ms_loss.item()

        return {
            "loss": total_loss,
            "losses": {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()},
            "metrics": metrics,
            "f0_pred": f0_pred.detach(),
            "voicing_pred": voicing_pred.detach(),
        }

    @torch.no_grad()
    def synthesize(
        self,
        phonemes: torch.Tensor,
        pitch: torch.Tensor,
        duration: torch.Tensor,
        singing_ratio: torch.Tensor,
        lang_id: Optional[torch.Tensor] = None,
        phone_mask: Optional[torch.Tensor] = None,
        n_steps: int = 10,
        solver: str = "euler",
        cfg_scale: float = 2.0,
        temperature: float = 1.0,
        pitch_curve: Optional[torch.Tensor] = None,
        tech: Optional[torch.Tensor] = None,  # [B, T_phone, n_techniques] Phase 3
    ) -> Dict[str, torch.Tensor]:
        """
        Full inference: score → HuBERT features + F0.

        Args:
            pitch_curve: [B, T_frame] optional user-drawn pitch curve (Hz).
                         If provided, overrides F0 predictor output.
            tech: [B, T_phone, n_techniques] optional per-phoneme technique flags
                  (mix_voice, falsetto, breathy, pharyngeal, vibrato, glissando).
                  None or all-zeros → natural singing (no explicit technique).

        Returns:
            dict with 'hubert' [B, T, 768], 'f0' [B, T], 'f0_coarse' [B, T]
        """
        # Zero out MIDI pitch for rap phonemes: training data has pitch=0 for
        # all speech samples, so encoder + f0_predictor expect pitch=0 when
        # singing_ratio≈0. Without this, applying rap mode to a singing sample
        # feeds note_pitch_proj values it never saw → f0 prediction goes wild.
        if singing_ratio.dim() > 1:
            # per-phoneme [B, T_phone]: zero pitch where rap
            pitch = pitch * (singing_ratio > 0.5).long()
        elif singing_ratio.dim() == 1 and singing_ratio.min() < 0.5:
            # per-sample [B]: zero pitch for fully-rap samples
            pitch = pitch * (singing_ratio > 0.5).unsqueeze(1).long()

        # 1. Encode
        phone_hidden = self.encoder(
            phonemes, pitch, duration, singing_ratio,
            lang_id=lang_id, phone_mask=phone_mask, tech=tech,
        )

        # 2. Duration (use given or predicted)
        if self.predict_duration and duration is None:
            dur_pred = self.duration_predictor(phone_hidden, phone_mask).squeeze(-1)
            duration = torch.expm1(dur_pred).round().long().clamp(min=1)

        T_frame = duration.sum(dim=1).max().item()
        frame_hidden = self.length_regulator(phone_hidden, duration, max_len=T_frame)
        frame_mask = self._make_frame_mask(duration, T_frame)

        # 3. Variance adaptor (no targets → uses predictions)
        frame_hidden, _ = self.variance_adaptor(frame_hidden, mask=frame_mask)

        # 4. F0 prediction
        note_pitch_frame = self._expand_to_frames(pitch, duration, T_frame)
        if singing_ratio.dim() == 1:
            sr_frame = singing_ratio
        else:
            sr_frame = self._expand_to_frames(singing_ratio, duration, T_frame)

        f0_pred, voicing, _ = self.f0_predictor(
            frame_hidden, note_pitch_frame, sr_frame, mask=frame_mask
        )

        # Override with user pitch curve if provided
        if pitch_curve is not None:
            # Align pitch_curve length to T_frame (duration-derived frame count)
            if pitch_curve.size(1) != T_frame:
                pitch_curve = torch.nn.functional.interpolate(
                    pitch_curve.unsqueeze(1), size=T_frame, mode="linear"
                ).squeeze(1)
            f0_pred = pitch_curve

        # Pitch correction: clamp predicted F0 to stay within ±max_cents of the
        # MIDI note. Prevents drift/跑調 while preserving vibrato and expression.
        # Skipped when user provides pitch_curve (trusted input, no correction needed).
        if pitch_curve is None:
            max_cents = 100  # ±1 semitone
            note_hz = 440.0 * (2.0 ** ((note_pitch_frame.float() - 69) / 12))
            voiced_singing = (note_pitch_frame > 0) & (f0_pred > self.f0_predictor.f0_min)
            if sr_frame.dim() > 1:
                voiced_singing = voiced_singing & (sr_frame > 0.5)
            elif sr_frame.dim() == 1:
                voiced_singing = voiced_singing & (sr_frame > 0.5).unsqueeze(1)
            if voiced_singing.any():
                ratio = f0_pred / note_hz.clamp(min=1.0)
                max_ratio = 2.0 ** (max_cents / 1200)
                min_ratio = 2.0 ** (-max_cents / 1200)
                ratio_clamped = ratio.clamp(min_ratio, max_ratio)
                f0_corrected = note_hz * ratio_clamped
                f0_pred = torch.where(voiced_singing, f0_corrected, f0_pred)

        # 5. Build per-frame conditions
        f0_log = torch.log(f0_pred.clamp(min=1.0) + 1e-6).unsqueeze(-1)
        f0_emb = self.f0_cond_proj(f0_log)

        # 6. Generate HuBERT — branch on mode
        if self.mode == "deterministic":
            hubert_pred = self.det_head(frame_hidden, f0_emb, mask=frame_mask)
        elif self.mode == "discrete":
            logits = self.discrete_head(frame_hidden, f0_emb, mask=frame_mask)
            token_ids = logits.argmax(dim=-1)  # [B, T]
            hubert_pred = self.discrete_head.tokens_to_features(token_ids)
        elif self.mode == "hybrid":
            hubert_coarse = self.det_head(frame_hidden, f0_emb, mask=frame_mask)
            cond = torch.cat([frame_hidden, hubert_coarse, f0_emb], dim=-1)
            residual = self.flow_refiner.sample(cond, n_steps=n_steps, temperature=temperature)
            hubert_pred = hubert_coarse + residual
        else:
            flow_condition = frame_hidden + f0_emb
            hubert_pred = self.flow_decoder.sample(
                flow_condition,
                mask=frame_mask,
                n_steps=n_steps,
                solver=solver,
                cfg_scale=cfg_scale,
                temperature=temperature,
            )

        # 7. Generate F0 coarse (RVC format)
        f0_coarse = self.f0_predictor.f0_to_coarse(f0_pred)

        return {
            "hubert": hubert_pred,
            "f0": f0_pred,
            "f0_coarse": f0_coarse,
            "voicing": voicing,
            "frame_mask": frame_mask,
        }

    def _expand_to_frames(self, x: torch.Tensor, duration: torch.Tensor,
                          max_len: int) -> torch.Tensor:
        """Expand phone-level tensor to frame-level using durations."""
        result = []
        for i in range(x.size(0)):
            expanded = torch.repeat_interleave(x[i], duration[i], dim=0)
            result.append(expanded)

        out = torch.zeros(x.size(0), max_len, device=x.device, dtype=x.dtype)
        for i, r in enumerate(result):
            length = min(r.size(0), max_len)
            out[i, :length] = r[:length]
        return out

    def _make_frame_mask(self, duration: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create boolean mask from durations: True for valid frames."""
        total_frames = duration.sum(dim=1)  # [B]
        mask = torch.arange(max_len, device=duration.device).unsqueeze(0) < total_frames.unsqueeze(1)
        return mask

    def _multiscale_loss(self, target: torch.Tensor, condition: torch.Tensor,
                         mask: Optional[torch.Tensor], scales: list) -> torch.Tensor:
        """
        Multi-scale auxiliary loss: computes flow matching loss at different
        temporal resolutions to enforce both local and global consistency.
        """
        total = torch.tensor(0.0, device=target.device)

        for scale in scales:
            if scale == 1:
                continue  # scale=1 is the main flow loss, already computed
            # Average pooling over time
            B, T, D = target.shape
            T_scaled = T // scale
            if T_scaled < 1:
                continue

            target_s = target[:, :T_scaled * scale, :].view(B, T_scaled, scale, D).mean(2)
            cond_s = condition[:, :T_scaled * scale, :].view(
                B, T_scaled, scale, condition.size(-1)
            ).mean(2)

            mask_s = None
            if mask is not None:
                mask_s = mask[:, :T_scaled * scale].view(B, T_scaled, scale).all(2)

            loss, _, _, _, _ = self.flow_decoder.compute_loss(target_s, cond_s, mask=mask_s)
            total = total + loss

        return total / max(len(scales) - 1, 1)

    def count_parameters(self) -> dict:
        """Count parameters per component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "encoder": _count(self.encoder),
            "variance_adaptor": _count(self.variance_adaptor),
            "f0_predictor": _count(self.f0_predictor),
            "flow_decoder": _count(self.flow_decoder),
            "det_head": _count(self.det_head),
            "f0_cond_proj": _count(self.f0_cond_proj),
        }
        if self.predict_duration:
            counts["duration_predictor"] = _count(self.duration_predictor)
        counts["total"] = sum(counts.values())
        return counts
