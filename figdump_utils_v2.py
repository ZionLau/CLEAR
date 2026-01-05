
# -*- coding: utf-8 -*-
"""
figdump_utils_v2.py
Drop-in utilities to auto-generate Figure2-4 + Table5 during evaluation.

Assumptions about your model:
- model.forward(...) stores intermediate embeddings into `model._fig_cache`:
    model._fig_cache["z_news_input"]   : torch.Tensor [B, Dn]
    model._fig_cache["z_comment_input"]: torch.Tensor [B, Dc]
  (You can add this caching in your ClassifyModel.forward)

- model.dual_purifier.news_purifier exists and provides:
    input_norm, compute_heun_solver(z_raw), prototypes (ParameterList),
    num_classes, news_dim

Figures produced (filenames use cfg["prefix"]):
- Fig2a: delta_true histogram
- Fig2b: delta_wrong histogram
- Fig2c: delta_margin histogram
- Fig3a: scatter(kinetic_energy, vacuity)
- Fig3b: scatter(kinetic_energy, dirichlet_entropy)
- Fig4a: histogram(pos_logsim) + histogram(neg_logsim) (saved as two separate images)
- Fig4b: histogram(gap_per_sample)
- Table5: CSV summary of SoftREPA diagnostics

All files are saved into cfg["out_dir"].
"""
import os
import math
import csv
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _dirichlet_entropy(alpha: torch.Tensor) -> torch.Tensor:
    """
    Dirichlet entropy H[Dir(alpha)] per sample.
    alpha: [B, K], alpha>0
    returns: [B]
    """
    # H = ln B(alpha) + (alpha0-K) psi(alpha0) - sum_k (alpha_k-1) psi(alpha_k)
    alpha0 = alpha.sum(dim=1, keepdim=True)  # [B,1]
    K = alpha.size(1)

    lnB = torch.lgamma(alpha).sum(dim=1) - torch.lgamma(alpha0.squeeze(1))  # [B]
    term = (alpha0.squeeze(1) - K) * torch.digamma(alpha0.squeeze(1)) - ((alpha - 1.0) * torch.digamma(alpha)).sum(dim=1)
    return lnB + term


def _collect_fig23_batch(model, alpha, labels):
    """
    Collect Fig2/Fig3 related stats.
    Returns both:
      - distance-based deltas (old)
      - energy/score-based deltas (new, recommended)
    """
    purifier = _get_news_purifier(model)

    # ---- device safety ----
    if torch.is_tensor(labels):
        labels = labels.to(alpha.device)
    labels = labels.long()

    z_news_input = getattr(model, "_fig_cache", {}).get("z_news_input", None)
    if z_news_input is None:
        return None

    # z_raw in purifier space
    z_raw = purifier.input_norm(z_news_input)

    # heun purification (inference-style)
    z_pure, kinetic_energy = purifier.compute_heun_solver(z_raw)

    # ====== (1) distance-based (old) ======
    d_raw = torch.cdist(z_raw, torch.cat([p for p in purifier.prototypes], dim=0))  # [B, K*M]
    d_pure = torch.cdist(z_pure, torch.cat([p for p in purifier.prototypes], dim=0))

    # map label -> indices for gather (approx: use min over each class's prototypes)
    K = len(purifier.prototypes)
    # reshape to [B,K,M]
    M = purifier.prototypes[0].shape[0]
    d_raw_km = d_raw.view(d_raw.size(0), K, M)
    d_pure_km = d_pure.view(d_pure.size(0), K, M)

    d_raw_k = d_raw_km.min(dim=2).values   # [B,K]
    d_pure_k = d_pure_km.min(dim=2).values # [B,K]

    idx = labels.view(-1, 1)
    d_true_raw = d_raw_k.gather(1, idx).squeeze(1)
    d_true_pure = d_pure_k.gather(1, idx).squeeze(1)

    # best wrong class (by distance: closest wrong)
    mask = torch.ones_like(d_raw_k, dtype=torch.bool)
    mask.scatter_(1, idx, False)
    d_wrong_raw = d_raw_k.masked_fill(~mask, 1e9).min(dim=1).values
    d_wrong_pure = d_pure_k.masked_fill(~mask, 1e9).min(dim=1).values

    delta_true = (d_true_raw - d_true_pure)
    delta_wrong = (d_wrong_pure - d_wrong_raw)
    margin_raw = (d_wrong_raw - d_true_raw)
    margin_pure = (d_wrong_pure - d_true_pure)
    delta_margin = (margin_pure - margin_raw)

    # ====== (2) energy/score-based (NEW, recommended) ======
    # term_geo: closer => less negative (closer to 0)
    term_geo_raw = purifier.compute_geometric_potential(z_raw)   # [B,K]
    term_geo_pure = purifier.compute_geometric_potential(z_pure) # [B,K]

    beta = purifier.beta_params.unsqueeze(0).expand(z_raw.size(0), -1)
    gamma = F.softplus(purifier.gamma)  # scalar

    # raw kinetic term: instant energy at t=0 (training-style)
    v0 = purifier.get_news_velocity(z_raw, 0.0)
    instant_energy = torch.mean(v0 ** 2, dim=-1)  # [B]
    term_kin_raw = (-gamma * instant_energy).unsqueeze(1).expand_as(term_geo_raw)

    # pure kinetic term: heun kinetic energy (inference-style)
    term_kin_pure = (-gamma * kinetic_energy).unsqueeze(1).expand_as(term_geo_pure)

    score_raw = term_geo_raw + term_kin_raw + beta
    score_pure = term_geo_pure + term_kin_pure + beta

    score_true_raw = score_raw.gather(1, idx).squeeze(1)
    score_true_pure = score_pure.gather(1, idx).squeeze(1)

    # best wrong class (highest competing score)
    mask_s = torch.ones_like(score_raw, dtype=torch.bool)
    mask_s.scatter_(1, idx, False)
    score_wrong_raw = score_raw.masked_fill(~mask_s, -1e9).max(dim=1).values
    score_wrong_pure = score_pure.masked_fill(~mask_s, -1e9).max(dim=1).values

    delta_true_score = score_true_pure - score_true_raw
    delta_wrong_score = score_wrong_pure - score_wrong_raw
    margin_score_raw = score_true_raw - score_wrong_raw
    margin_score_pure = score_true_pure - score_wrong_pure
    delta_margin_score = margin_score_pure - margin_score_raw

    # ====== Fig3 stats ======
    vacuity = (score_raw.size(1) / torch.clamp(alpha.sum(dim=1), min=1e-8))
    ent = _dirichlet_entropy(alpha)

    return {
        # old (distance) Fig2
        "delta_true": delta_true.detach().cpu().numpy(),
        "delta_wrong": delta_wrong.detach().cpu().numpy(),
        "delta_margin": delta_margin.detach().cpu().numpy(),

        # new (energy/score) Fig2 (USE THIS)
        "delta_true_score": delta_true_score.detach().cpu().numpy(),
        "delta_wrong_score": delta_wrong_score.detach().cpu().numpy(),
        "delta_margin_score": delta_margin_score.detach().cpu().numpy(),

        # Fig3
        "kinetic": kinetic_energy.detach().cpu().numpy(),
        "vacuity": vacuity.detach().cpu().numpy(),
        "entropy": ent.detach().cpu().numpy(),
    }


def _softrepa_logsim_matrix_subsample(
    model,
    labels: torch.Tensor,
    tau: float = 0.05,
    subsample_n: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SoftREPA-style log-similarity matrix on a subsample (n x n)
    using cached z_news_input & z_comment_input.

    Returns:
      pos_logsim: [n] diag
      neg_logsim: [n*(n-1)] off-diagonal
      gap_per_sample: [n] pos - mean(neg row)
    """
    assert hasattr(model, "_fig_cache") and "z_news_input" in model._fig_cache and "z_comment_input" in model._fig_cache, \
        "Missing model._fig_cache['z_news_input'/'z_comment_input']. Add caching in model.forward."

    z_news = model._fig_cache["z_news_input"]
    z_comm = model._fig_cache["z_comment_input"]

    B = z_news.size(0)
    n = min(int(subsample_n), int(B))
    if n <= 1:
        return np.array([]), np.array([]), np.array([])

    z_news = z_news[:n]
    z_comm = z_comm[:n]
    y = labels[:n]

    purifier = model.dual_purifier.news_purifier

    with torch.no_grad():
        z_raw = purifier.input_norm(z_news)                                 # [n,D]
        z_target = purifier.get_target_distribution(y).detach()             # [n,D]

        # deterministic t for diagnostics (stable & interpretable)
        t = torch.full((n, 1), 0.5, device=z_raw.device, dtype=z_raw.dtype)
        z_t = (1 - t) * z_raw + t * z_target                                # [n,D]
        u_true = z_target - z_raw                                           # [n,D]

        # build pairwise (n,n) in chunks to avoid OOM
        # For each i, compute v_pred for all j with cond = comment_j
        logsim = torch.empty((n, n), device=z_raw.device, dtype=z_raw.dtype)

        # pre-expand common terms
        z_t_i = z_t  # [n,D]
        u_i = u_true # [n,D]
        t_flat = t.squeeze(1)  # [n]

        # chunk over j
        chunk = 32
        for j0 in range(0, n, chunk):
            j1 = min(n, j0 + chunk)
            c_chunk = z_comm[j0:j1]  # [cj, Dc]

            # We want for each i, pair with each cond in chunk
            # Create flattened tensors of shape [n*cj, ...]
            cj = j1 - j0
            z_t_flat = z_t_i.unsqueeze(1).expand(n, cj, -1).reshape(n * cj, -1)
            u_flat = u_i.unsqueeze(1).expand(n, cj, -1).reshape(n * cj, -1)
            t_pair = t_flat.unsqueeze(1).expand(n, cj).reshape(n * cj)
            c_flat = c_chunk.unsqueeze(0).expand(n, cj, -1).reshape(n * cj, -1)

            v_pred = purifier.get_news_velocity(z_t_flat, t_pair, cond=c_flat)  # [n*cj, D]
            mse = (v_pred - u_flat).pow(2).mean(dim=-1).reshape(n, cj)          # [n,cj]

            # log-similarity = -mse/tau  (avoid exp underflow)
            logsim[:, j0:j1] = -mse / float(tau)

        pos = torch.diag(logsim)  # [n]
        # off-diagonal
        neg_mask = ~torch.eye(n, device=logsim.device, dtype=torch.bool)
        neg = logsim[neg_mask]  # [n*(n-1)]
        # per-sample gap
        row_mean_neg = (logsim.masked_fill(~neg_mask, 0.0).sum(dim=1) / (n - 1))
        gap = pos - row_mean_neg

    return (
        pos.detach().cpu().numpy(),
        neg.detach().cpu().numpy(),
        gap.detach().cpu().numpy(),
    )


def _plot_hist(values: np.ndarray, path: str, title: str, xlabel: str, bins: int = 50):
    plt.figure()
    if values.size > 0:
        plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_scatter(x: np.ndarray, y: np.ndarray, path: str, title: str, xlabel: str, ylabel: str):
    plt.figure()
    if x.size > 0 and y.size > 0:
        plt.scatter(x, y, s=6, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


@dataclass
class FigDumpRunner:
    cfg: Dict[str, Any]
    _batches_seen: int = 0
    _fig23: List[Dict[str, np.ndarray]] = field(default_factory=list)
    _pos_logsim: List[np.ndarray] = field(default_factory=list)
    _neg_logsim: List[np.ndarray] = field(default_factory=list)
    _gap: List[np.ndarray] = field(default_factory=list)

    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def max_batches(self) -> Optional[int]:
        mb = self.cfg.get("max_batches", None)
        if mb is None:
            return None
        try:
            mb = int(mb)
        except Exception:
            return None
        if mb <= 0:
            return None
        return mb

    def out_dir(self) -> str:
        return _ensure_dir(str(self.cfg.get("out_dir", "figs")))

    def prefix(self) -> str:
        return str(self.cfg.get("prefix", ""))

    def step(self, model, alpha: torch.Tensor, labels: torch.Tensor):
        """
        Call this once per eval batch (after forward).
        """
        if not self.enabled():
            return

        mb = self.max_batches()
        if mb is not None and self._batches_seen >= mb:
            return

        # Figure 2/3 stats
        if self.cfg.get("do_fig23", True):
            self._fig23.append(_collect_fig23_batch(model, alpha, labels))

        # Figure 4/5 stats
        if self.cfg.get("do_fig45", True):
            tau = float(self.cfg.get("tau", 0.05))
            subsample_n = int(self.cfg.get("softrepa_subsample", 64))
            pos, neg, gap = _softrepa_logsim_matrix_subsample(model, labels, tau=tau, subsample_n=subsample_n)
            if pos.size > 0:
                self._pos_logsim.append(pos)
                self._neg_logsim.append(neg)
                self._gap.append(gap)

        self._batches_seen += 1

    def finalize(self) -> Optional[Dict[str, str]]:
        """
        Write Figure2-4 + Table5 into out_dir and return paths.
        """
        if not self.enabled():
            return None

        out_dir = self.out_dir()
        prefix = self.prefix()

        out_paths: Dict[str, str] = {}

        # --- Fig2/3 ---
        if len(self._fig23) > 0:
            delta_true = np.concatenate([x["delta_true"] for x in self._fig23])
            delta_wrong = np.concatenate([x["delta_wrong"] for x in self._fig23])
            delta_margin = np.concatenate([x["delta_margin"] for x in self._fig23])
            ke = np.concatenate([x["kinetic_energy"] for x in self._fig23])
            vacuity = np.concatenate([x["vacuity"] for x in self._fig23])
            ent = np.concatenate([x["dirichlet_entropy"] for x in self._fig23])

            p2a = os.path.join(out_dir, f"{prefix}fig2a_delta_true.png")
            p2b = os.path.join(out_dir, f"{prefix}fig2b_delta_wrong.png")
            p2c = os.path.join(out_dir, f"{prefix}fig2c_delta_margin.png")
            _plot_hist(delta_true, p2a, "Fig2a: Purification brings closer to TRUE class", "Δ_true = d_true_raw - d_true_pure")
            _plot_hist(delta_wrong, p2b, "Fig2b: Purification pushes away from WRONG class", "Δ_wrong = d_wrong_pure - d_wrong_raw")
            _plot_hist(delta_margin, p2c, "Fig2c: Purification improves separation margin", "Δ_margin = (d_wrong-d_true)_pure - (d_wrong-d_true)_raw")
            out_paths.update({"fig2a": p2a, "fig2b": p2b, "fig2c": p2c})

            p3a = os.path.join(out_dir, f"{prefix}fig3a_ke_vs_vacuity.png")
            p3b = os.path.join(out_dir, f"{prefix}fig3b_ke_vs_entropy.png")
            _plot_scatter(ke, vacuity, p3a, "Fig3a: Kinetic energy vs vacuity", "Kinetic energy", "Vacuity = K / sum(alpha)")
            _plot_scatter(ke, ent, p3b, "Fig3b: Kinetic energy vs Dirichlet entropy", "Kinetic energy", "Dirichlet entropy")
            out_paths.update({"fig3a": p3a, "fig3b": p3b})

        # --- Fig4 & Table5 ---
        if len(self._pos_logsim) > 0:
            pos_all = np.concatenate(self._pos_logsim)
            neg_all = np.concatenate(self._neg_logsim) if len(self._neg_logsim) else np.array([])
            gap_all = np.concatenate(self._gap) if len(self._gap) else np.array([])

            p4pos = os.path.join(out_dir, f"{prefix}fig4a_pos_logsim_hist.png")
            p4neg = os.path.join(out_dir, f"{prefix}fig4a_neg_logsim_hist.png")
            p4gap = os.path.join(out_dir, f"{prefix}fig4b_gap_hist.png")
            _plot_hist(pos_all, p4pos, "Fig4a: SoftREPA log-sim (positive pairs)", "logsim = -mse/tau")
            _plot_hist(neg_all, p4neg, "Fig4a: SoftREPA log-sim (negative pairs)", "logsim = -mse/tau")
            _plot_hist(gap_all, p4gap, "Fig4b: SoftREPA separation gap per sample", "gap = logsim(i,i) - mean_j!=i logsim(i,j)")
            out_paths.update({"fig4_pos": p4pos, "fig4_neg": p4neg, "fig4_gap": p4gap})

            # Table5: summary
            csv_path = os.path.join(out_dir, f"{prefix}table5_softrepa_summary.csv")
            rows = [
                ("pos_logsim_mean", float(np.mean(pos_all)) if pos_all.size else math.nan),
                ("pos_logsim_std", float(np.std(pos_all)) if pos_all.size else math.nan),
                ("neg_logsim_mean", float(np.mean(neg_all)) if neg_all.size else math.nan),
                ("neg_logsim_std", float(np.std(neg_all)) if neg_all.size else math.nan),
                ("gap_mean", float(np.mean(gap_all)) if gap_all.size else math.nan),
                ("gap_std", float(np.std(gap_all)) if gap_all.size else math.nan),
                ("n_pos", int(pos_all.size)),
                ("n_neg", int(neg_all.size)),
                ("n_gap", int(gap_all.size)),
                ("tau", float(self.cfg.get("tau", 0.05))),
                ("softrepa_subsample", int(self.cfg.get("softrepa_subsample", 64))),
                ("max_batches_used", int(self._batches_seen)),
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                for k, v in rows:
                    w.writerow([k, v])
            out_paths["table5"] = csv_path

        return out_paths
