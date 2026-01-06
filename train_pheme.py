import time
import math
import pandas as pd
import torch
# import shap
from torch import nn
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from torch.utils.data import TensorDataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch.nn.functional as F
try:
    from scipy.optimize import linear_sum_assignment  # Hungarian assignment
except Exception:
    linear_sum_assignment = None
# from torchviz import make_dot
import ast
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from scipy.spatial.distance import jensenshannon
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, recall_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from torch.distributions import Dirichlet, kl_divergence
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from figdump_utils_v2 import FigDumpRunner
import csv
import atexit

@torch.no_grad()
def sliced_w2_squared(x: torch.Tensor, y: torch.Tensor, num_proj: int = 32) -> torch.Tensor:

    assert x.dim() == 2 and y.dim() == 2 and x.shape == y.shape
    B, D = x.shape
    dirs = torch.randn(num_proj, D, device=x.device, dtype=x.dtype)
    dirs = F.normalize(dirs, dim=1)  # [P, D]
    x_proj = x @ dirs.t()  # [B, P]
    y_proj = y @ dirs.t()
    x_sort, _ = torch.sort(x_proj, dim=0)
    y_sort, _ = torch.sort(y_proj, dim=0)
    return (x_sort - y_sort).pow(2).mean()

class SimpleCSVLogger:

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._writer = None
        self._fieldnames = None
        self._fp = None

    def write(self, row: dict):
        # Lazily open and create header using the first row's keys
        if self._fp is None:
            file_exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
            self._fp = open(self.path, "a", newline="", encoding="utf-8")
            if not file_exists:
                self._fieldnames = list(row.keys())
                self._writer = csv.DictWriter(self._fp, fieldnames=self._fieldnames)
                self._writer.writeheader()
            else:
                # If appending to an existing file, assume its header matches this run.
                # (Keep it simple; use a new filename if you change schema.)
                self._fieldnames = list(row.keys())
                self._writer = csv.DictWriter(self._fp, fieldnames=self._fieldnames)

        self._writer.writerow(row)
        self._fp.flush()

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None
from figdump_utils_v2 import FigDumpRunner

class NewsFeaturePurification(nn.Module):


    def __init__(self, news_dim, num_classes, num_prototypes_per_class=1, hidden_dim=128):
        super().__init__()
        self.news_dim = news_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class


        # --- OT-FM schedule (classic FM warm-up then OT-FM) ---
        self.ot_start_epoch = 14  # use classic FM for epochs < this
        self._cur_epoch = 0

        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(num_prototypes_per_class, news_dim))
            for _ in range(num_classes)
        ])


        self.news_velocity_net = nn.Sequential(
            nn.Linear(news_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, news_dim)
        )


        self.lambda_params = nn.Parameter(torch.full((num_classes,), 0.5))  # λ_k
        self.beta_params = nn.Parameter(torch.full((num_classes,), 2.0))  # β_k
        self.gamma = nn.Parameter(torch.tensor(1.0))  # γ

        # 归一化
        self.input_norm = nn.LayerNorm(news_dim)

    def get_target_distribution(self, labels):

        batch_size = labels.size(0)
        z_target = torch.zeros(batch_size, self.news_dim, device=labels.device)

        for k in range(self.num_classes):
            mask = (labels == k)
            if mask.sum() > 0:

                prototype_indices = torch.randint(0, self.num_prototypes_per_class,
                                                  (mask.sum(),), device=labels.device)
                selected_prototypes = self.prototypes[k][prototype_indices]


                noise = torch.randn_like(selected_prototypes) * 0.1
                z_target[mask] = selected_prototypes + noise

        return z_target
    def set_epoch(self, epoch: int):
        """Update current epoch for FM/OT-FM switching."""
        self._cur_epoch = int(epoch)

    def get_news_velocity(self, z, t):

        if isinstance(t, (float, int)):
            t = torch.full((z.shape[0], 1), float(t), device=z.device)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        inp = torch.cat([z, t], dim=-1)
        return self.news_velocity_net(inp)


    def _ot_match_targets(self, z_raw, z_target, labels):
        """Discrete OT coupling (within-class) to match each z_raw to one target sample.

        For each class k, solve a minimum-cost bipartite assignment between:
          - sources: {z_raw_i | y_i = k}
          - targets: {z_target_j | y_j = k}
        under squared Euclidean cost, yielding a permutation of targets.
        This implements the OT-FM pairing step (q*) while keeping the same target marginal.
        """
        z_raw_det = z_raw.detach()
        z_tar_det = z_target.detach()
        labels_1d = labels.view(-1)

        z_target_perm = z_target.clone()

        for k in range(self.num_classes):
            idx = (labels_1d == k).nonzero(as_tuple=False).view(-1)
            n = int(idx.numel())
            if n <= 1:
                continue

            z0 = z_raw_det[idx]   # [n, D]
            z1 = z_tar_det[idx]   # [n, D]

            with torch.no_grad():
                cost = torch.cdist(z0.float(), z1.float(), p=2).pow(2)  # [n, n]
                cost_np = cost.cpu().numpy()

            if linear_sum_assignment is None:
                # Fallback: greedy matching (approximate OT)
                remaining = set(range(n))
                col_ind_list = [-1] * n
                for r in range(n):
                    best_c = min(remaining, key=lambda c: cost_np[r][c])
                    col_ind_list[r] = best_c
                    remaining.remove(best_c)
                col_ind = torch.tensor(col_ind_list, device=z_target.device, dtype=torch.long)
            else:
                _, col_ind_np = linear_sum_assignment(cost_np)
                col_ind = torch.tensor(col_ind_np, device=z_target.device, dtype=torch.long)

            z_target_perm[idx] = z_target[idx][col_ind]

        return z_target_perm
    def forward_train(self, z_raw, labels, return_stats: bool = False, return_per_sample: bool = False):

        z_raw = self.input_norm(z_raw)

        z_target = self.get_target_distribution(labels)
        # OT-FM pairing (enabled after warm-up): within-class optimal coupling
        if getattr(self, '_cur_epoch', 0) >= getattr(self, 'ot_start_epoch', 0):
            z_target = self._ot_match_targets(z_raw, z_target, labels)
        batch_size = z_raw.size(0)
        t = torch.rand(batch_size, 1, device=z_raw.device)

        z_t = (1 - t) * z_raw + t * z_target
        u_true = z_target - z_raw

        t_expanded = t.squeeze(1)
        v_pred = self.get_news_velocity(z_t, t_expanded)

        resid = v_pred - u_true

        # per-sample MSE, shape [B]
        loss_per_sample = (resid ** 2).mean(dim=1)

        # label-based reweighting for class imbalance (upweight class 1)
        # Scheme C: no FM label-weight for the first 7 epochs, then use a mild weight (1.5) for class 1.
        labels_1d = labels.view(-1)
        label_w = torch.ones_like(loss_per_sample, dtype=loss_per_sample.dtype)

        cur_epoch = int(getattr(self, '_cur_epoch', 0))
        w1 = 1.0 if cur_epoch <= 5 else 1.3  # 1.5
        label_w[labels_1d == 1] = w1

        loss = torch.sum(loss_per_sample * label_w) / (torch.sum(label_w) + 1e-8)
        if (not return_stats) and (not return_per_sample):
            return loss

        if return_stats:
            with torch.no_grad():
                stats = {
                    "fm_mse": float(loss.detach().cpu()),
                    "u_norm": float(u_true.norm(dim=1).mean().detach().cpu()),
                    "v_norm": float(v_pred.norm(dim=1).mean().detach().cpu()),
                    "resid_norm": float(resid.norm(dim=1).mean().detach().cpu()),
                    "t_mean": float(t.mean().detach().cpu()),
                    "swd2_raw_target": float(sliced_w2_squared(z_raw.detach(), z_target.detach(), num_proj=32).detach().cpu()),
                }
        else:
            stats = None

        if return_per_sample and return_stats:
            return loss, loss_per_sample, stats
        if return_per_sample:
            return loss, loss_per_sample
        return loss, stats

    def compute_heun_solver(self, z_raw):

        dt = 1.0  # Δt = 1 (从0到1)


        v0 = self.get_news_velocity(z_raw, 0.0)
        z_guess = z_raw + v0 * dt


        v1 = self.get_news_velocity(z_guess, 1.0)
        z_pure = z_raw + (dt / 2) * (v0 + v1)


        kinetic_energy = (torch.mean(v0 ** 2, dim=-1) + torch.mean(v1 ** 2, dim=-1)) / 2

        return z_pure, kinetic_energy

    def compute_geometric_potential(self, z_pure):
        term_geo_list = []
        for k in range(self.num_classes):

            dists = torch.cdist(z_pure, self.prototypes[k])
            min_dist, _ = torch.min(dists, dim=1)


            # Mean Squared Error = (L2 Distance)^2 / Dimensions
            mse_dist = (min_dist ** 2) / self.news_dim


            geo_potential = -F.softplus(self.lambda_params[k]) * mse_dist
            term_geo_list.append(geo_potential)
        return torch.stack(term_geo_list, dim=1)

    def prototype_class_mindist2(self, z_raw_norm: torch.Tensor) -> torch.Tensor:
        """Min squared distance to each class prototype cluster.

        Args:
            z_raw_norm: [B, D] normalized embedding (same space as prototypes)

        Returns:
            dist2: [B, K], where dist2[b,k] = min_h ||z_b - p_{k,h}||^2
        """
        # Stack prototypes -> [K, H, D]
        protos = []
        for k in range(self.num_classes):
            pk = self.prototypes[k]
            if pk.dim() == 1:
                pk = pk.unsqueeze(0)
            protos.append(pk)
        protos = torch.stack(protos, dim=0)

        # Compute squared distances -> [B, K, H]
        z = z_raw_norm.unsqueeze(1).unsqueeze(2)
        dist2 = ((z - protos.unsqueeze(0)) ** 2).sum(dim=-1)
        return dist2.min(dim=-1).values  # [B, K]


    def predict(self, z_news_input, return_stats: bool = False):

        z_raw = self.input_norm(z_news_input)
        batch_size = z_raw.size(0)

        if not self.training:

            z_pure, kinetic_energy = self.compute_heun_solver(z_raw)


            term_geo = self.compute_geometric_potential(z_pure)


            term_kin = -F.softplus(self.gamma) * kinetic_energy.unsqueeze(1)  # [B, 1]
            term_kin = term_kin.expand(-1, self.num_classes)  # [B, K]


            term_beta = self.beta_params.unsqueeze(0).expand(z_raw.size(0), -1)  # [B, K]

        else:

            term_geo = self.compute_geometric_potential(z_raw)

            v0 = self.get_news_velocity(z_raw, 0.0)


            instant_energy = torch.mean(v0 ** 2, dim=-1)

            term_kin = -F.softplus(self.gamma) * instant_energy.unsqueeze(1).expand(-1, self.num_classes)

            term_beta = self.beta_params.unsqueeze(0).expand(batch_size, -1)

        final_logits = term_geo + term_kin + term_beta
        final_logits = torch.clamp(final_logits, min=-10.0, max=10.0)

        alpha = F.softplus(final_logits) + 1


        if return_stats:
            stats = {}

            stats['_z_raw_norm'] = z_raw.detach()

            if self.training:
                stats['_instant_energy'] = instant_energy.detach()
            else:
                stats['_instant_energy'] = kinetic_energy.detach() if 'kinetic_energy' in locals() else torch.zeros(z_raw.size(0), device=z_raw.device)
            return alpha, stats

        return alpha



class DualStreamFusionModel(nn.Module):
    def __init__(self, news_dim, comment_dim, sentiment_counts_dim=8,
                 hidden_dim=128, num_heads=3, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # === 新闻分支 (使用修正后的净化模块) ===
        self.news_purifier = NewsFeaturePurification(
            news_dim=news_dim,
            num_classes=num_classes,
            num_prototypes_per_class=num_heads,
            hidden_dim=hidden_dim
        )


    def compute_aux_losses(self, z_news_input, z_comment_input, y_veracity, return_stats: bool = False,):
        """
        计算辅助Loss - 对应论文的优化目标
        """

        fm_loss_per_sample = None
        if return_stats:
            loss_fm_news, fm_loss_per_sample, fm_stats = self.news_purifier.forward_train(
                z_news_input, y_veracity, return_stats=True, return_per_sample=True
            )
        else:
            fm_stats = None
            loss_fm_news, fm_loss_per_sample = self.news_purifier.forward_train(
                z_news_input, y_veracity, return_per_sample=True
            )



        prototypes = torch.cat([p.unsqueeze(0) for p in self.news_purifier.prototypes], dim=0)
        loss_proto = 0.0
        for k1 in range(self.num_classes):
            for k2 in range(k1 + 1, self.num_classes):
                p1 = F.normalize(prototypes[k1], dim=1)  # [M, D]
                p2 = F.normalize(prototypes[k2], dim=1)  # [M, D]
                similarity = torch.mm(p1, p2.t())  # [M, M]
                loss_proto += torch.mean((similarity + 1) ** 2)  # 鼓励正交


        loss_fm_senti = torch.tensor(0.0, device=z_news_input.device)
        # loss_fm_senti = self.sentiment_rectifier.compute_loss(z_comment_input, y_counts)
        if return_stats:
            return loss_fm_news, loss_proto, loss_fm_senti, fm_stats, fm_loss_per_sample
        return loss_fm_news, loss_proto, loss_fm_senti, fm_loss_per_sample
    def predict(self, z_news_input, z_comment_input=None, sentiment_target=None, return_stats: bool = False):

        if return_stats:
            dirichlet_params, stats = self.news_purifier.predict(z_news_input, return_stats=True)
        else:
            dirichlet_params = self.news_purifier.predict(z_news_input)
            stats = None


        if True:
            print("\n" + "=" * 30 + " DEBUG MONITOR " + "=" * 30)
            with torch.no_grad():
                idx = 0
                alphas = dirichlet_params[idx].detach().cpu().numpy()
                print(f"DEBUG [Sample 0] Dirichlet parameters (α_k): {alphas}")


                lam = F.softplus(self.news_purifier.lambda_params).detach().cpu().numpy()
                beta = self.news_purifier.beta_params.detach().cpu().numpy()
                gamma = F.softplus(self.news_purifier.gamma).detach().cpu().numpy()
                print(f"DEBUG Params -> Lambda: {lam} | Beta: {beta} | Gamma: {gamma}")

                if not self.training:

                    z_raw = self.news_purifier.input_norm(z_news_input)
                    z_pure, kinetic_energy = self.news_purifier.compute_heun_solver(z_raw)

                    for k in range(self.num_classes):
                        dists = torch.cdist(z_pure[idx:idx + 1], self.news_purifier.prototypes[k])
                        min_dist = torch.min(dists).item()
                        print(f"DEBUG Class {k} Min Distance: {min_dist:.4f}")

                    ke = kinetic_energy[idx].item()
                    print(f"DEBUG Kinetic Energy: {ke:.4f}")

            print("=" * 75 + "\n")

        return (dirichlet_params, stats) if return_stats else dirichlet_params

class ClassifyModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, news_labels, comments_labels, is_lock=False):
        super(ClassifyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        config = self.bert.config
        if is_lock:
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.news_classifier = nn.Linear(5 * config.hidden_size, news_labels)
        # self.dirichlet_predictor = nn.Linear(5 * config.hidden_size, news_labels)
        self.comment_linear = nn.Linear(8 * config.hidden_size, 5 * config.hidden_size)

        self.dual_purifier = DualStreamFusionModel(
            news_dim=5 * config.hidden_size,
            comment_dim=5 * config.hidden_size,
            sentiment_counts_dim=comments_labels,
            hidden_dim=256,
            num_heads=3,
            num_classes=news_labels
        )

        self.contrastive_predictor = nn.Linear(5 * config.hidden_size, 4 * config.hidden_size)

        self.dirichlet_predictor_s1 = nn.Linear(2 * config.hidden_size, comments_labels)
        self.dirichlet_predictor_s2 = nn.Linear(2 * config.hidden_size, comments_labels)
        self.dirichlet_predictor_s3 = nn.Linear(2 * config.hidden_size, comments_labels)
        self.dirichlet_predictor_s4 = nn.Linear(2 * config.hidden_size, comments_labels)
        self.dirichlet_predictor_s5 = nn.Linear(2 * config.hidden_size, comments_labels)
        self.comments_labels = comments_labels

        self.w_gcb_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size)))
        self.w_gca_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size)))
        self.w_fcb_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.w_fca_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.w_gsb_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size)))
        self.w_gsa_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size)))
        self.w_fsb_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.w_fsa_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.w_tempp = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))

        self.batch_norm_temp0 = nn.BatchNorm1d(config.hidden_size)
        self.batch_norm_temp2 = nn.BatchNorm1d(config.hidden_size)
        self.batch_norm_tmp0 = nn.BatchNorm1d(config.hidden_size)
        self.batch_norm_tmp2 = nn.BatchNorm1d(config.hidden_size)

        self.w_gcb_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size)))
        self.w_gca_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size)))
        self.w_fcb_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.w_fca_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.w_gsb_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size)))
        self.w_gsa_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size)))
        self.w_fsb_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.w_fsa_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.w_tempc = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.batch_norm_temp0c = nn.BatchNorm1d(config.hidden_size)
        self.batch_norm_temp2c = nn.BatchNorm1d(config.hidden_size)
        self.batch_norm_tmp0c = nn.BatchNorm1d(config.hidden_size)
        self.batch_norm_tmp2c = nn.BatchNorm1d(config.hidden_size)

        self.w_r = nn.Parameter(nn.init.xavier_uniform_(torch.ones(4 * config.hidden_size, 2 * config.hidden_size)))
        self.w_z = nn.Parameter(nn.init.xavier_uniform_(torch.ones(4 * config.hidden_size, 2 * config.hidden_size)))
        self.w_h = nn.Parameter(nn.init.xavier_uniform_(torch.ones(4 * config.hidden_size, 2 * config.hidden_size)))
        self.batch_norm_r = nn.BatchNorm1d(2 * config.hidden_size)
        self.batch_norm_z = nn.BatchNorm1d(2 * config.hidden_size)
        self.w_com1 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(4 * config.hidden_size, 2 * config.hidden_size)))
        self.w_com2 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(4 * config.hidden_size, 2 * config.hidden_size)))
        self.w_com3 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(4 * config.hidden_size, 2 * config.hidden_size)))
        self.w_com4 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(4 * config.hidden_size, 2 * config.hidden_size)))
        self.w_com5 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(4 * config.hidden_size, 2 * config.hidden_size)))

        self.blbl_1 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size // 2)))
        self.blbl_2 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size // 2)))
        self.blbl_3 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.blbl_4 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.blbl_5 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size // 2)))
        self.blbl_6 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, config.hidden_size // 2)))
        self.blbl_7 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(config.hidden_size, 2 * config.hidden_size)))
        self.blbl_8 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(2 * config.hidden_size, config.hidden_size)))
        self.blbl_7680 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(8 * config.hidden_size, config.hidden_size)))
        self.batch_norm_blbl_1 = nn.BatchNorm1d(config.hidden_size)
        self.batch_norm_blbl_2 = nn.BatchNorm1d(2 * config.hidden_size)
        self.batch_norm_blbl_3 = nn.BatchNorm1d(config.hidden_size)
        self.batch_norm_blbl_4 = nn.BatchNorm1d(config.hidden_size)

    def HGD_FN_Post(self, message_1, message_2):
        phi_ci = F.leaky_relu(torch.cat(
            (torch.matmul(message_1, self.w_gcb_p), torch.matmul(message_2, self.w_gca_p)), dim=1), negative_slope=0.01)
        # print("Grad_fn of phi_ci: ", phi_ci.grad_fn)

        temp0 = torch.matmul(phi_ci, self.w_fcb_p)
        temp0 = self.batch_norm_temp0(temp0)  # Apply Batch Normalization
        # print("Grad_fn of temp0: ", temp0.grad_fn)

        temp1 = torch.mul(message_1, temp0)
        # print("Grad_fn of temp1: ", temp1.grad_fn)

        temp2 = torch.matmul((1 - phi_ci), self.w_fca_p)
        temp2 = self.batch_norm_temp2(temp2)  # Apply Batch Normalization
        # print("Grad_fn of temp2: ", temp2.grad_fn)

        temp3 = torch.mul(message_2, temp2)
        # print("Grad_fn of temp3: ", temp3.grad_fn)

        F_Ci = torch.tanh(temp1 + temp3)
        # print("Grad_fn of F_Ci: ", F_Ci.grad_fn)

        phi_si = F.leaky_relu(torch.cat(
            (torch.matmul(message_1, self.w_gsb_p), torch.matmul(message_2, self.w_gsa_p)), dim=1), negative_slope=0.01)
        # print("Grad_fn of phi_si: ", phi_si.grad_fn)

        tmp0 = torch.matmul(phi_si, self.w_fsb_p)
        tmp0 = self.batch_norm_tmp0(tmp0)  # Apply Batch Normalization
        # print("Grad_fn of tmp0: ", tmp0.grad_fn)

        tmp1 = torch.mul(message_1, tmp0)
        # print("Grad_fn of tmp1: ", tmp1.grad_fn)

        tmp2 = torch.matmul(phi_si, self.w_fsa_p)
        tmp2 = self.batch_norm_tmp2(tmp2)  # Apply Batch Normalization
        # print("Grad_fn of tmp2: ", tmp2.grad_fn)

        tmp3 = torch.mul(message_2, tmp2)
        # print("Grad_fn of tmp3: ", tmp3.grad_fn)

        F_Si = torch.tanh(tmp1 + tmp3)
        # print("Grad_fn of F_Si: ", F_Si.grad_fn)

        temppp = torch.matmul(phi_si, self.w_tempp)
        # print("Grad_fn of temppp: ", temppp.grad_fn)

        m_ip = torch.cat((F_Si, torch.mul(F_Ci, (1 - temppp))), dim=1)
        # print("Grad_fn of m_i: ", m_ip.grad_fn)

        return m_ip

    def HGD_FN_comment(self, com_n, com_n_n):
        phi_ci = F.leaky_relu(torch.cat(
            (torch.matmul(com_n, self.w_gcb_c), torch.matmul(com_n_n, self.w_gca_c)), dim=1), negative_slope=0.01)
        # print("Grad_fn of phi_ci: ", phi_ci.grad_fn)

        temp0 = torch.matmul(phi_ci, self.w_fcb_c)
        temp0 = self.batch_norm_temp0c(temp0)  # Apply Batch Normalization
        # print("Grad_fn of temp0: ", temp0.grad_fn)

        temp1 = torch.mul(com_n, temp0)
        # print("Grad_fn of temp1: ", temp1.grad_fn)

        temp2 = torch.matmul((1 - phi_ci), self.w_fca_c)
        temp2 = self.batch_norm_temp2c(temp2)  # Apply Batch Normalization
        # print("Grad_fn of temp2: ", temp2.grad_fn)

        temp3 = torch.mul(com_n_n, temp2)
        # print("Grad_fn of temp3: ", temp3.grad_fn)

        F_Ci = torch.tanh(temp1 + temp3)
        # print("Grad_fn of F_Ci: ", F_Ci.grad_fn)

        phi_si = F.leaky_relu(torch.cat(
            (torch.matmul(com_n, self.w_gsb_c), torch.matmul(com_n_n, self.w_gsa_c)), dim=1), negative_slope=0.01)
        # print("Grad_fn of phi_si: ", phi_si.grad_fn)

        tmp0 = torch.matmul(phi_si, self.w_fsb_c)
        tmp0 = self.batch_norm_tmp0c(tmp0)  # Apply Batch Normalization
        # print("Grad_fn of tmp0: ", tmp0.grad_fn)

        tmp1 = torch.mul(com_n, tmp0)
        # print("Grad_fn of tmp1: ", tmp1.grad_fn)

        tmp2 = torch.matmul(phi_si, self.w_fsa_c)
        tmp2 = self.batch_norm_tmp2c(tmp2)  # Apply Batch Normalization
        # print("Grad_fn of tmp2: ", tmp2.grad_fn)

        tmp3 = torch.mul(com_n_n, tmp2)
        # print("Grad_fn of tmp3: ", tmp3.grad_fn)

        F_Si = torch.tanh(tmp1 + tmp3)
        # print("Grad_fn of F_Si: ", F_Si.grad_fn)

        tempcp = torch.matmul(phi_si, self.w_tempc)
        # print("Grad_fn of temppp: ", temppp.grad_fn)

        m_ic = torch.cat((F_Si, torch.mul(F_Ci, (1 - tempcp))), dim=1)
        # print("Grad_fn of m_i: ", m_ic.grad_fn)

        return m_ic

    def SGRU(self, Xt, Ht_1):
        concatenated = torch.cat((Xt, Ht_1), dim=1)
        Rt = torch.sigmoid(self.batch_norm_r(torch.matmul(concatenated, self.w_r)))
        Zt = torch.sigmoid(self.batch_norm_z(torch.matmul(concatenated, self.w_z)))  # batch_size, 1536
        Ht_hidden = torch.tanh(torch.matmul(torch.cat((Xt, Rt * Ht_1), dim=1), self.w_h))
        Ht = Zt * Ht_1 + (1 - Zt) * Ht_hidden
        return Ht

    def blbl(self, post, llm, HGD):
        p1 = torch.cat((torch.matmul(post, self.blbl_1), torch.matmul(llm, self.blbl_2)), dim=1)  # bs, 768
        p2_1 = F.leaky_relu(torch.mul(p1, llm), negative_slope=0.01)
        p2_1 = self.batch_norm_blbl_1(p2_1)
        p2_2 = torch.matmul(torch.cat((p1, llm), dim=1), self.blbl_3)  # 1536-->768
        p3 = torch.cat((torch.mul(p2_1, p2_2), llm), dim=1)  # bs, 1536
        p3 = self.batch_norm_blbl_2(p3)

        H1 = torch.cat((HGD, llm), dim=1)
        H2 = torch.matmul(H1, self.blbl_4)  # 1536-->768
        H3_1 = torch.matmul(F.leaky_relu(H2, negative_slope=0.01), self.blbl_5)  # 768-->384
        H3_2 = torch.matmul(torch.softmax(H2, dim=1), self.blbl_6)  # 768-->384
        H3 = torch.cat((H3_1, H3_2), dim=1)  # bs, 768
        H4_1 = torch.mul(post, H3)
        H4_1 = self.batch_norm_blbl_3(H4_1)
        H4_1_1 = torch.mul(H1, torch.matmul(H4_1, self.blbl_7))  # bs, 1536
        H4_2 = torch.matmul(torch.cat((post, H3), dim=1), self.blbl_8)  # bs, 1536-->768
        H4_2 = self.batch_norm_blbl_4(H4_2)
        H5 = torch.cat((H4_1_1, H4_2, p3), dim=1)  # bs, 1536+768+1536(768*5)

        return H5

    def compute_conflict(self, B1, B2):
        """
        计算两个专家之间的冲突度 C
        B1, B2: [batch_size, N]
        返回 C: [batch_size, 1]
        """
        sum_b1 = B1.sum(dim=1, keepdim=True)  # [batch_size, 1]
        sum_b2 = B2.sum(dim=1, keepdim=True)  # [batch_size, 1]
        sum_b1b2 = (B1 * B2).sum(dim=1, keepdim=True)  # [batch_size, 1]
        C = sum_b1 * sum_b2 - sum_b1b2  # [batch_size, 1]

        return C

    def combine_two_opinions(self, B1, u1, B2, u2):
        """
        组合两个专家的意见
        B1, B2: [batch_size, N] (Dirichlet参数)
        u1, u2: [batch_size, 1] (不确定度)
        返回：
        combined_B: [batch_size, N]
        combined_u: [batch_size, 1]
        """

        C = self.compute_conflict(B1, B2)  # [batch_size, 1]

        denominator = 1 - C + 1e-8  # [batch_size, 1]


        combined_B = (B1 * B2 + B1 * u2 + B2 * u1) / denominator  # [batch_size, N]
        combined_u = (u1 * u2) / denominator  # [batch_size, 1]

        return combined_B, combined_u

    def combine_multiple_opinions(self, Bs, Us):

        if not Bs or not Us or len(Bs) != len(Us):
            raise ValueError("Bs 和 Us 必须是非空列表，且长度相同")

        k = len(Bs)
        combined_B = Bs[0]
        combined_u = Us[0]

        for i in range(1, k):
            combined_B, combined_u = self.combine_two_opinions(combined_B, combined_u, Bs[i], Us[i])

        return combined_B, combined_u

    def bert_pool(self, input_ids):
        """Run BERT with a correct attention_mask (masking PAD=0) and return pooled output."""
        attn_mask = (input_ids != 0).long()
        _, pooled = self.bert(input_ids, None, attn_mask, output_all_encoded_layers=False)
        return pooled

    def forward(self, batch_Data, labels=None, sentiment_labels=None, log_fm_stats: bool = False):
        batch_Data = batch_Data.transpose(0, 1).to(device)
        input_post = batch_Data[0].to(torch.int64)
        input_com1 = batch_Data[1].to(torch.int64)
        input_com11 = batch_Data[2].to(torch.int64)
        input_com12 = batch_Data[3].to(torch.int64)
        input_com13 = batch_Data[4].to(torch.int64)
        input_com2 = batch_Data[5].to(torch.int64)
        input_com21 = batch_Data[6].to(torch.int64)
        input_com22 = batch_Data[7].to(torch.int64)
        input_com23 = batch_Data[8].to(torch.int64)
        input_com3 = batch_Data[9].to(torch.int64)
        input_com31 = batch_Data[10].to(torch.int64)
        input_com32 = batch_Data[11].to(torch.int64)
        input_com33 = batch_Data[12].to(torch.int64)
        input_com4 = batch_Data[13].to(torch.int64)
        input_com41 = batch_Data[14].to(torch.int64)
        input_com42 = batch_Data[15].to(torch.int64)
        input_llm = batch_Data[16].to(torch.int64)
        pooled_post = self.bert_pool(input_post)
        pooled_com1 = self.bert_pool(input_com1)
        pooled_com11 = self.bert_pool(input_com11)
        pooled_com12 = self.bert_pool(input_com12)
        pooled_com13 = self.bert_pool(input_com13)

        pooled_com2 = self.bert_pool(input_com2)
        pooled_com21 = self.bert_pool(input_com21)
        pooled_com22 = self.bert_pool(input_com22)
        pooled_com23 = self.bert_pool(input_com23)

        pooled_com3 = self.bert_pool(input_com3)
        pooled_com31 = self.bert_pool(input_com31)
        pooled_com32 = self.bert_pool(input_com32)
        pooled_com33 = self.bert_pool(input_com33)

        pooled_com4 = self.bert_pool(input_com4)
        pooled_com41 = self.bert_pool(input_com41)
        pooled_com42 = self.bert_pool(input_com42)

        pooled_llm = self.bert_pool(input_llm)

        contrast_p_1 = self.HGD_FN_Post(pooled_post, pooled_com1)
        contrast_p_2 = self.HGD_FN_Post(pooled_post, pooled_com2)
        contrast_p_3 = self.HGD_FN_Post(pooled_post, pooled_com3)
        contrast_p_4 = self.HGD_FN_Post(pooled_post, pooled_com4)

        contrast_c1_1 = self.HGD_FN_comment(pooled_com1, pooled_com11)
        contrast_c1_2 = self.HGD_FN_comment(pooled_post, pooled_com12)
        contrast_c1_3 = self.HGD_FN_comment(pooled_post, pooled_com13)
        Ht = self.SGRU(contrast_c1_2, contrast_c1_1)
        HT_1 = self.SGRU(contrast_c1_3, Ht)
        contrast_c2_1 = self.HGD_FN_comment(pooled_com2, pooled_com21)
        contrast_c2_2 = self.HGD_FN_comment(pooled_com2, pooled_com22)
        contrast_c2_3 = self.HGD_FN_comment(pooled_com2, pooled_com23)
        Ht = self.SGRU(contrast_c2_2, contrast_c2_1)
        HT_2 = self.SGRU(contrast_c2_3, Ht)
        contrast_c3_1 = self.HGD_FN_comment(pooled_com3, pooled_com31)
        contrast_c3_2 = self.HGD_FN_comment(pooled_com3, pooled_com32)
        contrast_c3_3 = self.HGD_FN_comment(pooled_com3, pooled_com33)
        Ht = self.SGRU(contrast_c3_2, contrast_c3_1)
        HT_3 = self.SGRU(contrast_c3_3, Ht)
        contrast_c4_1 = self.HGD_FN_comment(pooled_com4, pooled_com41)
        contrast_c4_2 = self.HGD_FN_comment(pooled_com4, pooled_com42)
        combine_4 = self.SGRU(contrast_c4_2, contrast_c4_1)

        combine_1 = torch.matmul(torch.cat((contrast_p_1, HT_1), dim=1), self.w_com1)
        combine_2 = torch.matmul(torch.cat((contrast_p_2, HT_2), dim=1), self.w_com2)
        combine_3 = torch.matmul(torch.cat((contrast_p_3, HT_3), dim=1), self.w_com3)

        HGD = torch.matmul(torch.cat((combine_1, combine_2, combine_3, combine_4), dim=1), self.blbl_7680)
        Ending = self.blbl(pooled_post, pooled_llm, HGD)
        conments_end = torch.cat((HT_1, HT_2, HT_3, combine_4), dim=1)
        Ending_conmments = self.comment_linear(conments_end)
        # result2 = self.dropout(Ending)
        # classification_logits = self.news_classifier(result2)
        # dirichlet_params = F.relu(self.dirichlet_predictor(result2)) + 1
        # contrastive_params = self.contrastive_predictor(Ending)

        z_news_input = self.dropout(Ending)
        z_comment_input = self.dropout(Ending_conmments)
        aux_losses = {}
        if self.training and labels is not None:

            out_aux = self.dual_purifier.compute_aux_losses(
                z_news_input, z_comment_input, labels, return_stats=log_fm_stats
            )
            if log_fm_stats:
                l_fm, l_proto, l_senti, fm_stats, fm_loss_per_sample = out_aux
            else:
                l_fm, l_proto, l_senti, fm_loss_per_sample = out_aux
                fm_stats = None

            aux_losses['loss_fm_news'] = l_fm
            aux_losses['loss_fm_news_per_sample'] = fm_loss_per_sample  # [B], keep grad for weighting
            aux_losses['loss_proto'] = l_proto
            aux_losses['loss_fm_senti'] = l_senti

            if fm_stats is not None:
                for k, v in fm_stats.items():
                    aux_losses[f"fm/{k}"] = v

            dirichlet_params, pred_stats = self.dual_purifier.predict(
                z_news_input, z_comment_input, sentiment_target=sentiment_labels, return_stats=True
            )
            if pred_stats is not None:
                aux_losses.update(pred_stats)


        else:

            dirichlet_params = self.dual_purifier.predict(
                z_news_input, z_comment_input, sentiment_target=None
            )

        # contrastive_params = self.contrastive_predictor(Ending)

        # Alpha_s1 = F.relu(self.dirichlet_predictor_s1(HT_1)) + 1
        # Alpha_s2 = F.relu(self.dirichlet_predictor_s2(HT_2)) + 1
        # Alpha_s3 = F.relu(self.dirichlet_predictor_s3(HT_3)) + 1
        # Alpha_s4 = F.relu(self.dirichlet_predictor_s4(HT_4)) + 1
        # Alpha_s5 = F.relu(self.dirichlet_predictor_s5(HT_5)) + 1
        #
        # sum_alpha_s1 = Alpha_s1.sum(dim=1, keepdim=True)
        # sum_alpha_s2 = Alpha_s2.sum(dim=1, keepdim=True)
        # sum_alpha_s3 = Alpha_s3.sum(dim=1, keepdim=True)
        # sum_alpha_s4 = Alpha_s4.sum(dim=1, keepdim=True)
        # sum_alpha_s5 = Alpha_s5.sum(dim=1, keepdim=True)
        #
        # b_s1 = (Alpha_s1 - 1) / sum_alpha_s1
        # u_s1 = self.comments_labels / sum_alpha_s1
        #
        # b_s2 = (Alpha_s2 - 1) / sum_alpha_s2
        # u_s2 = self.comments_labels / sum_alpha_s2
        #
        # b_s3 = (Alpha_s3 - 1) / sum_alpha_s3
        # u_s3 = self.comments_labels / sum_alpha_s3
        #
        # b_s4 = (Alpha_s4 - 1) / sum_alpha_s4
        # u_s4 = self.comments_labels / sum_alpha_s4
        #
        # b_s5 = (Alpha_s5 - 1) / sum_alpha_s5
        # u_s5 = self.comments_labels / sum_alpha_s5
        #
        # Bs = [b_s1, b_s2, b_s3, b_s4, b_s5]
        # Us = [u_s1, u_s2, u_s3, u_s4, u_s5]
        #
        # combined_b, combined_u = self.combine_multiple_opinions(Bs, Us)
        # Alpha_by_combined_b = combined_b + 1

        return dirichlet_params, aux_losses


class Metrics:
    def __init__(self, num_classes):

        self.num_classes = num_classes
        self.reset()

    def reset(self):

        self.all_preds = []
        self.all_probs = []
        self.all_labels = []

    def update(self, preds, probs, labels):

        self.all_preds.append(preds.cpu().numpy())
        self.all_probs.append(probs.cpu().numpy())
        self.all_labels.append(labels.cpu().numpy())

    def compute(self):

        preds = np.concatenate(self.all_preds)
        probs = np.concatenate(self.all_probs)
        labels = np.concatenate(self.all_labels)

        metrics_dict = {}


        if self.num_classes == 2:
            auroc = roc_auc_score(labels, probs[:, 1])
        else:
            auroc = roc_auc_score(labels, probs, multi_class='ovo', average='macro')
        metrics_dict['AUROC'] = auroc


        if self.num_classes == 2:
            auprc = average_precision_score(labels, probs[:, 1])
        else:

            labels_binarized = label_binarize(labels, classes=np.arange(self.num_classes))
            auprc = average_precision_score(labels_binarized, probs, average='macro')
        metrics_dict['AUPRC'] = auprc


        uar = recall_score(labels, preds, average='macro')
        metrics_dict['UAR'] = uar


        ece = self.expected_calibration_error(probs, preds, labels, n_bins=15)
        metrics_dict['ECE'] = ece


        mce = self.maximum_calibration_error(probs, preds, labels, n_bins=15)
        metrics_dict['MCE'] = mce

        return metrics_dict

    def expected_calibration_error(self, probs, preds, labels, n_bins=15):

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs.max(axis=1) > bin_lower) & (probs.max(axis=1) <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = (preds[in_bin] == labels[in_bin]).mean()
                confidence_in_bin = probs[in_bin].max(axis=1).mean()
                ece += np.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def maximum_calibration_error(self, probs, preds, labels, n_bins=15):

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs.max(axis=1) > bin_lower) & (probs.max(axis=1) <= bin_upper)
            if in_bin.any():
                accuracy_in_bin = (preds[in_bin] == labels[in_bin]).mean()
                confidence_in_bin = probs[in_bin].max(axis=1).mean()
                mce = max(mce, np.abs(confidence_in_bin - accuracy_in_bin))
        return mce


class DataProcessForMultipleSentences(object):
    """Preprocess multiple text fields into a TensorDataset.

    This project uses multiple short English text fields (post/news, multiple comments, and an LLM review).
    We therefore support *different* max lengths per field to save compute while staying robust.

    Default (recommended):
      - post/news: 64
      - comments: 64
      - llm review: 128
    """

    DEFAULT_MAX_LEN_CFG = {"post": 64, "comments": 64, "llm": 128}

    def __init__(self, bert_tokenizer, max_len_cfg=None, max_workers=10):
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

        self.max_len_cfg = dict(self.DEFAULT_MAX_LEN_CFG)
        if max_len_cfg is not None:
            self.max_len_cfg.update(max_len_cfg)

    def get_input(self, dataset, max_len_cfg=None):
        # Support either a dict config or a single int (legacy behavior: one length for all fields)
        cfg = dict(self.max_len_cfg)
        if max_len_cfg is not None:
            if isinstance(max_len_cfg, int):
                cfg.update({"post": int(max_len_cfg), "comments": int(max_len_cfg), "llm": int(max_len_cfg)})
            else:
                cfg.update(max_len_cfg)

        max_len_post = int(cfg.get("post", 64))
        max_len_comments = int(cfg.get("comments", 64))
        max_len_llm = int(cfg.get("llm", 128))

        message_post = dataset.iloc[:, 1].tolist()
        com1 = dataset.iloc[:, 4].tolist()
        com11 = dataset.iloc[:, 5].tolist()
        com12 = dataset.iloc[:, 6].tolist()
        com13 = dataset.iloc[:, 7].tolist()
        com2 = dataset.iloc[:, 8].tolist()
        com21 = dataset.iloc[:, 9].tolist()
        com22 = dataset.iloc[:, 10].tolist()
        com23 = dataset.iloc[:, 11].tolist()
        com3 = dataset.iloc[:, 12].tolist()
        com31 = dataset.iloc[:, 13].tolist()
        com32 = dataset.iloc[:, 14].tolist()
        com33 = dataset.iloc[:, 15].tolist()
        com4 = dataset.iloc[:, 16].tolist()
        com41 = dataset.iloc[:, 17].tolist()
        com42 = dataset.iloc[:, 18].tolist()

        fake_labels = dataset.iloc[:, 2].tolist()
        llm_outputs = dataset.iloc[:, 3].tolist()
        comment_labels = dataset.iloc[:, 19].tolist()

        def debug_non_str_cells(
                df: pd.DataFrame,
                col_indices,
                id_col: int = 0,
                news_col: int = 1,
                max_print: int = 30,
        ):
            print("=== Debug non-string cells ===")
            for ci in col_indices:
                s = df.iloc[:, ci]
                bad_mask = ~s.apply(lambda x: isinstance(x, str))
                bad_idx = np.where(bad_mask.values)[0]

                if len(bad_idx) == 0:
                    print(f"[col {ci}] OK (all str)")
                    continue

                print(f"[col {ci}] Found {len(bad_idx)} non-str cells. Showing up to {max_print}:")
                for j, ridx in enumerate(bad_idx[:max_print]):
                    row_id = df.iloc[ridx, id_col]
                    news = df.iloc[ridx, news_col]
                    val = df.iloc[ridx, ci]
                    print(f"  - row={ridx}, id={row_id}, news[:50]={str(news)[:50]!r}, value={val!r} (type={type(val)})")
            print("=== End debug ===")

        # Validate text columns (avoid tokenizer crash on NaN/float/etc.)
        text_col_indices = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        debug_non_str_cells(
            dataset,
            col_indices=text_col_indices,
            id_col=0,
            news_col=1,
            max_print=50
        )

        # Tokenize
        message_post = list(self.pool.map(self.bert_tokenizer.tokenize, message_post))
        com1 = list(self.pool.map(self.bert_tokenizer.tokenize, com1))
        com11 = list(self.pool.map(self.bert_tokenizer.tokenize, com11))
        com12 = list(self.pool.map(self.bert_tokenizer.tokenize, com12))
        com13 = list(self.pool.map(self.bert_tokenizer.tokenize, com13))

        com2 = list(self.pool.map(self.bert_tokenizer.tokenize, com2))
        com21 = list(self.pool.map(self.bert_tokenizer.tokenize, com21))
        com22 = list(self.pool.map(self.bert_tokenizer.tokenize, com22))
        com23 = list(self.pool.map(self.bert_tokenizer.tokenize, com23))

        com3 = list(self.pool.map(self.bert_tokenizer.tokenize, com3))
        com31 = list(self.pool.map(self.bert_tokenizer.tokenize, com31))
        com32 = list(self.pool.map(self.bert_tokenizer.tokenize, com32))
        com33 = list(self.pool.map(self.bert_tokenizer.tokenize, com33))

        com4 = list(self.pool.map(self.bert_tokenizer.tokenize, com4))
        com41 = list(self.pool.map(self.bert_tokenizer.tokenize, com41))
        com42 = list(self.pool.map(self.bert_tokenizer.tokenize, com42))

        llm_outputs = list(self.pool.map(self.bert_tokenizer.tokenize, llm_outputs))

        # Pad with different lengths
        message_post = list(self.pool.map(self.trunate_and_pad, message_post, repeat(max_len_post, len(message_post))))

        com1 = list(self.pool.map(self.trunate_and_pad, com1, repeat(max_len_comments, len(com1))))
        com11 = list(self.pool.map(self.trunate_and_pad, com11, repeat(max_len_comments, len(com11))))
        com12 = list(self.pool.map(self.trunate_and_pad, com12, repeat(max_len_comments, len(com12))))
        com13 = list(self.pool.map(self.trunate_and_pad, com13, repeat(max_len_comments, len(com13))))

        com2 = list(self.pool.map(self.trunate_and_pad, com2, repeat(max_len_comments, len(com2))))
        com21 = list(self.pool.map(self.trunate_and_pad, com21, repeat(max_len_comments, len(com21))))
        com22 = list(self.pool.map(self.trunate_and_pad, com22, repeat(max_len_comments, len(com22))))
        com23 = list(self.pool.map(self.trunate_and_pad, com23, repeat(max_len_comments, len(com23))))

        com3 = list(self.pool.map(self.trunate_and_pad, com3, repeat(max_len_comments, len(com3))))
        com31 = list(self.pool.map(self.trunate_and_pad, com31, repeat(max_len_comments, len(com31))))
        com32 = list(self.pool.map(self.trunate_and_pad, com32, repeat(max_len_comments, len(com32))))
        com33 = list(self.pool.map(self.trunate_and_pad, com33, repeat(max_len_comments, len(com33))))

        com4 = list(self.pool.map(self.trunate_and_pad, com4, repeat(max_len_comments, len(com4))))
        com41 = list(self.pool.map(self.trunate_and_pad, com41, repeat(max_len_comments, len(com41))))
        com42 = list(self.pool.map(self.trunate_and_pad, com42, repeat(max_len_comments, len(com42))))

        llm_outputs = list(self.pool.map(self.trunate_and_pad, llm_outputs, repeat(max_len_llm, len(llm_outputs))))

        # Unpack to tensors
        seqs_post = [x[0] for x in message_post]
        seqs_com1 = [x[0] for x in com1]
        seqs_com11 = [x[0] for x in com11]
        seqs_com12 = [x[0] for x in com12]
        seqs_com13 = [x[0] for x in com13]

        seqs_com2 = [x[0] for x in com2]
        seqs_com21 = [x[0] for x in com21]
        seqs_com22 = [x[0] for x in com22]
        seqs_com23 = [x[0] for x in com23]

        seqs_com3 = [x[0] for x in com3]
        seqs_com31 = [x[0] for x in com31]
        seqs_com32 = [x[0] for x in com32]
        seqs_com33 = [x[0] for x in com33]

        seqs_com4 = [x[0] for x in com4]
        seqs_com41 = [x[0] for x in com41]
        seqs_com42 = [x[0] for x in com42]

        seqs_llm_outputs = [x[0] for x in llm_outputs]

        t_seqs_post = torch.tensor(seqs_post, dtype=torch.long)
        t_seqs_com1 = torch.tensor(seqs_com1, dtype=torch.long)
        t_seqs_com11 = torch.tensor(seqs_com11, dtype=torch.long)
        t_seqs_com12 = torch.tensor(seqs_com12, dtype=torch.long)
        t_seqs_com13 = torch.tensor(seqs_com13, dtype=torch.long)

        t_seqs_com2 = torch.tensor(seqs_com2, dtype=torch.long)
        t_seqs_com21 = torch.tensor(seqs_com21, dtype=torch.long)
        t_seqs_com22 = torch.tensor(seqs_com22, dtype=torch.long)
        t_seqs_com23 = torch.tensor(seqs_com23, dtype=torch.long)

        t_seqs_com3 = torch.tensor(seqs_com3, dtype=torch.long)
        t_seqs_com31 = torch.tensor(seqs_com31, dtype=torch.long)
        t_seqs_com32 = torch.tensor(seqs_com32, dtype=torch.long)
        t_seqs_com33 = torch.tensor(seqs_com33, dtype=torch.long)

        t_seqs_com4 = torch.tensor(seqs_com4, dtype=torch.long)
        t_seqs_com41 = torch.tensor(seqs_com41, dtype=torch.long)
        t_seqs_com42 = torch.tensor(seqs_com42, dtype=torch.long)

        t_seqs_llm_outputs = torch.tensor(seqs_llm_outputs, dtype=torch.long)
        t_labels = torch.tensor(fake_labels, dtype=torch.long)
        comment_labels = [ast.literal_eval(label) for label in comment_labels]
        t_com_labels = torch.tensor(comment_labels, dtype=torch.long)

        return TensorDataset(
            t_seqs_post,
            t_seqs_com1, t_seqs_com11, t_seqs_com12, t_seqs_com13,
            t_seqs_com2, t_seqs_com21, t_seqs_com22, t_seqs_com23,
            t_seqs_com3, t_seqs_com31, t_seqs_com32, t_seqs_com33,
            t_seqs_com4, t_seqs_com41, t_seqs_com42,
            t_seqs_llm_outputs,
            t_labels,
            t_com_labels
        )

    def trunate_and_pad(self, seq, max_seq_len):
        if len(seq) > (max_seq_len - 2):
            seq = seq[0: (max_seq_len - 2)]
        seq = ['[CLS]'] + seq + ['[SEP]']
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        padding = [0] * (max_seq_len - len(seq))
        seq_mask = [1] * len(seq) + padding
        seq_segment = [0] * len(seq) + padding
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment

def load_data(filepath, pretrained_model_name_or_path, max_len_cfg, batch_size):
    io = pd.io.excel.ExcelFile(filepath)
    raw_train_data = pd.read_excel(io, sheet_name='train')
    raw_val_data = pd.read_excel(io, sheet_name='val')
    raw_test_data = pd.read_excel(io, sheet_name='test')
    # raw_out_data = pd.read_excel(io, sheet_name='out')

    io.close()
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=True)
    processor = DataProcessForMultipleSentences(bert_tokenizer=bert_tokenizer, max_len_cfg=max_len_cfg)
    train_data = processor.get_input(raw_train_data)
    val_data = processor.get_input(raw_val_data)
    test_data = processor.get_input(raw_test_data)
    # out_data = processor.get_input(raw_out_data, max_seq_len)

    train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_iter = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    # out_iter = DataLoader(dataset=out_data, batch_size=batch_size, shuffle=True)

    total_train_batch = math.ceil(len(raw_train_data) / batch_size)
    total_val_batch = math.ceil(len(raw_val_data) / batch_size)
    total_test_batch = math.ceil(len(raw_test_data) / batch_size)
    # total_out_batch = math.ceil(len(raw_out_data) / batch_size)

    return train_iter, val_iter, test_iter, total_train_batch, total_val_batch, total_test_batch


def pack_batch_fields(batch_data, device=None):
    """Pack the first 17 text-field tensors in a batch into a single tensor.

    Each field tensor has shape [B, L_field]. Since we allow different max lengths per field
    (e.g., post=64, comments=64, llm=128), we pad all fields in the *current batch* to the
    maximum length among them (typically the llm length), then stack to [B, 17, L_max].

    Padding value is 0, which matches BERT's PAD token id in this project.
    """
    fields = list(batch_data[:17])  # 17 tensors, each [B, L_i]
    if len(fields) != 17:
        raise ValueError(f"Expected 17 text fields, got {len(fields)}")

    # Ensure tensors are on CPU for cheap padding, then move once.
    # (If they are already on GPU, this still works; padding happens on that device.)
    max_len = max(int(t.size(1)) for t in fields)
    padded = []
    for t in fields:
        if t.dim() != 2:
            raise ValueError(f"Each field tensor must be 2D [B, L], got shape={tuple(t.shape)}")
        pad_right = max_len - int(t.size(1))
        if pad_right > 0:
            t = F.pad(t, (0, pad_right), value=0)
        padded.append(t)

    batch_tensor = torch.stack(padded, dim=1)  # [B, 17, L_max]
    if device is not None:
        batch_tensor = batch_tensor.to(device, non_blocking=True)
    return batch_tensor


def evaluate_accuracy(data_iter, net, device, batch_count, metrics, figdump_cfg=None):
    all_probs, all_preds, all_labels = [], [], []
    runner = FigDumpRunner(figdump_cfg or {"enabled": False})
    metrics.reset()
    with torch.no_grad():
        for batch_data in tqdm(data_iter, desc='eval', total=batch_count):
            labels = batch_data[17].to(device, non_blocking=True)

            batch_tensor_test = pack_batch_fields(batch_data, device=device)

            out = net(batch_tensor_test)
            if isinstance(out, (tuple, list)):
                classification_logits = out[0]
            else:
                classification_logits = out


            labels = labels.to(classification_logits.device, non_blocking=True)

            runner.step(net, classification_logits, labels)

            probs = classification_logits / (classification_logits.sum(dim=1, keepdim=True) + 1e-12)
            preds = probs.argmax(dim=1)

            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    metrics.update(all_preds, all_probs, all_labels)

    report = classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), digits=4)
    accuracy = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())

    metrics_dict = metrics.compute()

    out_paths = runner.finalize()
    if out_paths:
        print("[FigDump] saved:", out_paths)
    else:
        print("[FigDump] nothing saved. Check figdump_cfg.enabled / do_fig23 / do_fig45 / max_batches.")

    return report, accuracy, metrics_dict

class MultiViewLoss(nn.Module):
    def __init__(self, lambda_fused=0.5, lambda_m=0.2, lambda_t=0.1, hard_ratio=1.0, temperature=0.07, eta1=1.0, eta2=0.01,
                 fm_w_min=0.95, fm_w_max=1.05, fm_use_hard_upweight=False):
        super(MultiViewLoss, self).__init__()
        self.lambda_fused = lambda_fused  # 权重用于 Fused Loss
        self.lambda_m = lambda_m  # 权重用于 M Loss
        self.lambda_t = lambda_t  # 用于 KL 散度正则化的权重
        self.temperature = temperature
        self.hard_ratio = hard_ratio
        self.eta1 = eta1
        self.eta2 = eta2

        # --- stable FM reweighting (conservative) ---
        self.fm_w_min = fm_w_min
        self.fm_w_max = fm_w_max
        self.fm_use_hard_upweight = fm_use_hard_upweight
    def _calculate_hardness(self, dirichlet_params):
        """修正后的Dirichlet熵计算"""
        sum_params = dirichlet_params.sum(dim=1, keepdim=True).detach()
        term1 = torch.lgamma(sum_params).squeeze()
        term2 = torch.lgamma(dirichlet_params).sum(dim=1)
        term3 = torch.sum((dirichlet_params - 1) * (torch.digamma(dirichlet_params) - torch.digamma(sum_params)), dim=1)
        entropy = term1 - term2 + term3
        end = torch.sigmoid(entropy)
        # print(end)
        return end

    def _reweight_similarity(self, sim_matrix, pos_mask, neg_mask, hardness, hard_radio):

        # 正样本保持原权重
        pos_weighted = sim_matrix * pos_mask

        # 负样本加权：硬度越大（高熵样本）权重越高
        hardness = hardness + 1
        hardness_2dim = hardness.unsqueeze(1)
        hardness_matrix = hardness_2dim * hardness_2dim.T
        # 对角线使用原 hardness 值（自身对比）
        for i in range(hardness_matrix.shape[0]):
            hardness_matrix[i, i] = hardness[i]

        # 对 hardness_matrix 的所有元素应用阈值处理
        # 计算 hard_radio 分位数的阈值
        threshold = torch.quantile(hardness_matrix, hard_radio)
        # 低于阈值的置 1，高于或等于的保持原值
        hardness_matrix = torch.where(hardness_matrix < threshold,
                                      torch.tensor(1.0, device=hardness_matrix.device),
                                      hardness_matrix)

        neg_weighted = sim_matrix * neg_mask * hardness_matrix

        # 对角线恢复原始相似度（自身对比）
        diag_weight = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        diag_sim = sim_matrix * diag_weight

        return pos_weighted + neg_weighted + diag_sim

    # def contrastive_loss(self, embeddings, dirichlet_params_fused, dirichlet_params_M, labels, epoch):
    #     """
    #     NLP监督对比损失（含Dirichlet硬样本感知）
    #     :param embeddings: 文本嵌入 [batch_size, emb_dim]
    #     :param dirichlet_params: Dirichlet参数 [batch_size, num_classes]
    #     :param labels: 样本标签 [batch_size] (0/1)
    #     :return: 加权对比损失
    #     """
    #     batch_size = embeddings.size(0)
    #
    #     embeddings = F.normalize(embeddings, p=2, dim=1).detach()
    #
    #     # sim_matrix = torch.einsum('ik,jk->ij', embeddings, embeddings)  # 计算每对嵌入的点积
    #     # sim_matrix = sim_matrix / (torch.einsum('ik,ik->i', embeddings, embeddings).unsqueeze(1) *
    #     #                            torch.einsum('jk,jk->j', embeddings, embeddings).unsqueeze(0))  # 归一化
    #     sim_matrix = torch.mm(embeddings, embeddings.t())
    #
    #     sim_matrix = torch.exp(sim_matrix / self.temperature)
    #     # print('sim_matrix', sim_matrix)

    #     pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    #     pos_mask.fill_diagonal_(1)  # 排除自身对比
    #     neg_mask = 1 - pos_mask
    #     # print('pos_mask', pos_mask, pos_mask.shape)
    #     # print('neg_mask', neg_mask, neg_mask.shape)

    #     entropy = self._calculate_hardness(dirichlet_params_fused)
    #     # print(entropy)
    #     # entropy_M = self._calculate_hardness(dirichlet_params_M)

    #     # num_hard_samples = int(batch_size * self.hard_ratio)
    #     # hard_indices = entropy.topk(num_hard_samples).indices  # Top hard_ratio samples
    #     # hardness_weights = torch.zeros_like(entropy)
    #     # hardness_weights[hard_indices] = entropy[hard_indices]
    #     # print('hardness_weights', hardness_weights)
    #

    #     sim_logit = 0
    #     # if epoch <= 25:
    #     #     sim_logit = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    #     if True:
    #         sim_matrix_DIRICHLET = self._reweight_similarity(
    #             sim_matrix=sim_matrix,
    #             pos_mask=pos_mask,
    #             neg_mask=neg_mask,
    #             hardness=entropy,
    #             hard_radio=self.hard_ratio
    #         )
    #         sim_logit = sim_matrix_DIRICHLET - torch.logsumexp(sim_matrix_DIRICHLET, dim=1, keepdim=True)
    #     pos_pairs = (sim_logit * pos_mask).sum(dim=1) + 1e-8
    #     neg_pairs = (sim_logit * neg_mask).sum(dim=1) + 1e-8
    #     # print('--------------------------------------------------------')
    #
    #     loss = -torch.log(pos_pairs / (pos_pairs + neg_pairs)).mean()
    #
    #     return loss

    def compute_L_E(self, y, alpha, sample_weights=None):
        # y: one-hot [B,K], alpha: Dirichlet params [B,K]
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)  # [B,1]
        digamma_sum_alpha = torch.digamma(sum_alpha)  # [B,1]
        digamma_alpha = torch.digamma(alpha)  # [B,K]
        term = y * (digamma_sum_alpha - digamma_alpha)  # [B,K]
        L_E_per_sample = torch.sum(term, dim=1)  # [B]

        # label-based weights (e.g., upweight class 1)
        true_labels = torch.argmax(y, dim=1)  # [B]
        label_w = torch.ones_like(L_E_per_sample, dtype=torch.float)
        label_w[true_labels == 1] = 2.425  # 2.425

        if sample_weights is None:
            denom = torch.sum(label_w) + 1e-8
            return torch.sum(L_E_per_sample * label_w) / denom

        w = sample_weights.detach().float().view(-1)
        w = torch.clamp(w, min=0.0)

        total_w = w * label_w
        denom = torch.sum(total_w) + 1e-8
        return torch.sum(L_E_per_sample * total_w) / denom

    def forward(self, epoch, dirichlet_params_fused, news_labels,
                comment_counts, aux_losses=None, eta3=0, sample_weights=None):
        # -----------------------
        # Fused Loss (有标注数据)
        # -----------------------
        # 1. 交叉熵损失
        # fused_log_likelihood = F.cross_entropy(classification_logits, news_labels)

        # 2. 获取 one-hot 编码的标签
        y_onehot = F.one_hot(news_labels, num_classes=dirichlet_params_fused.size(1)).float()

        # 3. 构造 \tilde{\alpha}^m
        # tilde_alpha_fused = y_onehot + (1.0 - y_onehot) * dirichlet_params_fused
        # alpha0_fused = torch.sum(tilde_alpha_fused, dim=1, keepdim=True)
        #
        # # 4. 定义均匀先验 Dirichlet([1, ..., 1])
        # prior_fused = torch.ones_like(tilde_alpha_fused)
        # prior0_fused = torch.sum(prior_fused, dim=1, keepdim=True)
        #
        # # 5. 计算 KL 散度
        # kl_fused = (
        #         torch.lgamma(alpha0_fused)
        #         - torch.sum(torch.lgamma(tilde_alpha_fused), dim=1, keepdim=True)
        #         - torch.lgamma(prior0_fused)
        #         + torch.sum(torch.lgamma(prior_fused), dim=1, keepdim=True)
        #         + torch.sum(
        #     (tilde_alpha_fused - prior_fused) * (torch.digamma(tilde_alpha_fused) - torch.digamma(alpha0_fused)), dim=1,
        #     keepdim=True)
        # ).mean()

        # 6. 组合 Fused Loss
        # Note: hard/shift weights reweight the evidential classification term (L_EDL)
        L_E = self.compute_L_E(y_onehot, dirichlet_params_fused, sample_weights=sample_weights)

        # # print(L_E)
        # fused_loss = L_E + self.lambda_t * kl_fused
        # # -----------------------
        # # M Loss (基于评论标签)
        # # -----------------------
        # # 1. Dirichlet-Multinomial 负对数似然损失
        # S_com = torch.sum(dirichlet_params_comment, dim=1, keepdim=True)  # S = sum(alpha_k)
        # N_com = torch.sum(comment_counts, dim=1, keepdim=True)  # N = sum(y_k)
        # # 使用 torch.lgamma 来计算 Gamma 函数的对数
        # log_likelihood_dm = (
        #         torch.lgamma(S_com) - torch.lgamma(S_com + N_com) +
        #         torch.sum(torch.lgamma(dirichlet_params_comment + comment_counts), dim=1, keepdim=True) -
        #         torch.sum(torch.lgamma(dirichlet_params_comment), dim=1, keepdim=True)
        # )
        # m_log_likelihood = -log_likelihood_dm.mean()
        #
        # # 2. 构造 \tilde{\alpha}^m
        # tilde_alpha_com = dirichlet_params_comment
        # alpha0_com = torch.sum(tilde_alpha_com, dim=1, keepdim=True)
        #
        # # 3. 定义均匀先验 Dirichlet([1, ..., 1])
        # prior_com = torch.ones_like(tilde_alpha_com)
        # prior0_com = torch.sum(prior_com, dim=1, keepdim=True)
        #
        # # 4. 计算 KL 散度 使用 torch.distributions
        # posterior = Dirichlet(tilde_alpha_com)
        # prior = Dirichlet(prior_com)
        # kl_com = kl_divergence(posterior, prior).mean()
        # # print('KL:', kl_com)
        #
        # # 5. 组合 M Loss
        # m_loss = self.lambda_m * m_log_likelihood + self.lambda_t * kl_com

        # ContrastiveLoss = self.contrastive_loss(contrastive_params, dirichlet_params_fused, dirichlet_params_comment,
        #                                         news_labels, epoch)

        # overall_loss = (
        #         fused_loss
        #         + m_loss * 0.05
        #         # + ContrastiveLoss * self.lambda_fused
        # )
        print("L_E", L_E.item() * self.lambda_fused)
        overall_loss = L_E * self.lambda_fused

        if aux_losses:
            # ---- OT-FM regression loss (unweighted mean; keep regression stable) ----
            if 'loss_fm_news' in aux_losses:
                loss_fm_news = aux_losses['loss_fm_news']
            elif 'loss_fm_news_per_sample' in aux_losses:
                loss_fm_news = aux_losses['loss_fm_news_per_sample'].mean()
            else:
                loss_fm_news = None

            if loss_fm_news is not None:
                term_fm = self.eta1 * loss_fm_news
                overall_loss += term_fm
                try:
                    raw_val = float(loss_fm_news.detach().item())
                    w_val = float(term_fm.detach().item())
                except Exception:
                    raw_val = float(loss_fm_news.detach().mean().item())
                    w_val = float(term_fm.detach().mean().item())
                print('loss_fm_news raw={:.4f}, weighted={:.4f}, eta1={:.4f}'.format(raw_val, w_val, float(self.eta1)))
            if 'loss_proto' in aux_losses:
                overall_loss += self.eta2 * aux_losses['loss_proto']
                print('loss_proto', self.eta2 * aux_losses['loss_proto'].item())
            if 'loss_fm_senti' in aux_losses:
                overall_loss += eta3 * aux_losses['loss_fm_senti']
                print('loss_fm_senti', eta3 * aux_losses['loss_fm_senti'].item())

        # print('m_loss', m_loss.item())   # kept for reference
        print('overall_loss', overall_loss.item())

        return overall_loss


if __name__ == '__main__':
    def seed_everything(seed=42):
        print(f"Global Seed set to {seed}")

        # 1. Python
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        # 2. Numpy
        np.random.seed(seed)

        # 3. PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    seed_everything(3407)
    batch_size = 256
    MAX_LEN_CFG = {'post': 128, 'comments': 64, 'llm': 128}
    Max_epoch = 150
    num_classes_news = 2
    num_classes_sentiment = 7
    lambda_soft_max = 0.2

    train_iter, val_iter, test_iter, train_batch_count, val_batch_count, test_batch_count = load_data(
        'Pheme.xlsx',
        'bert-large-uncased', MAX_LEN_CFG, batch_size)

    metrics_val = Metrics(num_classes=num_classes_news)
    metrics_test = Metrics(num_classes=num_classes_news)
    model = ClassifyModel('bert-large-uncased', news_labels=num_classes_news, comments_labels=num_classes_sentiment,
                          is_lock=True)


    loss_fn = MultiViewLoss(lambda_fused=0.6, lambda_m=0.8260254001054855, lambda_t=0.1, hard_ratio=0.6,
                            temperature=0.07, eta1=0.2, eta2=0.01)

    proto_module = model.dual_purifier.news_purifier.prototypes
    proto_params = list(proto_module.parameters())


    proto_ids = set(map(id, proto_params))


    base_params = filter(lambda p: id(p) not in proto_ids, model.parameters())


    optimizer = torch.optim.Adam([

        {'params': base_params, 'lr': 5e-5},

        {'params': proto_params, 'lr': 5e-3}
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    RESULT_val = []
    RESULT_test = []
    RESULT_dict = []
    LOSS = []
    background = []
    DiyWeight = []
    DiyWeightGrad = []
    GradNorm = []
    VAL_ACC = 0
    Test_ACC = 0
    eta3 = 0
    Best_model_filename = None
    # background = torch.tensor(background).to(device)
    # unlabeled_iterator = iter(out_iter)
    background_count = 0
    # ===== FM metrics logging (for plots) =====
    FM_LOG_ENABLED = True
    FM_LOG_EVERY = 1  # log every N steps
    fm_logger = SimpleCSVLogger(os.path.join("figdump", "train_fm_metrics.csv")) if FM_LOG_ENABLED else None
    if fm_logger is not None:
        atexit.register(fm_logger.close)
    global_step = 0
    for epoch in range(1, Max_epoch + 1):
        start = time.time()
        model.train()

        # --- set epoch for FM/OT-FM switching ---
        try:
            model.dual_purifier.news_purifier.set_epoch(epoch)
        except Exception:
            pass

        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0

        for step, batch_data in tqdm(enumerate(train_iter), desc='train epoch:{}/{}'.format(epoch, Max_epoch)
                , total=train_batch_count):
            batch_tensor = pack_batch_fields(batch_data, device=device)

            batch_labels = batch_data[17].clone().detach().to(device)
            comments_labels = batch_data[18].clone().detach().to(device)
            torch.autograd.set_detect_anomaly(True)

            optimizer.zero_grad()
            log_fm_stats = FM_LOG_ENABLED and (global_step % FM_LOG_EVERY == 0)
            dirichlet_params_fused, aux_losses = model(batch_tensor, batch_labels, comments_labels, log_fm_stats=log_fm_stats)


            # ===== Hard/Shift mining weights (only affect L_EDL) =====
            sample_weights = None
            if aux_losses is not None and ('_z_raw_norm' in aux_losses) and ('_instant_energy' in aux_losses):
                try:
                    with torch.no_grad():

                        z_raw_norm = aux_losses['_z_raw_norm']           # [B, D]
                        inst_energy = aux_losses['_instant_energy']      # [B] or [B,1]
                        alpha_det = dirichlet_params_fused.detach()      # [B, K]

                        batch_labels_1d = batch_labels.view(-1)
                        inst_energy = inst_energy.view(-1).float()


                        p = alpha_det / (alpha_det.sum(dim=1, keepdim=True) + 1e-8)
                        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1).view(-1).float()  # [B]

                        dist2 = model.dual_purifier.news_purifier.prototype_class_mindist2(z_raw_norm)  # [B,K]
                        d_sorted, _ = torch.sort(dist2, dim=1)
                        d1 = d_sorted[:, 0].view(-1).float()
                        d2 = d_sorted[:, 1].view(-1).float()
                        margin = (d2 - d1).clamp(min=0.0).view(-1).float()


                        def _zscore(x):
                            x = x.float().view(-1)
                            mu = x.mean()
                            sd = x.std(unbiased=False)
                            return (x - mu) / (sd + 1e-6)


                        kappa = 0.5
                        gap = _zscore(d1) + kappa * _zscore(inst_energy)


                        inv_margin = -torch.log(margin + 1e-6)
                        hard = _zscore(entropy) + _zscore(inv_margin)

                        mining_start_epoch = 40
                        if epoch < mining_start_epoch:
                            sample_weights = None
                        else:
                            alpha_shift = 0.03
                            alpha_hard  = 0.03
                            w_clip_min  = 0.94
                            w_clip_max  = 1.00

                            w = torch.ones_like(entropy)  # [B]
                            w = w * (1.0 - alpha_shift * torch.sigmoid(gap))
                            w = w * (1.0 - alpha_hard  * torch.sigmoid(hard))

                            w = w.clamp(w_clip_min, w_clip_max)
                            w = w / (w.mean().detach() + 1e-8)

                            sample_weights = w

                            # ---- debug (first batch only) ----
                            if step == 0:
                                w_mean = float(sample_weights.mean().item())
                                w_min = float(sample_weights.min().item())
                                w_max = float(sample_weights.max().item())
                                print(f"mining(B-soft): w_mean={w_mean:.4f} (min={w_min:.4f}, max={w_max:.4f})")

                except Exception as e:
                    # 安全回退：不加权
                    if not hasattr(model, "_mining_err_printed"):
                        print("[Mining-B] ERROR -> disable weighting:", repr(e))
                        model._mining_err_printed = True
                    sample_weights = None

            loss = loss_fn(
                epoch,
                dirichlet_params_fused,
                batch_labels,
                comments_labels,
                aux_losses=aux_losses,
                eta3=eta3,
                sample_weights=sample_weights
            )
            # ===== write FM stats for plotting =====
            if log_fm_stats and fm_logger is not None:
                def _to_float(v):
                    if torch.is_tensor(v):
                        if v.numel() == 1:
                            return float(v.detach().cpu())
                        return float(v.detach().mean().cpu())
                    try:
                        return float(v)
                    except Exception:
                        return None

                row = {
                    "epoch": epoch,
                    "step_in_epoch": step,
                    "global_step": global_step,
                    "loss_total": _to_float(loss),
                    "loss_fm_news": _to_float(aux_losses.get("loss_fm_news", 0.0)) if isinstance(aux_losses, dict) else None,
                    "loss_proto": _to_float(aux_losses.get("loss_proto", 0.0)) if isinstance(aux_losses, dict) else None,
                    "loss_softrepa": _to_float(aux_losses.get("loss_softrepa", 0.0)) if isinstance(aux_losses, dict) else None,
                    "fm_mse": _to_float(aux_losses.get("fm/fm_mse", None)) if isinstance(aux_losses, dict) else None,
                    "u_norm": _to_float(aux_losses.get("fm/u_norm", None)) if isinstance(aux_losses, dict) else None,
                    "v_norm": _to_float(aux_losses.get("fm/v_norm", None)) if isinstance(aux_losses, dict) else None,
                    "resid_norm": _to_float(aux_losses.get("fm/resid_norm", None)) if isinstance(aux_losses, dict) else None,
                    "t_mean": _to_float(aux_losses.get("fm/t_mean", None)) if isinstance(aux_losses, dict) else None,
                    "swd2_raw_target": _to_float(aux_losses.get("fm/swd2_raw_target", None)) if isinstance(aux_losses, dict) else None,
                }
                fm_logger.write(row)
            loss.backward()

            if True:
                print(f"\n[Epoch {epoch} | Batch {step}] Gradient Check:")

                proto_grad_sum = 0.0
                has_proto_grad = False

                for p in model.dual_purifier.news_purifier.prototypes.parameters():
                    if p.grad is not None:
                        proto_grad_sum += p.grad.norm().item()
                        has_proto_grad = True

                if has_proto_grad:
                    print(f"  -> Prototypes Grad Norm:    {proto_grad_sum:.6f}")
                    if proto_grad_sum == 0:
                        print("  !!! WARNING: 原型梯度为 0！可能是 Loss 没传回来！ !!!")
                else:
                    print("  !!! ERROR: Prototypes 完全没有梯度 (None)！ !!!")

                vel_net = model.dual_purifier.news_purifier.news_velocity_net
                if vel_net[0].weight.grad is not None:
                    vel_grad = vel_net[0].weight.grad.norm().item()
                    print(f"  -> News Velocity Net Grad:  {vel_grad:.6f}")
                else:
                    print("  !!! ERROR: News Velocity Net 没有梯度！FM Loss 可能断了！ !!!")

            train_loss_sum += loss.item()
            logits = dirichlet_params_fused.softmax(dim=1)
            train_acc_sum += (logits.argmax(dim=1) == batch_labels).sum().item()
            n += batch_labels.shape[0]
            optimizer.step()
        model.eval()
        result_val, val_acc, metrics_dict_val = 0,0,0
        result_test, test_acc, metrics_dict_test = 0, 0, 0

        if epoch > 0:

            _, val_acc, _ = evaluate_accuracy(
                val_iter, model, device, val_batch_count, metrics_val,
                figdump_cfg=None
            )

            # _ = evaluate_accuracy(
            #     test_iter, model, device, test_batch_count, metrics_test,
            #     figdump_cfg=None
            # )

            if val_acc > VAL_ACC:
                VAL_ACC = val_acc
                Best_model_filename = f"{epoch}Max_Acc_{VAL_ACC:.4f}.bin"
                torch.save(model, Best_model_filename)


    best_model = torch.load(Best_model_filename, map_location=device)
    best_model.to(device)
    best_model.eval()

    metrics_test_final = Metrics(num_classes=num_classes_news)

    result_test, test_acc, metrics_dict_test = evaluate_accuracy(
        test_iter, best_model, device, test_batch_count, metrics_test_final,
        figdump_cfg=None
    )

    print(f"[DONE] Best VAL Acc = {VAL_ACC:.4f}, ckpt = {Best_model_filename}")
    print("result_test", result_test)
    print("Test Metrics:", metrics_dict_test)

