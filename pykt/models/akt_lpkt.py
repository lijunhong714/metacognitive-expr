#!/usr/bin/env python
# coding=utf-8
"""AKT-LPKT: 双通路知识追踪（AKT 主干 + LPKT 循环调控器）。

设计约束
--------
1. **作用点**：证据权重 ``omega`` 作用在状态更新上（LPKT 侧的学习增量、遗忘门，
   以及 AKT 知识检索器的 value 流），不作用在输出概率上。
2. **成因条件化**：监控头输出成因后验 ``pi``（K 个通道），状态增量是 K 个
   **各自学习得到的**更新算子的 pi-加权混合，不是单一标量缩放。
3. **无循环依赖**：``omega`` / ``pi`` 只由 LPKT 侧的更新前状态 h_tilde_pre、
   题目难度、过程特征与本次 response 计算，不使用 AKT 检索器的输出。
4. **无启发式参数**：本文件不含硬编码阈值、先验概率、固定增益、温度系数或
   自监督目标中的偏置常数。数值行为全部由可学习参数决定；出现的常数只有
   LPKT 原有的 ``(lg+1)/2`` 映射与数值安全用的 ``clamp_min``。

在 pykt 中的接口
----------------
构造::

    AKTLPKT(num_c, num_q, num_at, num_it, num_phi, q_matrix=..., **model_config,
            emb_type=..., emb_path=...)

前向::

    y, reg_loss = model(cc, cr, cq, itseqs, atseqs, phi, mask)

``y`` 形状 (B, T)，与 AKT / LPKT 一致，由 ``model_forward`` 再切 ``[:,1:]``。
``itseqs`` / ``atseqs`` 由 pykt 的 ``LPKTDataset`` 提供；``phi``（过程特征，如作答
时长、同题重试序数、提示次数、同概念间隔）需要自定义 dataloader，缺失时传 None。

注册（各一行分支）：
  * ``pykt/models/init_model.py``     : ``elif model_name == "akt_lpkt": ...``
  * ``pykt/models/train_model.py``    : ``model_forward`` / ``cal_loss`` 的 akt 分支
  * ``pykt/models/evaluate_model.py`` : ``evaluate`` 的 akt 分支
"""
import math
from enum import IntEnum

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


# ======================================================================================
# 通路 A：AKT 主干（单调注意力 + 知识检索器；检索器的 value 流由 omega 门控）
# ======================================================================================
def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """AKT 式注意力：在 softmax 前乘以与"相对距离"相关的可学习单调项。"""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    mask = mask.to(scores.device)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen, device=scores.device).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(scores.device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :].float()
        dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.0)
        dist_scores = dist_scores.sqrt().detach()

    softplus = nn.Softplus()
    gamma = -1.0 * softplus(gamma).unsqueeze(0)
    if pdiff is None:
        total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1],
                                         pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(torch.clamp((dist_scores * gamma * diff).exp(),
                                               min=1e-5), max=1e5)
    scores = scores * total_effect
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen, device=scores.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same=True):
        super().__init__()
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.v_linear = nn.Linear(d_model, d_model, bias=True)
        self.k_linear = nn.Linear(d_model, d_model, bias=True)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if not self.kq_same:
            xavier_uniform_(self.q_linear.weight)
        constant_(self.k_linear.bias, 0.0)
        constant_(self.v_linear.bias, 0.0)
        if not self.kq_same:
            constant_(self.q_linear.bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None, v_gate=None):
        bs = q.size(0)
        if v_gate is not None:
            # 逐位置缩放 value 流 == "这条历史交互贡献多少证据"
            v = v * v_gate.unsqueeze(-1)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad,
                           self.gammas, pdiff)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out_proj(concat)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads,
                                                   dropout, kq_same)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None, v_gate=None):
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
        if mask == 0:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask,
                                           zero_pad=True, pdiff=pdiff, v_gate=v_gate)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask,
                                           zero_pad=False, pdiff=pdiff, v_gate=v_gate)
        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class AKTArchitecture(nn.Module):
    """blocks_1 编码交互历史；blocks_2 是知识检索器（看不到当前 response）。"""

    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, kq_same=True):
        super().__init__()
        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model, d_model // n_heads, d_ff, n_heads, dropout, kq_same)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model, d_model // n_heads, d_ff, n_heads, dropout, kq_same)
            for _ in range(n_blocks * 2)
        ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data, evidence_gate=None):
        y = qa_embed_data
        x = q_embed_data
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False,
                          pdiff=pid_embed_data)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True,
                          pdiff=pid_embed_data, v_gate=evidence_gate)
                flag_first = True
        return x


# ======================================================================================
# 通路 B：LPKT 式循环调控器（显式逐概念状态 + 成因混合更新 + 证据权重）
# ======================================================================================
class RecurrentRegulator(nn.Module):
    """维护逐概念状态 h ∈ R^{B×(n_c+1)×d_k}，逐步输出 omega 与 pi。

    每步：
      1. 学习单元得到候选学习增益 —— 每个成因一个独立算子；
      2. 由 pi 混合成实际增量，再由 omega 缩放；
      3. 遗忘门读入 omega，"保留还是遗忘"的方向由网络自己学。
    """

    def __init__(self, n_concept, n_it, n_at, n_phi, d_model, d_k, n_causes,
                 dropout, q_matrix):
        super().__init__()
        self.n_concept = n_concept
        self.d_k = d_k
        self.n_causes = n_causes
        self.register_buffer("q_matrix", q_matrix.float())

        self.it_embed = nn.Embedding(n_it + 1, d_k)
        self.at_embed = nn.Embedding(n_at + 1, d_k)
        xavier_uniform_(self.it_embed.weight)
        xavier_uniform_(self.at_embed.weight)

        self.phi_proj = nn.Linear(max(n_phi, 1), d_k, bias=False)
        # 作答时长通道的整体开关（可学习，不预设是否使用）
        self.at_scale = nn.Parameter(torch.zeros(1))

        self.learning_cell = nn.Linear(d_model + 3 * d_k, d_k)
        self.learning_norm = nn.LayerNorm(d_k)
        self.dropout = nn.Dropout(dropout)

        gain_in = 4 * d_k
        self.cause_gain = nn.ModuleList([nn.Linear(gain_in, d_k) for _ in range(n_causes)])
        self.cause_gate = nn.ModuleList([nn.Linear(gain_in, d_k) for _ in range(n_causes)])

        mon_in = 2 * d_k + 3 + d_model          # h_tilde_pre + phi + (difficulty, onehot(r)) + 交互表示
        self.monitor = nn.Sequential(
            nn.Linear(mon_in, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.pi_head = nn.Linear(d_model, n_causes)
        self.omega_head = nn.Linear(d_model, 1)

        self.forget_gate = nn.Linear(3 * d_k + 1, d_k)

    def forward(self, inter_emb, q_rows, it_idx, at_idx, phi, rids, pid_diff, mask):
        """q_rows: (B,T,n_c+1) 已由父模块按 q_matrix 查表展开的题目-概念指示矩阵。"""
        bs, seqlen, _ = inter_emb.shape
        nq = self.n_concept + 1
        if pid_diff is None:
            # 概念级数据集（num_q == 0）没有题目难度嵌入，难度项置零
            pid_diff = inter_emb.new_zeros(bs, seqlen, 1)

        it_emb = self.it_embed(it_idx)
        at_emb = self.at_embed(at_idx) * torch.sigmoid(self.at_scale).view(1, 1, 1)
        phi_emb = self.phi_proj(phi) if phi is not None else torch.zeros_like(at_emb)

        h = torch.zeros(bs, nq, self.d_k, device=inter_emb.device)
        learning_pre = torch.zeros(bs, self.d_k, device=inter_emb.device)

        omega_out = inter_emb.new_zeros(bs, seqlen)
        pi_out = inter_emb.new_zeros(bs, seqlen, self.n_causes)
        h_read_out = inter_emb.new_zeros(bs, seqlen, self.d_k)

        for t in range(seqlen):
            q_row = q_rows[:, t]                                       # (B,n_c+1)
            cnt = q_row.sum(dim=-1, keepdim=True).clamp_min(1.0)       # 数值安全
            h_tilde_pre = torch.bmm(q_row.unsqueeze(1), h).squeeze(1) / cnt
            h_read_out[:, t] = h_tilde_pre

            learning = self.learning_norm(self.learning_cell(torch.cat(
                [inter_emb[:, t], it_emb[:, t], at_emb[:, t], phi_emb[:, t]], dim=-1)))
            learning = self.dropout(learning)

            base = torch.cat([learning_pre, it_emb[:, t], learning, h_tilde_pre], dim=-1)
            gains = torch.stack([torch.tanh(op(base)) for op in self.cause_gain], dim=1)
            gates = torch.stack([torch.sigmoid(op(base)) for op in self.cause_gate], dim=1)
            candidate = gates * ((gains + 1.0) / 2.0)                  # LPKT 的 (lg+1)/2 映射

            pi_in = torch.cat([
                h_tilde_pre,
                phi_emb[:, t],
                pid_diff[:, t],
                F.one_hot(rids[:, t].clamp(0, 1).long(), 2).float(),
                inter_emb[:, t],
            ], dim=-1)
            z = self.monitor(pi_in)
            pi = F.softmax(self.pi_head(z), dim=-1)                    # (B,K)
            omega = torch.sigmoid(self.omega_head(z))                  # (B,1)

            delta = omega * (pi.unsqueeze(-1) * candidate).sum(dim=1)  # (B,d_k)

            gamma_f = torch.sigmoid(self.forget_gate(torch.cat([
                h,
                delta.unsqueeze(1).expand(-1, nq, -1),
                it_emb[:, t].unsqueeze(1).expand(-1, nq, -1),
                omega.unsqueeze(1).expand(-1, nq, -1),
            ], dim=-1)))

            # 把 delta 写到该题涉及的每个概念上：(B,n_c+1,1) @ (B,1,d_k) -> (B,n_c+1,d_k)
            h_new = torch.bmm(q_row.unsqueeze(2), delta.unsqueeze(1)) + gamma_f * h
            step_mask = mask[:, t].float().view(bs, 1, 1)
            h = step_mask * h_new + (1.0 - step_mask) * h

            omega_out[:, t] = omega.squeeze(-1) * mask[:, t].float()
            pi_out[:, t] = pi
            learning_pre = learning

        return omega_out, pi_out, h_read_out


# ======================================================================================
# 双通路模型
# ======================================================================================
class AKTLPKT(nn.Module):
    """AKT 主干 + LPKT 循环调控器。

    注意 pykt 的命名习惯：``n_question`` 是**概念数**（num_c），``n_pid`` 是
    **题目 id 数**（num_q），与 AKT 保持一致。
    """

    def __init__(self, n_question, n_pid, n_at, n_it, n_phi, d_model, n_blocks,
                 dropout, d_ff=256, kq_same=1, final_fc_dim=512, num_attn_heads=8,
                 separate_qa=False, l2=1e-5, n_causes=3, d_k=None, q_matrix=None,
                 emb_type="qid", emb_path="", pretrain_dim=768, **kwargs):
        super().__init__()
        self.model_name = "akt_lpkt"
        self.n_question = n_question
        self.n_pid = n_pid
        self.n_at = n_at
        self.n_it = n_it
        self.d_model = d_model
        self.n_causes = n_causes
        self.d_k = d_k if d_k is not None else d_model
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.l2 = l2

        # 共享交互表示（AKT 的 base_emb）
        self.q_embed = nn.Embedding(n_question, d_model)
        if separate_qa:
            self.qa_embed = nn.Embedding(2 * n_question + 1, d_model)
        else:
            self.qa_embed = nn.Embedding(2, d_model)
        if n_pid > 0:
            self.difficult_param = nn.Embedding(n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(n_question + 1, d_model)
            self.qa_embed_diff = nn.Embedding(2 * n_question + 1, d_model)

        self.akt = AKTArchitecture(n_blocks, d_model, d_ff, num_attn_heads,
                                   dropout, kq_same == 1)

        if q_matrix is None:
            q_matrix = torch.eye(n_question + 1)
        self.regulator = RecurrentRegulator(n_question, n_it, n_at, n_phi,
                                            d_model, self.d_k, n_causes,
                                            dropout, q_matrix)

        self.out = nn.Sequential(
            nn.Linear(2 * d_model + self.d_k, final_fc_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.reset()

    def reset(self):
        if self.n_pid > 0:
            for p in self.parameters():
                if p.dim() >= 1 and p.size(0) == self.n_pid + 1:
                    torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_embed_data = self.qa_embed(q_data + self.n_question * target)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, cids, rids, qids, itseqs=None, atseqs=None, phi=None,
                mask=None, qtest=False, return_aux=False):
        """cids=概念序列(cc)，rids=作答序列(cr)，qids=题目 id 序列(cq)。"""
        bs, seqlen = cids.shape
        q_embed_data, qa_embed_data = self.base_emb(cids, rids)

        pid_embed_data = None
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(cids)
            pid_embed_data = self.difficult_param(qids)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(rids)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.0).sum() * self.l2
        else:
            c_reg_loss = torch.zeros((), device=cids.device)

        if mask is None:
            mask = torch.ones_like(cids, dtype=torch.bool)
        mask = mask.bool()
        if itseqs is None:
            itseqs = torch.zeros_like(cids)
        if atseqs is None:
            atseqs = torch.zeros_like(cids)
        it_idx = itseqs.clamp(min=0, max=self.n_it)
        at_idx = atseqs.clamp(min=0, max=self.n_at)

        # 通路 B 先跑：得到逐步的证据权重（因果，无循环依赖）
        # 概念级数据集（num_q == 0，如 assist2015）没有题目 id，cq 为空，退化用概念 id 充当题目索引。
        # 仅在模型本身没有题目维度时生效，有题目 id 的数据集一律走原路径
        if self.n_pid == 0 and (qids is None or qids.size(1) == 0):
            qids = cids
        q_rows = self.regulator.q_matrix[qids]                          # (B,T,n_c+1)
        q_rows = q_rows * mask.unsqueeze(-1).float()
        omega, pi, h_read = self.regulator(qa_embed_data, q_rows, it_idx, at_idx,
                                           phi, rids, pid_embed_data, mask)

        # 通路 A：知识检索器的 value 流被 omega 门控
        d_output = self.akt(q_embed_data, qa_embed_data, pid_embed_data,
                            evidence_gate=omega)

        concat_q = torch.cat([d_output, h_read, q_embed_data], dim=-1)
        preds = torch.sigmoid(self.out(concat_q).squeeze(-1))

        if return_aux:
            aux = {"omega": omega, "pi": pi, "h_read": h_read, "x_akt": d_output}
            return preds, c_reg_loss, aux
        if qtest:
            return preds, c_reg_loss, concat_q
        return preds, c_reg_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 4, 12
    num_c, num_q = 10, 20          # pykt 命名：num_c 概念数、num_q 题目数
    n_it, n_at, n_phi = 50, 50, 4

    q_matrix = torch.zeros(num_q + 1, num_c + 1)
    for q in range(num_q + 1):
        q_matrix[q, q % num_c] = 1.0

    model = AKTLPKT(
        n_question=num_c, n_pid=num_q, n_at=n_at, n_it=n_it, n_phi=n_phi,
        d_model=32, n_blocks=1, dropout=0.1, d_ff=64, num_attn_heads=4,
        n_causes=3, q_matrix=q_matrix, emb_type="qid",
    ).to(device)
    cids = torch.randint(0, num_c, (B, T), device=device)
    qids = torch.randint(0, num_q, (B, T), device=device)
    rids = torch.randint(0, 2, (B, T), device=device)
    its = torch.randint(0, n_it, (B, T), device=device)
    ats = torch.randint(0, n_at, (B, T), device=device)
    phi = torch.randn(B, T, n_phi, device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mask[:, -3:] = False

    preds, reg_loss, aux = model(cids, rids, qids, its, ats, phi, mask, return_aux=True)
    n_param = sum(p.numel() for p in model.parameters())
    print("params          :", f"{n_param:,}")
    print("preds           :", tuple(preds.shape), "| finite:", bool(torch.isfinite(preds).all()))
    print("reg_loss        :", round(float(reg_loss.detach()), 6))
    print("omega           :", tuple(aux["omega"].shape), "| pi:", tuple(aux["pi"].shape),
          "| h_read:", tuple(aux["h_read"].shape))
    w = aux["omega"].detach()
    p = aux["pi"].detach()
    print("omega mean/std  : %.4f / %.4f" % (w.mean(), w.std()))
    print("pi row-sum min/max: %.6f / %.6f" % (p.sum(-1).min(), p.sum(-1).max()))
    print("padded omega==0 :", bool((aux["omega"][:, -3:] == 0).all()))
