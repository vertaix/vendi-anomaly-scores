from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


def symmetric_eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    A = 0.5 * (A + A.T)
    evals, evecs = torch.linalg.eigh(A)
    return evals, evecs


def density_matrix_from_raw_features(
    Phi: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    S = Phi.T @ Phi
    T = torch.trace(S).clamp_min(eps)
    rho = S / T
    rho = 0.5 * (rho + rho.T)
    return rho, T


def vendi_score_from_rho(
    rho: torch.Tensor,
    q: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    evals, _ = symmetric_eigh(rho)
    lam = evals.clamp_min(eps)

    if math.isclose(q, 1.0):
        log_vs = -(lam * lam.log()).sum()
    else:
        tr_rho_q = (lam**q).sum()
        log_vs = torch.log(tr_rho_q) / (1.0 - q)

    vs = torch.exp(log_vs)
    return vs, log_vs


@torch.no_grad()
def build_lowrank_rff_removal_state(
    Phi_base: torch.Tensor,
    q: float = 1.0,
    rank: int = 32,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    rho, T = density_matrix_from_raw_features(Phi_base, eps=eps)
    base_vs, base_log_vs = vendi_score_from_rho(rho, q=q, eps=eps)

    evals, evecs = symmetric_eigh(rho)
    lam = evals.clamp_min(eps)

    idx_desc = torch.argsort(lam, descending=True)
    lam_desc = lam[idx_desc]
    U_desc = evecs[:, idx_desc]

    r_eff = min(rank, lam_desc.numel())
    lam_r = lam_desc[:r_eff]
    U_r = U_desc[:, :r_eff]

    if math.isclose(q, 1.0):
        g_r = -(lam_r.log() + 1.0)
    else:
        tr_rho_q_r = (lam_r**q).sum().clamp_min(eps)
        g_r = (q * lam_r ** (q - 1.0)) / ((1.0 - q) * tr_rho_q_r)

    tr_GH_rho_approx = torch.sum(g_r * lam_r)

    return {
        "U_r": U_r,
        "lam_r": lam_r,
        "g_r": g_r,
        "tr_GH_rho_approx": tr_GH_rho_approx,
        "rho": rho,
        "T": T,
        "base_vs": base_vs,
        "base_log_vs": base_log_vs,
        "q": torch.tensor(float(q), device=Phi_base.device, dtype=Phi_base.dtype),
        "rank": torch.tensor(int(rank), device=Phi_base.device),
    }


@torch.no_grad()
def approximate_removal_delta_log_vs_raw_features_lowrank(
    Phi_base: torch.Tensor,
    q: float = 1.0,
    rank: int = 32,
    eps: float = 1e-12,
    state: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    if state is None:
        state = build_lowrank_rff_removal_state(
            Phi_base=Phi_base,
            q=q,
            rank=rank,
            eps=eps,
        )

    U_r = state["U_r"]
    g_r = state["g_r"]
    T = state["T"]
    tr_GH_rho_approx = state["tr_GH_rho_approx"]

    proj = Phi_base @ U_r
    qform_approx = torch.sum((proj * proj) * g_r.unsqueeze(0), dim=1)
    sq_norms = torch.sum(Phi_base * Phi_base, dim=1)
    denom = (T - sq_norms).clamp_min(eps)
    delta_log_vs_approx = -(
        qform_approx - sq_norms * tr_GH_rho_approx
    ) / denom

    return {
        "delta_log_vs_approx": delta_log_vs_approx,
        "qform_approx": qform_approx,
        "sq_norms": sq_norms,
        "proj": proj,
        "tr_GH_rho_approx": tr_GH_rho_approx,
        "base_vs": state["base_vs"],
        "base_log_vs": state["base_log_vs"],
        "T": T,
        "U_r": U_r,
        "lam_r": state["lam_r"],
        "g_r": g_r,
    }


@torch.no_grad()
def approximate_removal_delta_vs_raw_features_lowrank(
    Phi_base: torch.Tensor,
    q: float = 1.0,
    rank: int = 32,
    eps: float = 1e-12,
    state: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    out = approximate_removal_delta_log_vs_raw_features_lowrank(
        Phi_base=Phi_base,
        q=q,
        rank=rank,
        eps=eps,
        state=state,
    )
    out["delta_vs_approx"] = out["base_vs"] * out["delta_log_vs_approx"]
    return out
