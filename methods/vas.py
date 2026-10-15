import numpy as np
import torch

from methods._vendi_lowrank import (
    approximate_removal_delta_vs_raw_features_lowrank,
    build_lowrank_rff_removal_state,
)


class VAS:
    """Adaptive-q cosine Vendi removal using low-rank first-order scoring."""

    def __init__(
        self,
        q_mode="adaptive",
        fixed_q=None,
        adaptive_q_floor=0.1,
        rank="full",
        device="auto",
        eps=1e-12,
    ):
        self.q_mode = str(q_mode)
        self.fixed_q = None if fixed_q is None else float(fixed_q)
        self.adaptive_q_floor = float(adaptive_q_floor)
        self.rank = rank
        self.device = device
        self.eps = float(eps)

        if self.q_mode not in {"adaptive", "fixed"}:
            raise ValueError("q_mode must be either 'adaptive' or 'fixed'.")
        if self.q_mode == "fixed":
            if self.fixed_q is None:
                raise ValueError("fixed_q must be provided when q_mode='fixed'.")
            if self.fixed_q <= 0:
                raise ValueError("fixed_q must be positive.")
        if self.adaptive_q_floor <= 0:
            raise ValueError("adaptive_q_floor must be positive.")

    def _resolve_device(self):
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _l2_normalize(self, X):
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, self.eps)

    def _resolve_rank(self, n_features):
        if self.rank == "full":
            return int(n_features)
        rank = int(self.rank)
        if rank <= 0:
            raise ValueError("rank must be 'full' or a positive integer.")
        return min(rank, int(n_features))

    def _adaptive_q_from_lambda1_features(self, phi_base_np):
        density = phi_base_np.T @ phi_base_np
        trace = float(np.trace(density))
        if trace <= self.eps:
            raise ValueError("Cannot compute adaptive q from zero-trace feature density")
        density = density / trace
        density = 0.5 * (density + density.T)
        lambda1 = float(np.linalg.eigvalsh(density)[-1])
        adaptive_q = max(2.0 * (1.0 - lambda1), self.adaptive_q_floor)
        return float(adaptive_q), lambda1

    def fit(self, X):
        self.device_ = self._resolve_device()
        self.phi_base_np_ = self._l2_normalize(X)
        if self.phi_base_np_.shape[0] < 2:
            raise ValueError("VAS requires at least two samples.")

        if self.q_mode == "fixed":
            self.resolved_q_ = float(self.fixed_q)
            self.lambda1_ = None
        else:
            self.resolved_q_, self.lambda1_ = self._adaptive_q_from_lambda1_features(
                self.phi_base_np_
            )
        self.rank_ = self._resolve_rank(self.phi_base_np_.shape[1])
        self.phi_base_ = torch.from_numpy(self.phi_base_np_).to(
            device=self.device_,
            dtype=torch.float64,
        )
        self.state_ = build_lowrank_rff_removal_state(
            Phi_base=self.phi_base_,
            q=self.resolved_q_,
            rank=self.rank_,
            eps=self.eps,
        )
        return self

    def decision_function(self, X):
        phi = self._l2_normalize(X)
        if phi.shape == self.phi_base_np_.shape and np.array_equal(phi, self.phi_base_np_):
            phi_torch = self.phi_base_
            state = self.state_
        else:
            phi_torch = torch.from_numpy(phi).to(device=self.device_, dtype=torch.float64)
            state = None

        out = approximate_removal_delta_vs_raw_features_lowrank(
            Phi_base=phi_torch,
            q=self.resolved_q_,
            rank=self.rank_,
            eps=self.eps,
            state=state,
        )
        return -out["delta_vs_approx"].detach().cpu().numpy()
