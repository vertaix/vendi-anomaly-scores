import numpy as np
import torch


class KNN:
    """Average Euclidean kNN distance after row-wise L2 normalization."""

    def __init__(self, n_neighbors=5, eps=1e-12, device="auto", dtype="float32"):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = int(n_neighbors)
        self.eps = float(eps)
        self.device = device
        self.dtype = dtype

    def _resolve_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _resolve_dtype(self):
        if self.dtype == "float32":
            return torch.float32
        if self.dtype == "float64":
            return torch.float64
        raise ValueError("dtype must be 'float32' or 'float64'.")

    def _prepare_features(self, X):
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, self.eps)

    def _as_torch(self, X):
        return torch.as_tensor(X, device=self.device_, dtype=self.dtype_)

    def fit(self, X):
        fit_X = self._prepare_features(X)
        if fit_X.shape[0] < 2:
            raise ValueError(f"{self.__class__.__name__} requires at least 2 samples.")

        self.device_ = self._resolve_device()
        self.dtype_ = self._resolve_dtype()
        self._fit_X = fit_X
        self._fit_X_torch = self._as_torch(fit_X)
        self._fit_scores_ = self._score_fit_data()
        return self

    def _score_fit_data(self):
        n_fit = self._fit_X_torch.shape[0]
        k = min(self.n_neighbors, n_fit - 1)
        if k < 1:
            raise ValueError(f"{self.__class__.__name__} requires at least 2 samples.")

        distances = torch.cdist(self._fit_X_torch, self._fit_X_torch, p=2.0)
        diag_idx = torch.arange(n_fit, device=self.device_)
        distances[diag_idx, diag_idx] = torch.inf
        topk_distances = torch.topk(distances, k=k, largest=False, dim=1).values
        return topk_distances.mean(dim=1).detach().cpu().numpy()

    def _score_query_data(self, X_query):
        query_torch = self._as_torch(X_query)
        k = min(self.n_neighbors, self._fit_X_torch.shape[0])
        distances = torch.cdist(query_torch, self._fit_X_torch, p=2.0)
        topk_distances = torch.topk(distances, k=k, largest=False, dim=1).values
        return topk_distances.mean(dim=1).detach().cpu().numpy()

    def decision_function(self, X):
        X_prepared = self._prepare_features(X)
        scoring_fit_data = X_prepared.shape == self._fit_X.shape and np.array_equal(
            X_prepared,
            self._fit_X,
        )
        if scoring_fit_data:
            return self._fit_scores_.copy()
        return self._score_query_data(X_prepared)
