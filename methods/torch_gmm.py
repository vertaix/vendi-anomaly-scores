import numpy as np
import torch


class TorchGMM:
    """torchgmm tied-covariance GMM with hard-assignment Mahalanobis scores."""

    def __init__(
        self,
        min_components=1,
        max_components=10,
        covariance_type="tied",
        reg_covar=1e-6,
        n_init=1,
        max_iter=100,
        random_state=None,
        device="auto",
        tol=1e-3,
        init_iters=10,
        dtype="float32",
        reg_covar_retry_multipliers=(1.0, 10.0, 100.0, 1000.0),
    ):
        self.min_components = int(min_components)
        self.max_components = int(max_components)
        self.covariance_type = covariance_type
        self.reg_covar = float(reg_covar)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.random_state = 0 if random_state is None else int(random_state)
        self.device = device
        self.tol = float(tol)
        self.init_iters = int(init_iters)
        self.dtype = dtype
        self.reg_covar_retry_multipliers = tuple(float(v) for v in reg_covar_retry_multipliers)

        if self.min_components < 1:
            raise ValueError("min_components must be at least 1.")
        if self.max_components < self.min_components:
            raise ValueError("max_components must be >= min_components.")
        if self.covariance_type != "tied":
            raise ValueError("TorchGMM requires covariance_type='tied'.")
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if not self.reg_covar_retry_multipliers:
            raise ValueError("reg_covar_retry_multipliers must not be empty.")

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

    def _l2_normalize(self, X):
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-12)

    def _as_torch(self, X):
        return torch.as_tensor(X, device=self.device_, dtype=self._resolve_dtype())

    def _cholesky_with_jitter(self, covariance):
        eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        jitter = 0.0
        for attempt in range(6):
            try:
                adjusted = covariance + jitter * eye
                return adjusted, torch.linalg.cholesky(adjusted)
            except RuntimeError:
                jitter = self.reg_covar * (10.0 ** attempt)
        adjusted = covariance + jitter * eye
        return adjusted, torch.linalg.cholesky(adjusted)

    def _precision_from_covariance(self, covariance):
        covariance, chol = self._cholesky_with_jitter(covariance)
        try:
            return torch.cholesky_inverse(chol)
        except RuntimeError:
            return torch.linalg.pinv(covariance)

    def fit(self, X):
        try:
            from torchgmm.bayes import GaussianMixture as TorchGaussianMixture
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TorchGMM requires the optional 'torchgmm' package. "
                "Install dependencies from requirements.txt."
            ) from exc

        X_norm = self._l2_normalize(X)
        n_samples, n_features = X_norm.shape
        if n_samples < 1:
            raise ValueError("TorchGMM requires at least 1 sample.")

        self.device_ = self._resolve_device()
        X_torch = self._as_torch(X_norm)
        max_components = min(self.max_components, n_samples)

        self.bic_scores_ = []
        best_bic = np.inf
        best_model = None
        self.reg_covar_used_ = None

        for n_components in range(self.min_components, max_components + 1):
            best_component_model = None
            best_component_nll = np.inf
            best_component_reg_covar = None

            for init_idx in range(self.n_init):
                seed = self.random_state + 1009 * n_components + init_idx
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                model, total_nll, used_reg_covar, _ = self._fit_torchgmm_with_reg_retries(
                    TorchGaussianMixture=TorchGaussianMixture,
                    X_torch=X_torch,
                    n_components=n_components,
                )
                if total_nll < best_component_nll:
                    best_component_nll = total_nll
                    best_component_model = model
                    best_component_reg_covar = used_reg_covar

            if best_component_model is None:
                self.bic_scores_.append((n_components, np.inf))
                continue

            n_params = (
                n_components * n_features
                + (n_components - 1)
                + n_features * (n_features + 1) / 2
            )
            bic = float(2.0 * best_component_nll + n_params * np.log(n_samples))
            self.bic_scores_.append((n_components, bic))
            if bic < best_bic:
                best_bic = bic
                best_model = best_component_model
                self.reg_covar_used_ = best_component_reg_covar

        if best_model is None:
            raise RuntimeError("All TorchGMM BIC candidates failed.")

        self.best_model_ = best_model
        self.best_n_components_ = int(best_model.num_components)
        self.best_bic_ = float(best_bic)
        self.weights_ = best_model.model_.component_probs.detach().cpu().numpy()
        self.means_ = best_model.model_.means.detach().cpu().numpy()
        covariance = best_model.model_.covariances
        self.covariance_ = covariance.detach().cpu().numpy()
        self.precision_ = self._precision_from_covariance(covariance).detach().cpu().numpy()
        return self

    def _fit_torchgmm_with_reg_retries(self, TorchGaussianMixture, X_torch, n_components):
        failures = []
        for multiplier in self.reg_covar_retry_multipliers:
            used_reg_covar = self.reg_covar * multiplier
            try:
                model = TorchGaussianMixture(
                    num_components=n_components,
                    covariance_type="tied",
                    init_strategy="kmeans",
                    convergence_tolerance=self.tol,
                    covariance_regularization=used_reg_covar,
                    batch_size=None,
                    trainer_params=self._torchgmm_trainer_params(),
                )
                model.fit(X_torch)
                total_nll = float(model.score_samples(X_torch).reshape(-1).sum().detach().cpu().item())
                return model, total_nll, used_reg_covar, failures
            except Exception as exc:
                failures.append((used_reg_covar, f"{exc.__class__.__name__}: {exc}"))
        return None, np.inf, None, failures

    def _torchgmm_trainer_params(self):
        if self.device_.type == "cuda":
            accelerator = "gpu"
            devices = 1
        elif self.device_.type == "cpu":
            accelerator = "cpu"
            devices = 1
        else:
            accelerator = "auto"
            devices = "auto"
        return {
            "accelerator": accelerator,
            "devices": devices,
            "max_epochs": self.max_iter,
            "enable_progress_bar": False,
            "enable_checkpointing": False,
            "logger": False,
            "num_sanity_val_steps": 0,
        }

    def decision_function(self, X):
        X_norm = self._l2_normalize(X)
        X_torch = self._as_torch(X_norm)
        means = torch.as_tensor(self.means_, device=self.device_, dtype=X_torch.dtype)
        precision = torch.as_tensor(self.precision_, device=self.device_, dtype=X_torch.dtype)
        means_precision = means @ precision
        x_precision = X_torch @ precision
        quad_x = (x_precision * X_torch).sum(dim=1)
        quad_means = (means_precision * means).sum(dim=1)
        cross = X_torch @ means_precision.T
        distances = (quad_x[:, None] - 2.0 * cross + quad_means[None, :]).clamp_min(0.0)
        assignments = self.best_model_.predict(X_torch)
        scores = distances[torch.arange(distances.shape[0], device=self.device_), assignments]
        return scores.detach().cpu().numpy()
