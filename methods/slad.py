from deepod.models.tabular import SLAD as DeepODSLAD


class SLAD:
    """Thin wrapper around deepod SLAD with a stable scoring interface."""

    def __init__(self, device="cpu", verbose=0, random_state=None, **kwargs):
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.kwargs = dict(kwargs)

    def _build_model(self):
        for params in (
            {
                "device": self.device,
                "verbose": self.verbose,
                "random_state": self.random_state,
                **self.kwargs,
            },
            {"device": self.device, "random_state": self.random_state, **self.kwargs},
            {"verbose": self.verbose, "random_state": self.random_state, **self.kwargs},
            {"random_state": self.random_state, **self.kwargs},
            {"device": self.device, "verbose": self.verbose, **self.kwargs},
            {"device": self.device, **self.kwargs},
            {"verbose": self.verbose, **self.kwargs},
            {**self.kwargs},
        ):
            try:
                return DeepODSLAD(**params)
            except TypeError:
                continue
        return DeepODSLAD()

    def fit(self, X, y=None):
        self.model = self._build_model()
        self.model.fit(X, y)
        return self

    def decision_function(self, X):
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        return self.model.predict(X)
