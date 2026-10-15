import numpy as np
from pyod.models.loda import LODA as PyODLODA


class LODA:
    """Thin wrapper around pyod LODA."""

    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state
        self.kwargs = dict(kwargs)

    def fit(self, X, y=None):
        if self.random_state is None:
            self.model = PyODLODA(**self.kwargs)
            self.model.fit(X, y)
            return self

        rng_state = np.random.get_state()
        try:
            np.random.seed(int(self.random_state))
            self.model = PyODLODA(**self.kwargs)
            self.model.fit(X, y)
        finally:
            np.random.set_state(rng_state)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)
