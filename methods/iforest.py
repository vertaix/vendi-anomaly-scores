from pyod.models.iforest import IForest as PyODIForest


class IForest:
    """Thin wrapper around pyod Isolation Forest."""

    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state
        self.kwargs = dict(kwargs)

    def fit(self, X, y=None):
        self.model = PyODIForest(random_state=self.random_state, **self.kwargs)
        self.model.fit(X, y)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)
