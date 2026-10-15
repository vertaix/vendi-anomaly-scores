from pyod.models.lof import LOF as PyODLOF


class LOF:
    """Thin wrapper around pyod LOF."""

    def __init__(self, **kwargs):
        self.model = PyODLOF(**kwargs)

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)
