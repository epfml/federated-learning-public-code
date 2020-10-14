# -*- coding: utf-8 -*-


class EarlyStoppingTracker(object):
    def __init__(self, patience, delta=0, mode="max"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_value = None
        self.counter = 0

    def __call__(self, value):
        if self.patience is None or self.patience <= 0:
            return False

        if self.best_value is None:
            self.best_value = value
            self.counter = 0
            return False

        if self.mode == "max":
            if value > self.best_value + self.delta:
                return self._positive_update(value)
            else:
                return self._negative_update(value)
        elif self.mode == "min":
            if value < self.best_value - self.delta:
                return self._positive_update(value)
            else:
                return self._negative_update(value)
        else:
            raise ValueError(f"Illegal mode for early stopping: {self.mode}")

    def _positive_update(self, value):
        self.counter = 0
        self.best_value = value
        return False

    def _negative_update(self, value):
        self.counter += 1
        if self.counter > self.patience:
            return True
        else:
            return False
