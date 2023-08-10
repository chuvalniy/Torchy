from sklearn.datasets import make_classification

import layer
import sequential
from tests.gradient_check import GradientCheck


def test_model():
    X, y = make_classification(n_samples=10, n_features=5, n_redundant=0)

    model = sequential.Sequential(
        layer.Linear(n_input=5, n_output=10),
        layer.BatchNorm1d(n_output=10),
        layer.ReLU(),
        layer.Linear(n_input=10, n_output=2)
    )

    GradientCheck.check_model_gradient(model, X, y)
