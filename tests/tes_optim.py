import numpy as np
from sklearn.datasets import make_classification

import layer
import sequential
from loss import CrossEntropyLoss
from optim import _Optim, SGD, MomentumSGD, Adam, Adagrad, RMSProp
from value import Value


def get_model():
    model = sequential.Sequential(
        layer.Linear(n_input=20, n_output=10),
        layer.BatchNorm1d(n_output=10),
        layer.ReLU(),
        layer.Linear(n_input=10, n_output=2)
    )

    return model


def train_model(model: sequential.Sequential, optimizer: _Optim):
    X, y = make_classification(n_samples=10, n_features=20, n_redundant=5)

    criterion = CrossEntropyLoss()

    for epoch in range(100):
        predictions = model(X)

        loss, grad = criterion(predictions, y)

        optimizer.zero_grad()
        model.backward(grad)
        optimizer.step()

        indices = np.argmax(predictions, 1)

        accuracy = np.sum(indices == y) / y.shape[0]
        print(f"Accuracy: {np.round(accuracy * 100, 4)}, Loss: {loss}")

