import numpy as np
# only for using datasets
from sklearn.datasets import make_classification

import layer
import sequential
from tests.gradient_check import GradientCheck
from tests.test_layer import test_batchnorm1d
from tests.test_loss import test_cross_entropy_single, test_softmax, test_cross_entropy_batch


def train_model():
    X, y = make_classification(n_samples=10, n_features=5, n_redundant=0)

    model = sequential.Sequential(
        layer.Linear(n_input=5, n_output=8),
        layer.BatchNorm1d(n_output=8),
        layer.ReLU(),
        layer.Linear(n_input=8, n_output=2)
    )

    GradientCheck.check_model_gradient(model, X, y)
    # criterion = CrossEntropyLoss()
    # optimizer = optim.MomentumSGD(model.params(), lr=1e-1, weight_decay=1e-5)
    # scheduler = ReduceLROnPlateau(optimizer)
    #
    # for epoch in range(100):
    #     predictions = model(X)
    #     loss, grad = criterion(predictions, y)
    #
    #     optimizer.zero_grad()
    #     model.backward(grad)
    #     optimizer.step()
    #
    #     indices = np.argmax(predictions, axis=1)
    #     accuracy = (np.sum(indices == y)) / y.shape[0]
    #
    #     print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Learning rate: {optimizer.lr}")
    #
    #     scheduler.step(loss)


if __name__ == "__main__":
    np.random.seed(42)
