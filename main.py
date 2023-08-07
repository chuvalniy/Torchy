import numpy as np
# only for using datasets
from sklearn.datasets import make_classification

import layer
import loss
import optim
import sequential

if __name__ == "__main__":
    np.random.seed(42)

    X, y = make_classification(n_samples=50, n_features=5, n_redundant=0)

    model = sequential.Sequential(
        layer.Linear(n_input=5, n_output=8),
        layer.ReLU(),
        layer.Linear(n_input=8, n_output=2)
    )

    criterion = loss.CrossEntropyLoss()
    optimizer = optim.Adam(model.params(), lr=1e-1, weight_decay=1e-5)

    for epoch in range(500):
        predictions = model(X)
        loss, grad = criterion(predictions, y)

        optimizer.zero_grad()
        model.backward(grad)
        optimizer.step()

        indices = np.argmax(predictions, axis=1)
        accuracy = (np.sum(indices == y)) / y.shape[0]

        print(f"Loss: {loss}, Accuracy: {accuracy}")