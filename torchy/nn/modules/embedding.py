import numpy as np

from torchy.nn.initializations import init
from torchy.nn.values import Value
from .module import Module


class Embedding(Module):
    """
    Word embedding layer.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        :param num_embeddings: int - dictionary size.
        :param embedding_dim: int - embedding dimension for each word in dictionary.
        """
        super(Embedding, self).__init__()

        self.weight = init.normal(shape=(num_embeddings, embedding_dim))
        self.x = None

    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Computes forward pass for embedding layer.

        :param x: numpy array (batch_size, dictionary_size) - incoming data.
        :return: numpy array (batch_size, sequence_length, embedding_dim) - word embeddings.
        """
        self.x = np.copy(x)

        return self.weight.data[self.x]

    def backward(self, d_out: np.ndarray) -> None:
        """
        Computes backward pass with respect to incoming data.

        :param d_out: numpy array (batch_size, sequence_length, embedding_dim) -
        gradient of loss function with respect to output of forward pass.
        """
        np.add.at(self.weight.grad, self.x, d_out)

    @classmethod
    def from_pretrained(cls, weight: np.ndarray) -> 'Embedding':
        """
        Takes pretrained weight parameters and creates new instance of an embedding object
        with pretrained weights.

        :param weight: numpy array (num_embeddings, embedding_dim) - pretrained embedding weights.
        :return: Embedding - embedding class instance with pretrained weights.
        """
        num_embeddings, embedding_dim = weight.shape
        embedding = cls(num_embeddings, embedding_dim)
        embedding.weight = Value(weight)

        return embedding
