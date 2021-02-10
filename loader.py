import numpy as np
import tensorflow as tf
from scipy import sparse as sp

class Loader():
    def __init__(self, feature_matrix, adjacency_matrix, label, epochs=None, sample_weights=None):
        self.feature_matrix = feature_matrix
        self.adjacency_matrix = adjacency_matrix
        self.epochs = epochs
        self.sample_weights = sample_weights
        self.label = label
        self.data = [self.feature_matrix, self.adjacency_matrix, self.label]

    def collate(self, data):
        output = list(data)
        output = tuple(output)
        output = (output[:-1], output[-1])
        if self.sample_weights is not None:
            output += (self.sample_weights,)
        return tuple(output)

    def load(self):
        output = self.collate(self.data)
        return tf.data.Dataset.from_tensors(output).repeat(self.epochs)
