

import joblib
import numpy as np

class GitHubClassifier:

    def __init__(self):
        self.words = open("words.txt", 'r', encoding='utf-8').read().splitlines()
        self.delimiters = ['.', ',', '(', ')', '[', ']', '{', '}', '=', ':',
                           '\'', '\"', '_', '\n', '\t', ' ', '#', '*', '%', '<-',
                           '`', '-', '\\', '/', '+', ';', '|', '&', '@', '<', '>', '^', '!']
        self.model = joblib.load("trained_model.sav")

    def split_sample(self, sample):
        """
        splits a string by self.delimiters
        :param sample: string (represents few lines of code)
        :return: array of strings
        """
        new_sample = sample.lower()
        for delimiter in self.delimiters:
            new_sample = new_sample.replace(delimiter, " ")
        return new_sample.split()

    def samples_to_vec(self, samples):
        """
        converts a samples represented by string (few lines of code) into
        a matrix X of floats
        :param samples: array of samples (strings)
        :return: numpy array (shape = num_of_samples * num_of_features)
        """
        del_len, words_len = len(self.delimiters), len(self.words)
        X = np.zeros((len(samples), words_len + del_len + 2))
        for n in range(len(samples)):
            vec = np.zeros(words_len + del_len + 2)
            vec[0] = len(min(samples[n].split('\n'), key=len))
            vec[1] = len(max(samples[n].split('\n'), key=len))
            for i in range(del_len):
                vec[i + 2] = samples[n].count(self.delimiters[i])
            sample = self.split_sample(samples[n])
            for j in range(words_len):
                vec[del_len + 2 + j] = sample.count(self.words[j])
            X[n] = vec
        return X

    def classify(self, X):
        """
        Receives a list of m unclassified pieces of code, and predicts for each
        one the Github project it belongs to.
        :param X: a numpy array of shape (m,) containing the code segments (strings)
        :return: y_hat - a numpy array of shape (m,) where each entry is a number between 0 and 6
        0 - building_tool
        1 - espnet
        2 - horovod
        3 - jina
        4 - PuddleHub
        5 - PySolFC
        6 - pytorch_geometric
        """
        return self.model.predict(self.samples_to_vec(X))


