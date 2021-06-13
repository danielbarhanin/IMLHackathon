import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split


class ParseProjects:
    """
    parses the coding files projects, extracting features and creating a RandomForest classifier.
    """

    def __init__(self):
        self.delimiters = ['.', ',', '(', ')', '[', ']', '{', '}', '=', ':',
                           '\'', '\"', '_', '\n', '\t', ' ', '#', '*', '%', '<-',
                           '`', '-', '\\', '/', '+', ';', '|', '&', '@', '<', '>', '^', '!']

        self.projects = ["building_tool", "espnet", "horovod", "jina", "PaddleHub", "PySolFC", "pytorch_geometric"]

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

    def parse_files(self):
        """
        parses the files of the dataset for the training - a code file for each one of the 7 projects.
        :return: X - the train data containing strings (represents few lines of code) , y - the labeled vector.
        """
        X = []
        y = []
        for i in range(len(self.projects)):
            with open("Projects_data\\" + self.projects[i] + '_all_data.txt', 'r', encoding='utf-8') as f:
                while True:
                    num_lines = np.random.randint(1, 6)
                    sample = ""
                    for line in range(num_lines):
                        sample += f.readline()
                    if sample == "":
                        break

                    X.append(sample)
                    y.append(i)
        return X, y

    def build_words_features(self, X, y):
        """
        extracts words features from the given parsed code files of the projects.
        :param X: the parsed code files.
        :param y: the labeled vector
        :return:
        """
        all_words = set()
        sets = np.array([set(), set(), set(), set(), set(), set(), set()])
        dicts = np.array([{}, {}, {}, {}, {}, {}, {}])

        for i in range(len(y)):
            for word in self.split_sample(X[i]):
                if word in dicts[y[i]]:
                    dicts[y[i]][word] += 1
                else:
                    dicts[y[i]][word] = 1

        for j in range(len(self.projects)):
            a = sorted(dicts[j].keys(), key=dicts[j].get, reverse=True)
            sets[j] = set(a[:150])
            all_words = all_words | sets[j]
        return list(all_words)

    def samples_to_vec(self, samples, words):
        """
        converts a samples represented by string (few lines of code) into
        a matrix X of floats
        :param samples: array of samples (strings)
        :return: numpy array (shape = num_of_samples * num_of_features)
        """
        del_len, words_len = len(self.delimiters), len(words)
        X = np.zeros((len(samples), words_len + del_len + 2))
        for n in range(len(samples)):
            vec = np.zeros(words_len + del_len + 2)
            vec[0] = len(min(samples[n].split('\n'), key=len))
            vec[1] = len(max(samples[n].split('\n'), key=len))
            for i in range(del_len):
                vec[i + 2] = samples[n].count(self.delimiters[i])
            sample = self.split_sample(samples[n])
            for j in range(words_len):
                vec[del_len + 2 + j] = sample.count(words[j])
            X[n] = vec
        return X

    def train_model(self):
        """
        trained a RandomForestClassifier on the given code files of the projects and print the score
        on the test set.
        """
        X, y = self.parse_files()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        words_features = self.build_words_features(X_train, y_train)
        X_train, x_test = self.samples_to_vec(X_train, words_features), self.samples_to_vec(X_test, words_features)
        trained_model = RandomForestClassifier(max_depth=102, n_estimators=120).fit(X_train, y_train)
        joblib.dump(trained_model, "trained_model.sav")

        # test model
        print(trained_model.score(x_test, y_test))

