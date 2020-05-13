import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer

def constant_seed(f):
    def g(*args, **kwargs):
        state = np.random.get_state()
        np.random.seed(0)
        ret = f(*args, **kwargs)
        np.random.set_state(state)
        return ret
    return g

class Processor:
    def __init__(self, datatypes):
        self.datatypes = datatypes

    def fit(self, matrix):
        preprocessors, cutoffs = [], []
        for i, (column, datatype) in enumerate(self.datatypes):
            preprocessed_col = matrix[:,i].reshape(-1, 1)

            if 'categorical' in datatype:
                preprocessor = LabelBinarizer()
            else:
                preprocessor = MinMaxScaler()

            preprocessed_col = preprocessor.fit_transform(preprocessed_col)
            cutoffs.append(preprocessed_col.shape[1])
            preprocessors.append(preprocessor)

        self.cutoffs = cutoffs
        self.preprocessors = preprocessors

    def transform(self, matrix):
        preprocessed_cols = []

        for i, (column, datatype) in enumerate(self.datatypes):
            preprocessed_col = matrix[:,i].reshape(-1, 1)
            preprocessed_col = self.preprocessors[i].transform(preprocessed_col)
            preprocessed_cols.append(preprocessed_col)

        return np.concatenate(preprocessed_cols, axis=1)


    def fit_transform(self, matrix):
        self.fit(matrix)
        return self.transform(matrix)

    def inverse_transform(self, matrix):
        postprocessed_cols = []
        j = 0
        for i, (column, datatype) in enumerate(self.datatypes):
            postprocessed_col = self.preprocessors[i].inverse_transform(matrix[:,j:j+self.cutoffs[i]])

            if 'categorical' in datatype:
                postprocessed_col = postprocessed_col.reshape(-1, 1)
            else:
                if 'positive' in datatype:
                    postprocessed_col = postprocessed_col.clip(min=0)

                if 'int' in datatype:
                    postprocessed_col = postprocessed_col.round()

            postprocessed_cols.append(postprocessed_col)

            j += self.cutoffs[i]

        return np.concatenate(postprocessed_cols, axis=1)

