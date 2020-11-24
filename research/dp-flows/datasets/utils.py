import jax.numpy as np
import numpy as onp
from jax import random, jit, grad
from jax.experimental import stax, optimizers
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer
import itertools


def constant_seed(f):
    def g(*args, **kwargs):
        state = onp.random.get_state()
        onp.random.seed(0)
        ret = f(*args, **kwargs)
        onp.random.set_state(state)
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

        return onp.concatenate(preprocessed_cols, axis=1)


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

        return onp.concatenate(postprocessed_cols, axis=1)


class AE():
    def __init__(self, dim):
        encoder_rng, decoder_rng = random.split(random.PRNGKey(0), 2)

        encoder_init, self.encoder_apply = stax.serial(
            stax.Dense(dim // 2),
            stax.Relu,
            stax.Dense(2),
        )
        _, self.encoder_params = encoder_init(encoder_rng, (dim,))

        decoder_init, self.decoder_apply = stax.serial(
            stax.Dense(dim // 2),
            stax.Relu,
            stax.Dense(dim),
        )
        _, self.decoder_params = decoder_init(decoder_rng, (dim,))

    def fit(self, X):
        opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
        opt_state = opt_init((self.encoder_params, self.decoder_params))

        def loss(params, inputs):
            encoder_params, decoder_params = params
            enc = self.encoder_apply(encoder_params, X)
            dec = self.decoder_apply(decoder_params, X)
            return np.square(inputs - dec).sum() + 1e-3 * np.abs(params).sum()

        @jit
        def step(i, opt_state, inputs):
            params = get_params(opt_state)
            gradient = grad(loss)(params, inputs)
            return opt_update(i, gradient, opt_state)

        print('Training autoencoder...')

        batch_size, itercount = 32, itertools.count()
        key = random.PRNGKey(0)
        for epoch in range(5):
            temp_key, key = random.split(key)
            X = random.permutation(temp_key, X)
            for batch_index in range(0, X.shape[0], batch_size):
                opt_state = step(next(itercount), opt_state, X[batch_index:batch_index+batch_size])

        self.encoder_params, self.decoder_params = get_params(opt_state)

    def transform(X):
        return self.encoder_apply(self.encoder_params, X)

    def inverse_transform(X):
        return self.decoder_apply(self.decoder_params, X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
