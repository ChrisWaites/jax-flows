import numpy as np
import pandas as pd
from .. import utils

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'salary'
]

datatypes = [
    ('age', 'positive int'),
    ('workclass', 'categorical'),
    ('education-num', 'categorical'),
    ('marital-status', 'categorical'),
    ('occupation', 'categorical'),
    ('relationship', 'categorical'),
    ('race', 'categorical'),
    ('sex', 'categorical binary'),
    ('capital-gain', 'positive float'),
    ('capital-loss', 'positive float'),
    ('hours-per-week', 'positive int'),
    ('native-country', 'categorical'),
    ('salary', 'categorical binary'),
]

processor = utils.Processor(datatypes)


@utils.constant_seed
def get_datasets(val_prop=0.1, test_prop=0.1):
    train = pd.read_csv('datasets/adult/adult.data', names=column_names)
    test = pd.read_csv('datasets/adult/adult.test', names=column_names)

    df = pd.concat([train, test])
    df = df.drop(columns=['education', 'fnlwgt'])

    for column, datatype in datatypes:
        if 'categorical' in datatype:
            df[column] = df[column].astype('category').cat.codes

    X = processor.fit_transform(df.values.astype('float32'))

    val_start = int(X.shape[0] * (1 - (val_prop + test_prop)))
    val_end = int(X.shape[0] * (1 - test_prop))

    return X, X[:val_start], X[val_start:val_end], X[val_end:]


def postprocess(X):
    return processor.postprocess(X)
