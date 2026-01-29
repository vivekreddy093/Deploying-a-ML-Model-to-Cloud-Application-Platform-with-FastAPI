import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from starter.ml.data import process_data
from starter.ml.model import train_model, inference


# fixtures
@pytest.fixture
def data():
    return pd.read_csv("data/clean_census.csv")


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def split_data(data):
    train, test = train_test_split(data, test_size=0.20)
    return train, test


# tests
def test_train_data(split_data, cat_features):
    train, _ = split_data
    Xtrain, ytrain, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert Xtrain.shape[0] == ytrain.shape[0]
    assert encoder is not None
    assert lb is not None


def test_inference_data(split_data, cat_features):
    train, test = split_data
    _, _, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    Xtest, ytest, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # assert if the number of samples is equal to the number of labels
    assert Xtest.shape[0] == ytest.shape[0]


def test_artifacts_persistence():
    model_file = "model/model.pkl"
    lb_file = "model/label_binarizer.pkl"

    with open(model_file, "rb") as fin:
        model = pickle.load(fin)

    with open(lb_file, "rb") as fin:
        lb = pickle.load(fin)

    # assert if they are not none
    assert model is not None
    assert lb is not None


def test_inference(split_data, cat_features):
    train, test = split_data

    Xtrain, ytrain, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    Xtest, ytest, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    with open("model/model.pkl", "rb") as fin:
        model = pickle.load(fin)

    model = train_model(Xtrain, ytrain)
    ypred = inference(model, Xtest)

    # assert if the number of predictions is equal to #test samples
    assert ypred.shape[0] == ytest.shape[0]
