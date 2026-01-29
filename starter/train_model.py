# script to train machine learning model.
# add the necessary imports for the starter code.
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


# add code to load in the data.
data = pd.read_csv("data/clean_census.csv")

# optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

Xtrain, ytrain, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# proces the test data with the process_data function.
Xtest, ytest, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# train and save a model.
model = train_model(Xtrain, ytrain)

with open("model/model.pkl", "wb") as fin:
    pickle.dump(model, fin)

with open("model/label_binarizer.pkl", "wb") as fin:
    pickle.dump(lb, fin)

with open("model/encoder.pkl", "wb") as fin:
    pickle.dump(encoder, fin)

# make predictions on the test data.
ypred = inference(model, Xtest)

# compute the model metrics.
precision, recall, fbeta = compute_model_metrics(ytest, ypred)

# print the metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {fbeta:.2f}")

# computing performance on model slices
with open("model/slice_output.txt", "w") as f:
    # iterate over the categorical features
    for feature in cat_features:
        # slice the data
        for value in test[feature].unique():
            mask = test[feature] == value
            test_slice = test[mask].copy()

            # process the data
            Xslice, yslice, _, _ = process_data(
                test_slice,
                categorical_features=cat_features,
                label="salary", training=False,
                encoder=encoder,
                lb=lb
            )

            # make predictions
            ypred = inference(model, Xslice)

            # compute the metrics
            precision, recall, fbeta = compute_model_metrics(yslice, ypred)

            # print the metrics
            print(f"Feature: {feature}, Value: {value}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1: {fbeta:.2f}")

            f.write(f"Feature: {feature}, Value: {value}\n")
            f.write(f"Precision: {precision:.2f}\n")
            f.write(f"Recall: {recall:.2f}\n")
            f.write(f"F1: {fbeta:.2f}\n")
            # add a separator
            f.write("-" * 20 + "\n")
