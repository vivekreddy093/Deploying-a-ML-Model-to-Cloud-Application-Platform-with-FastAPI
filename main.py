# put the code for your API here
import pandas as pd
import pickle

from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference

with open("model/model.pkl", "rb") as fin:
    model = pickle.load(fin)

with open("model/label_binarizer.pkl", "rb") as fin:
    lb = pickle.load(fin)

with open("model/encoder.pkl", "rb") as fin:
    encoder = pickle.load(fin)


# declare the data object with its components and their type
# [see https://archive.ics.uci.edu/ml/datasets/census+income]
class Data(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="United-States")


# instantiate the app
app = FastAPI()


# define a GET on the specified endpoint
@app.get("/")
def greet():
    return "Hello World!"


# use POST action to send data to the server
@app.post("/predict")
def predict(data: Data):
    # create DataFrame
    df_data = pd.DataFrame.from_dict([data.model_dump(by_alias=True)])
    df_data.columns = df_data.columns.str.replace("_", "-")

    # categorial features
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

    # prepare data for inference
    X, _, _, _ = process_data(
        df_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # run model
    prediction = inference(model, X)
    prediction = lb.inverse_transform(prediction)[0]

    return {"prediction": prediction}
