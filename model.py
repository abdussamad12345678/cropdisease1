import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

MODEL_FILE = "model.pkl"

def train_model():
    data = pd.read_csv("data.csv")

    X = data[["temperature", "humidity", "rainfall"]]
    y = data["disease"]

    model = RandomForestClassifier()
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model

def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_model()

    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

def predict_risk(model, temp, humidity, rainfall):
    return model.predict_proba([[temp, humidity, rainfall]])[0][1]
