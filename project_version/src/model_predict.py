# src/models/model_predict.py

import joblib
import pandas as pd


class ModelPredictor:
    def __init__(self, model_path="model_rf.pkl", scaler_path="scaler.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def preprocess(self, df):
        return self.scaler.transform(df)

    def predict(self, df):
        X = self.preprocess(df)
        preds = self.model.predict(X)
        return preds

    def predict_proba(self, df):
        X = self.preprocess(df)
        return self.model.predict_proba(X)