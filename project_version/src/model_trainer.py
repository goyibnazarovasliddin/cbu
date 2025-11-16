# src/models/model_trainer.py

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=random_state
        )

    def split(self, df):
        X = df.drop("default", axis=1)
        y = df["default"]
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

    def smote(self, X_train, y_train):
        sm = SMOTE(random_state=self.random_state)
        return sm.fit_resample(X_train, y_train)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=False)
        }

    def save_model(self, path="model_rf.pkl"):
        joblib.dump(self.model, path)

    def load_model(self, path="model_rf.pkl"):
        self.model = joblib.load(path)
        return self.model