import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:

    def __init__(self, target, drop_columns):
        self.target = target
        self.drop_columns = drop_columns
        self.encoders = {}

    def preprocess(self, df):

        # Drop unused columns
        df = df.drop(columns=self.drop_columns, errors="ignore")

        # Encode categorical features
        for col in df.select_dtypes(include="object"):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        # 🔒 SAVE ENCODERS
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.encoders, "models/encoders.pkl")

        # Split X and y
        X = df.drop(self.target, axis=1)
        y = df[self.target]

        return X, y
