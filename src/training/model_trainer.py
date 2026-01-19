import joblib
import json
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


class ModelTrainer:

    def train(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            "LogisticRegression": {
                "model": LogisticRegression(max_iter=1000),
                "params": {
                    "C": [0.01, 0.1, 1, 10]
                }
            },
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [200, 400],
                    "max_depth": [6, 10, None],
                    "min_samples_split": [2, 5]
                }
            },
            "GradientBoosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [200, 300],
                    "learning_rate": [0.03, 0.05],
                    "max_depth": [3, 5]
                }
            },
            "XGBoost": {
                "model": XGBClassifier(
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=42
                ),
                "params": {
                    "n_estimators": [300, 500],
                    "max_depth": [4, 6],
                    "learning_rate": [0.03, 0.05],
                    "subsample": [0.8, 1.0]
                }
            }
        }

        best_model = None
        best_score = 0

        for name, cfg in models.items():
            print(f"\n🔍 Tuning {name}")

            grid = GridSearchCV(
                cfg["model"],
                cfg["params"],
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            model = grid.best_estimator_
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            print(f"✅ {name} Best Accuracy: {acc:.4f}")
            print(f"⭐ Best Params: {grid.best_params_}")

            if acc > best_score:
                best_score = acc
                best_model = model

        os.makedirs("models", exist_ok=True)

        # SAVE BEST MODEL
        joblib.dump(best_model, "models/best_model.pkl")

        # SAVE FEATURE ORDER
        with open("models/feature_names.json", "w") as f:
            json.dump(list(X.columns), f)

        print(f"\n🏆 FINAL BEST MODEL ACCURACY: {best_score:.4f}")
        return best_score
