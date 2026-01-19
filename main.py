import os
import pandas as pd
import yaml
from src.preprocessing.data_preprocessing import DataPreprocessor
from src.training.model_trainer import ModelTrainer

# Load config
with open("src/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = config["data"]["raw_path"]

print("🔍 Looking for dataset at:")
print(os.path.abspath(data_path))

# Hard check
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"\n❌ Dataset NOT FOUND\n"
        f"Expected at: {os.path.abspath(data_path)}\n"
        f"👉 Fix path OR move dataset"
    )

# Load dataset
df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully")

# Preprocessing
preprocessor = DataPreprocessor(
    target=config["data"]["target_column"],
    drop_columns=config["data"]["drop_columns"]
)

X, y = preprocessor.preprocess(df)

# Training
trainer = ModelTrainer()
score = trainer.train(X, y)

print(f"✅ Training complete | Accuracy: {score:.2f}")
