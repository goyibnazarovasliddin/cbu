import os
import pandas as pd

from src.config import (
    CLEAN_DATA_DIR,
    RAW_DATA_DIR,
    MERGED_OUTPUT,
    FINAL_DATASET,
    MODEL_PATH,
    SCALER_PATH,
    TEST_SIZE,
    RANDOM_STATE
)

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineering import FeatureEngineering
from src.model_trainer import ModelTrainer


def main():
    print("\n===== ML PIPELINE STARTED =====")

    # 1. Load + Clean + Merge All Data
    print("\n>> Loading and merging cleaned datasets...")

    loader = DataLoader()

    merged_df = loader.load_and_merge_datasets(
        source=str(RAW_DATA_DIR),
        output_path=str(MERGED_OUTPUT),
        clean=True,
        merge_on="customer_id"
    )

    print("Merged dataset shape:", merged_df.shape)

    # Remove customer_id if exists
    if "customer_id" in merged_df.columns:
        merged_df = merged_df.drop(columns=["customer_id"])

    # 2. Feature Engineering
    print("\n>> Feature engineering...")

    fe = FeatureEngineering()
    df = fe.fill_missing_values(merged_df)
    df = fe.remove_low_corr(merged_df)

    print("After FE shape:", df.shape)

    # 3. Save FE-processed data
    df.to_csv(FINAL_DATASET, index=False)
    print("Processed dataset saved to:", FINAL_DATASET)

    # 4. Train/Test Split
    print("\n>> Train/Test split...")
    trainer = ModelTrainer(test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = trainer.split(df)

    # 5. SMOTE balancing
    print(">> Applying SMOTE balancing...")
    X_train_res, y_train_res = trainer.smote(X_train, y_train)

    # 6. Scaling
    print(">> Scaling numeric features...")
    X_train_scaled, X_test_scaled = fe.scale(X_train_res, X_test)

    # 7. Train Model
    print("\n>> Training RandomForest model...")
    model = trainer.fit(X_train_scaled, y_train_res)

    # 8. Evaluate Model
    print("\n===== MODEL EVALUATION =====")
    evaluation = trainer.evaluate(X_test_scaled, y_test)
    
    print("\nAccuracy:", evaluation["accuracy"])
    print("\nClassification Report:\n", evaluation["report"])

    # 9. Save Model + Scaler
    print("\n>> Saving model and scaler...")
    trainer.save_model(MODEL_PATH)
    fe.save_scaler(SCALER_PATH)

    print("Model saved to:", MODEL_PATH)
    print("Scaler saved to:", SCALER_PATH)

    print("\n===== PIPELINE FINISHED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()