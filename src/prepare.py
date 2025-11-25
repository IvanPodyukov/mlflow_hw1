# src/prepare.py
"""
Подготовка данных:
 - читает data/raw/titanic.csv
 - простая очистка и кодирование
 - разбиение на train/test
 - сохраняет в data/processed/train.csv и data/processed/test.csv
"""
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

ROOT = Path(__file__).resolve().parents[1]

def load_params():
    p = ROOT / "params.yaml"
    with open(p, "r") as f:
        return yaml.safe_load(f)

def preprocess(df):
    # Простая обработка: удалить лишние колонки, заполнить пропуски, закодировать Sex/Embarked
    df = df.copy()
    # Удаляем колонки, которые мало информативны/много уникальных значений
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)

    # Fill Age and Fare
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Embarked fill with mode
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Sex encoding
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # One-hot Embarked
    if "Embarked" in df.columns:
        dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=True)
        df = pd.concat([df.drop(columns=["Embarked"]), dummies], axis=1)

    # If 'Survived' is present, ensure integer
    if "Survived" in df.columns:
        df["Survived"] = df["Survived"].astype(int)

    return df

def main():
    params = load_params()
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    raw_path = ROOT / "data" / "raw" / "titanic.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Place titanic.csv there.")

    df = pd.read_csv(raw_path)
    df = preprocess(df)

    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Survived"] if "Survived" in df.columns else None
    )

    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    print(f"Saved processed data to {processed_dir}")

if __name__ == "__main__":
    main()
