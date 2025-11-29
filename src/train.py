# src/train.py
"""
Обучение модели и логирование в MLflow.
Сохраняет модель в каталог model/ (model/model.pkl)
"""
import yaml
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
import json

ROOT = Path(__file__).resolve().parents[1]

def load_params():
    p = ROOT / "params.yaml"
    with open(p, "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    model_params = params["model"]
    random_state = params["data"]["random_state"]

    processed_dir = ROOT / "data" / "processed"
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed train/test not found. Run src/prepare.py first (or dvc repro).")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target_col = "Survived"
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    # Создаём модель
    model = RandomForestClassifier(
        n_estimators=model_params.get("n_estimators", 100),
        max_depth=model_params.get("max_depth"),
        random_state=random_state,
        n_jobs=-1
    )

    # Start MLflow run
    mlflow.set_tracking_uri(f"sqlite:///mlflow.db")
    mlflow.set_experiment("titanic_experiment")
    with mlflow.start_run():
        # Логируем параметры
        mlflow.log_param("model_type", "RandomForest")
        for k,v in model_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("random_state", random_state)

        # Обучаем
        model.fit(X_train, y_train)

        # Предсказание + метрики
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        mlflow.log_metric("accuracy", acc)

        # Доп. отчет
        report = classification_report(y_test, preds, output_dict=True)
        # Сохраним отчет как артефакт
        report_path = ROOT / "model" / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))

        # Сохраняем модель
        model_path = ROOT / "model" / "model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        # Дополнительно можно логировать модель через mlflow.sklearn
        mlflow.sklearn.log_model(model, "rf_model")

        print(f"Accuracy: {acc:.4f}")
        print("Model and report saved to model/ and logged to MLflow.")

if __name__ == "__main__":
    main()
