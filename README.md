# mlops_hw1_Подюков_Иван — MLOps HW1 (DVC + MLflow) — Titanic

## Цель
Воспроизвести минимальный MLOps-пайплайн:
- версионирование данных через DVC
- автоматизация подготовки и обучения (dvc repro)
- логирование экспериментов в MLflow

Задача: предсказать `Survived` для пассажиров Titanic.

## Что в репозитории
- `src/prepare.py` — подготовка данных (сплит train/test)
- `src/train.py` — обучение модели RandomForestClassifier, логирование в MLflow. Логируются: параметры модели (n_estimators, max_depth, random_state), метрика accuracy
- `dvc.yaml` — DVC-пайплайн (prepare -> train)
- `params.yaml` — гиперпараметры
- Данные хранятся через DVC (`data/raw/titanic.csv` -> `dvc add`)


## Быстрый старт
```bash
# 1. клонировать репозиторий
git clone https://github.com/IvanPodyukov/mlflow_hw1.git
cd mlflow_hw1

# 2. создать виртуальное окружение и установить зависимости
python -m venv venv
source venv/bin/activate   # или venv\Scripts\activate на Windows
pip install -r requirements.txt

# 3. подтянуть данные
dvc pull

# 4. воспроизвести пайплайн
dvc repro

# 5. запустить MLflow UI (в отдельном терминале)
mlflow ui --backend-store-uri sqlite:///mlflow.db
# открыть http://127.0.0.1:5000