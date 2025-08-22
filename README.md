# Anemia Detection ML Service

## Project Overview
This project provides a machine learning-based service for detecting anemia from blood test data. It leverages multiple classification models to predict anemia status and exposes the functionality via a FastAPI web service.

## Workflow
1. **Data Analysis**
   - Initial data exploration, preprocessing, and model comparison were performed in the Jupyter Notebook: [`detect-anemia-by-blood-all-models.ipynb`](detect-anemia-by-blood-all-models.ipynb).

2. **Model Training**
   - The script [`train_models.py`](train_models.py) was used to train several machine learning models on the processed datasets. Trained models and scalers are saved in the `models/` directory.

3. **API Implementation**
   - The FastAPI application in the [`app/`](app/) directory loads the trained models and provides endpoints for anemia prediction. The main logic is organized into modules for configuration, model loading, services, and API controllers.

## Directory Structure
- `detect-anemia-by-blood-all-models.ipynb` — Data analysis and model comparison notebook
- `train_models.py` — Model training script
- `app/` — FastAPI application
  - `core/` — Configuration and model loading
  - `models/` — Model definitions
  - `services/` — Business logic
  - `api/v1/` — API endpoints
- `data/` — Source datasets
- `models/` — Saved models and scalers
- `requirements.txt` — Python dependencies

## How to Run
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Jupyter Notebook** for data analysis (optional):
   ```bash
   jupyter notebook detect-anemia-by-blood-all-models.ipynb
   ```
3. **Train models** (if needed):
   ```bash
   python train_models.py
   ```
4. **Start the FastAPI service**:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

## Requirements
All required Python packages are listed in `requirements.txt`.

## Credits
- Datasets: `01-biswaranjanrao-anemia-dataset.csv`, `02-ragishehab-anemia-dataset.csv`
- - https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset
- - https://www.kaggle.com/datasets/ragishehab/anemia-data-set
- ML models: Decision Tree, Random Forest, SVM, Logistic Regression, KNN, Naive Bayes, Gradient Boosting

---
