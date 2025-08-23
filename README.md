# Anemia Detection ML Service

## Project Overview
This project provides a machine learning-based service for detecting anemia from blood test data. It leverages multiple classification models to predict anemia status and exposes the functionality via a FastAPI web service.

<img width="911" height="240" alt="image" src="https://github.com/user-attachments/assets/777e16a9-1b45-437d-bffb-680335963b85" />
<img width="429" height="681" alt="image" src="https://github.com/user-attachments/assets/802a95b6-a27e-44a9-acd6-61a296b7849c" />


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

## Cleaning data
<img width="418" height="144" alt="image" src="https://github.com/user-attachments/assets/7677ba35-7824-43e8-9acf-309651e3cfbc" />

## Dataset overview
<img width="438" height="451" alt="image" src="https://github.com/user-attachments/assets/0fdcab63-38f5-4a1c-ac5e-c30de9fd2949" />

## Class distribution
<img width="1313" height="674" alt="image" src="https://github.com/user-attachments/assets/c2e9fe25-34d7-4e06-826c-3425fe3e4b93" />

## Feature analysis
<img width="592" height="536" alt="image" src="https://github.com/user-attachments/assets/746a22b1-ddff-4da7-8244-ab03b4bf058e" />
<img width="936" height="626" alt="image" src="https://github.com/user-attachments/assets/9ef64421-6cc9-4f6b-b9f3-7cb0d7d007ec" />

## Machine learning models performance comparison
<img width="1156" height="799" alt="image" src="https://github.com/user-attachments/assets/936a1d95-c5a0-482f-b2ed-a082c2372811" />

## Credits
- Datasets: `01-biswaranjanrao-anemia-dataset.csv`, `02-ragishehab-anemia-dataset.csv`
- - https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset
- - https://www.kaggle.com/datasets/ragishehab/anemia-data-set
- ML models: Decision Tree, Random Forest, SVM, Logistic Regression, KNN, Naive Bayes, Gradient Boosting

---
