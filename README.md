# 🫀 Heart Disease Prediction App

A machine learning application built with Streamlit to predict the likelihood of heart disease in patients based on various medical attributes.

## Features

- **Interactive Web Interface**: Built with Streamlit for easy user interaction.
- **Real-time Prediction**: Inputs patient data and gets instant predictions.
- **Machine Learning**: Uses a Logistic Regression model trained on heart disease data.

## Installation

1. **Clone the repository** (or download usage files):
   ```bash
   git clone <repository-url>
   cd "HeartDiseasePredict"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have `scikit-learn`, `pandas`, `numpy`, `joblib`, and `streamlit` installed.*

3. **Verify Model Files**:
   Ensure `model.pkl` and `imputer.pkl` are present in the directory.

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your default web browser (typically at `http://localhost:8501`).

## Dataset & Training

The model was trained using the `heart.csv` dataset (not included in this distribution by default).
- `heart.py`: Script used to train the model and save artifacts (`model.pkl`, `imputer.pkl`).

## Troubleshooting

### AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'
If you encounter this error, it involves a version mismatch between the `scikit-learn` version used to train the model and your current environment.
- **Solution**: The included `imputer.pkl` should be patched. If issues persist, try upgrading `scikit-learn` or retraining the model if you have `heart.csv`.

## Technologies

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas
- NumPy
