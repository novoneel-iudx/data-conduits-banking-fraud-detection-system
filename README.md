# Data Conduits DA204o Course Project

## Nigerian Banking Fraud Detection System

Fraud detection system for Nigerian banking transactions using ML ensemble + LSTM.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
streamlit run demo_app.py
```

## Results

- F1 Score: 0.8847 
- AUC-ROC: 0.9638 
- False Positive Rate: 0.001%

## Project Structure

```
completed/
├── notebooks/           # All analysis notebooks
├── data/                # Processed data
├── demo_app.py          # Streamlit demo
```

## Notebooks

Run in order:
1. `01_data_preparation_and_eda.ipynb` - Load data, feature engineering, EDA
2. `02a_baseline_models.ipynb` - Train LightGBM, CatBoost, XGBoost
3. `03_ensemble_and_evaluation.ipynb` - Optuna ensemble optimization
4. `04_model_interpretability.ipynb` - SHAP analysis
5. `05_lstm_sequence_features.ipynb` - Train LSTM for embeddings
6. `06_models_with_lstm.ipynb` - Models with LSTM features
7. `07_lstm_ensemble_comparison.ipynb` - Final comparison

## Data

Using NIBSS Fraud Dataset from Kaggle (1M transactions, 0.3% fraud rate).

## Models

- LightGBM, CatBoost, XGBoost with class weights
- LSTM for sequence embeddings (64-dim)
- Optuna-optimized weighted ensemble

## Key Features

61 engineered features including:
- Velocity (tx_count_24h, amount_sum_24h)
- Behavioral (amount_vs_mean_ratio, velocity_score)
- Temporal (hour, day, time gaps)
- Categorical (channel, location, merchant)
