# Housing Price Prediction

Predicting house prices using the Ames Housing dataset (Kaggle: House Prices - Advanced Regression Techniques).

## Goal
Predict `SalePrice` of a house from its features.
The model predicts `log(SalePrice)` to reduce the effect of very expensive houses.

## Dataset
- Source: [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- `train.csv` has the prices (used for training and validation)
- `test.csv` has no prices (used only for predictions)

## Steps
1. EDA (explore the data)
2. Preprocessing (missing values, encoding, scaling)
3. Feature engineering (TotalSF, TotalBath, HouseAge, RemodAge)
4. Model comparison (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)
5. Tuning with RandomizedSearchCV
6. Final evaluation and 5-fold cross-validation

## Final Model
Tuned Gradient Boosting Regressor

| Metric | Result |
|---|---|
| R² | 0.906 |
| RMSE (log) | 0.132 |
| MAE (log) | 0.085 |
| RMSE (dollars) | about $28,000 |
| Mean CV RMSE (5-fold) | 0.125 |

## Key Findings
- Most important features: `TotalSF` and `OverallQual`
- Results are stable across the 5 folds
- The model has larger errors on a few cheap or unusual houses

## Limitations
- Data comes from one city (Ames, Iowa)
- Big errors on outlier houses

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Google Colab

## How to Run
1. Download the dataset from Kaggle
2. Open `Housing Price Prediction.ipynb` in Google Colab
3. Upload `train.csv` and `test.csv`
4. Run all cells
