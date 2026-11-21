# Car Price Prediction using Machine Learning

## Project Overview

This project predicts the selling price of used cars using machine learning regression algorithms. It analyzes vehicle details such as manufacturing year, present price, kilometres driven, fuel type, transmission, selling type, and previous ownership.

The objective is to build and compare regression models and identify the model that provides the most reliable price predictions.

## Dataset

- Dataset: Car Dekho Used Car Dataset
- Original records: 301
- Records after cleaning: 299
- Features used: 7
- Target variable: `Selling_Price`

### Features

| Feature | Description |
|---|---|
| Year | Manufacturing year |
| Present_Price | Current showroom price |
| Driven_kms | Kilometres driven |
| Fuel_Type | Petrol, Diesel, or CNG |
| Selling_type | Dealer or Individual |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |

`Car_Name` was excluded from the baseline model.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Workflow

1. Dataset loading and inspection
2. Data cleaning and duplicate removal
3. Exploratory Data Analysis (EDA)
4. Data visualization
5. Feature selection
6. Train-test splitting
7. Numerical feature scaling
8. Categorical feature encoding using One-Hot Encoding
9. Model training and hyperparameter tuning
10. Repeated cross-validation
11. Model evaluation and comparison
12. Custom car price prediction
13. Saving the trained model and reports

## Machine Learning Models

Four regression algorithms were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Repeated 5-fold cross-validation with 3 repetitions and randomized hyperparameter search were used for model selection.

## Model Performance

The following results were obtained using an 80:20 train-test split.

| Model | Repeated CV RMSE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Linear Regression | 1.7316 | 2.5245 | 0.7527 |
| Decision Tree | 1.7149 | 2.7755 | 0.7011 |
| Random Forest | 1.5488 | 3.6614 | 0.4799 |
| Gradient Boosting | 1.5172 | 1.9566 | 0.8515 |

### Best Model: Gradient Boosting Regressor

Final test performance:

- MAE: 0.9217 lakh
- RMSE: 1.9566 lakh
- R² Score: 0.8515 (85.15%)

Gradient Boosting achieved the lowest repeated cross-validation RMSE and the strongest test-set performance among the evaluated models.

## Custom Prediction

The project also predicts the selling price of a sample car using the selected model.

Sample input:
- Year: 2015
- Present Price: 5.5 lakh
- Driven Kilometres: 30,000
- Fuel Type: Petrol
- Selling Type: Dealer
- Transmission: Manual
- Previous Owners: 0

Predicted selling price: **3.90 lakh**

This is an estimated prediction, not a guaranteed market value.

## Visualizations

The project generates the following visualizations:

1. Selling price distribution
2. Present price vs selling price
3. Selling price by fuel type
4. Selling price by transmission
5. Vehicle year vs selling price
6. Correlation heatmap
7. Actual vs predicted prices
8. Residual analysis
9. Feature importance (when supported by the selected model)

## Project Structure

```text
Task3_Car_Price_Prediction/
│
├── Dataset/
│   └── car data.csv
│
├── outputs/
│   ├── visualizations/
│   ├── reports/
│   ├── cleaned_car_data.csv
│   └── best_car_price_model.pkl
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Execution

Clone the repository:

```bash
git clone https://github.com/sagarbhovi/CodeAlpha_Car_Price_Prediction.git
```

Navigate to the project folder:

```bash
cd CodeAlpha_Car_Price_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Conclusion

This project demonstrates a complete machine learning regression workflow, from data preprocessing and exploratory analysis to model tuning and evaluation.

Gradient Boosting achieved the best results among the four evaluated algorithms, with an R² score of 85.15% on the test dataset.

The project provided practical experience in regression, feature engineering, cross-validation, hyperparameter tuning, and model evaluation.

## Internship

CodeAlpha Data Science Internship – Task 3: Car Price Prediction