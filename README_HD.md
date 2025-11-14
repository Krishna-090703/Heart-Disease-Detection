# Heart Disease Prediction using Machine Learning

## Overview
This project develops a machine learning pipeline to predict the presence of heart disease using the UCI Heart Disease dataset. The workflow includes data cleaning, preprocessing, exploratory analysis, model training, hyperparameter tuning, and evaluation of ensemble learning methods. The primary objective is to build a reliable binary classification model that identifies whether a patient is likely to have heart disease.

## Dataset
The dataset used is the Heart Disease UCI dataset, containing demographic, clinical, and diagnostic attributes.
Key features include age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, maximum heart rate, exercise-induced angina, oldpeak, slope, number of major vessels, and thalassemia type.

The target variable is converted into a binary class:
- 0: No heart disease
- 1: Heart disease present

## Project Structure
The workflow covers the following steps:
1. Data import and inspection
2. Handling missing values
3. Feature engineering and target transformation
4. Exploratory data analysis
5. Train-test splitting
6. Preprocessing using a pipeline with:
   - Median imputation for numerical variables
   - Most-frequent imputation and one-hot encoding for categorical variables
7. Model training using:
   - Random Forest Classifier
   - XGBoost Classifier
8. Hyperparameter tuning with GridSearchCV
9. Model evaluation using accuracy, precision, recall, F1 score, and confusion matrix
10. Feature importance interpretation using SHAP

## Models Used
- Random Forest Classifier
- XGBoost Classifier

Both models were trained within a preprocessing pipeline for consistent and reproducible transformations.

## Results
Random Forest and XGBoost were evaluated on test data.
The best-performing model achieved approximately 85 percent accuracy.
SHAP analysis was used to identify the most influential features contributing to predictions.

## Key Insights
- Chest pain type, age, cholesterol, resting blood pressure, thalassemia category, and maximum heart rate were among the most important predictors.
- Ensemble models provided strong performance due to their ability to handle non-linear relationships and mixed data types.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SHAP

## How to Run
1. Install required packages
2. Load the dataset into your working directory
3. Run the notebook or script containing the pipeline and model training steps
4. Review evaluation metrics and SHAP plots for model interpretation

## Conclusion
This project demonstrates the application of ensemble machine learning models for healthcare risk prediction. By combining preprocessing pipelines, hyperparameter tuning, and interpretability methods, it delivers a complete and transparent workflow suitable for real-world classification tasks.
