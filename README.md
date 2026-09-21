# ❤️ Heart Disease Prediction & Explainable AI

A machine learning web application that predicts the possibility of heart disease from patient health information. The project uses a Random Forest classifier for prediction and SHAP (SHapley Additive exPlanations) to explain why the model made a particular prediction.

The complete project includes data understanding, preprocessing, machine learning model development, model evaluation, explainability, a FastAPI backend, and an interactive web interface.



## 📌 Project Overview

Heart disease is an important health problem, and machine learning can be used to identify patterns in patient data that may be associated with heart disease.

The main goal of this project is to build an end-to-end machine learning application where a user can enter patient information and receive a prediction from a trained machine learning model.

The project does not stop at prediction. Since machine learning models can sometimes behave like a "black box", SHAP is used to provide an explanation of the important factors that influenced each individual prediction.

The application therefore follows this flow:


Patient Information
        ↓
Input Validation
        ↓
Data Preprocessing
        ↓
Random Forest Model
        ↓
Prediction + Probability
        ↓
SHAP Explainability
        ↓
Result shown on Web Interface
