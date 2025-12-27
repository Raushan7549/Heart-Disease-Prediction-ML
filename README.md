#Heart Disease Prediction Using Machine Learning
##📌 Project Overview

Heart disease is one of the leading causes of mortality worldwide. Early detection plays a crucial role in reducing risk and improving patient outcomes.
This project focuses on building an end-to-end machine learning pipeline to predict the presence of heart disease using real-world health and lifestyle data from the CDC Heart 2020 dataset.

The project applies data preprocessing, exploratory data analysis, class imbalance handling, and supervised learning models to perform accurate and reliable heart disease prediction.

##📊 Dataset Description

Source: Centers for Disease Control and Prevention (CDC) – Heart Disease 2020 Dataset

Type: Structured, tabular healthcare dataset

Records: ~319,000 patient records

Target Variable: HeartDisease

0 → No Heart Disease

1 → Heart Disease

The dataset contains demographic, medical, and lifestyle attributes such as age category, BMI, smoking habits, physical activity, sleep time, and general health indicators.

##🧠 Problem Statement

The dataset is highly imbalanced, with significantly fewer heart disease cases compared to non-heart disease cases.
This imbalance can lead to biased predictions if not handled properly.
The goal is to build predictive models that perform well on both majority and minority classes, especially identifying patients at risk.

##⚙️ Technologies & Tools

Programming Language: Python

Libraries:

Pandas, NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Matplotlib, Seaborn

##🔍 Methodology

Data loading and initial exploration

Label encoding of categorical features

Feature scaling using StandardScaler

Handling class imbalance using SMOTE

Exploratory Data Analysis (EDA) with visualizations

Model training and testing using supervised learning algorithms

Performance evaluation using standard classification metrics

##🤖 Machine Learning Models

Logistic Regression

Support Vector Machine (SVM – LinearSVC)

Both models were trained and evaluated to compare overall accuracy and minority-class detection capability.

##📈 Model Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

These metrics provide a comprehensive understanding of model performance, especially in imbalanced medical datasets where accuracy alone can be misleading.

##✅ Results & Insights

SVM achieved higher overall accuracy (~91%), performing well on the majority class.

Logistic Regression with SMOTE improved recall for heart disease cases, making it more suitable for early risk detection.

Key risk factors influencing heart disease prediction include age, BMI, physical activity, sleep duration, and general health condition.

##📌 Conclusion

This project demonstrates how machine learning can be effectively applied to real-world healthcare data for predictive analysis.
Handling class imbalance using SMOTE significantly improves model reliability for medical predictions.
The study highlights the importance of selecting appropriate evaluation metrics and models when working with health-related datasets.

##🚀 Future Scope

Integrating ensemble models such as Random Forest or XGBoost

Adding explainable AI techniques (SHAP/LIME) for interpretability

Deploying the model as a web or dashboard-based application

Expanding the dataset to improve generalization
