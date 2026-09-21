from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import shap

# Create FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using a trained Random Forest model",
    version="1.0.0"
)


# Serve frontend files
app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)


# Load trained model
model = joblib.load(
    "notebooks/heart_disease_random_forest_model.pkl"
)
# SHAP explainability setup
preprocessor = model.named_steps["preprocessor"]
rf_model = model.named_steps["model"]

explainer = shap.TreeExplainer(rf_model)

feature_names = preprocessor.get_feature_names_out()

# Convert technical feature names to readable names
feature_name_mapping = {
    "num__age": "Age",
    "num__resting_bp_s": "Resting Blood Pressure",
    "num__cholesterol": "Cholesterol",
    "num__max_heart_rate": "Maximum Heart Rate",
    "num__oldpeak": "Oldpeak",

    "cat__sex_0.0": "Sex: Female",
    "cat__sex_1.0": "Sex: Male",

    "cat__chest_pain_type_1.0": "Chest Pain: Typical Angina",
    "cat__chest_pain_type_2.0": "Chest Pain: Atypical Angina",
    "cat__chest_pain_type_3.0": "Chest Pain: Non-anginal Pain",
    "cat__chest_pain_type_4.0": "Chest Pain: Asymptomatic",

    "cat__fasting_blood_sugar_0.0": "Fasting Blood Sugar: Normal",
    "cat__fasting_blood_sugar_1.0": "Fasting Blood Sugar: High",

    "cat__resting_ecg_0.0": "Resting ECG: Normal",
    "cat__resting_ecg_1.0": "Resting ECG: ST-T Abnormality",
    "cat__resting_ecg_2.0": "Resting ECG: LV Hypertrophy",

    "cat__exercise_angina_0.0": "Exercise Angina: No",
    "cat__exercise_angina_1.0": "Exercise Angina: Yes",

    "cat__ST_slope_1.0": "ST Slope: Upsloping",
    "cat__ST_slope_2.0": "ST Slope: Flat",
    "cat__ST_slope_3.0": "ST Slope: Downsloping"
}


# Input data schema
class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    chest_pain_type: int = Field(..., ge=1, le=4)
    resting_bp_s: float = Field(..., ge=50, le=250)
    cholesterol: float = Field(..., ge=50, le=700)
    fasting_blood_sugar: int = Field(..., ge=0, le=1)
    resting_ecg: int = Field(..., ge=0, le=2)
    max_heart_rate: float = Field(..., ge=50, le=250)
    exercise_angina: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=-5, le=10)
    ST_slope: int = Field(..., ge=1, le=3)


# Home endpoint
@app.get("/")
def home():
    return FileResponse("frontend/index.html")


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict")
def predict(patient: PatientData):

    # Convert input to DataFrame
    data = pd.DataFrame([patient.model_dump()])

    # Make prediction
    prediction = model.predict(data)[0]

    # Get probability
    probability = model.predict_proba(data)[0][1]


    # ============================
    # SHAP EXPLANATION
    # ============================

    # Transform patient data
    data_transformed = preprocessor.transform(data)

    # Calculate SHAP values
    shap_values = explainer.shap_values(data_transformed)

    # Get SHAP values for Heart Disease class
    if isinstance(shap_values, list):
        patient_shap = shap_values[1][0]
    else:
        patient_shap = shap_values[0, :, 1]

    # Create explanation DataFrame
    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": patient_shap
    })

    # Calculate absolute contribution
    explanation_df["absolute_value"] = (
        explanation_df["shap_value"].abs()
    )

    # Sort strongest factors first
    explanation_df = explanation_df.sort_values(
        "absolute_value",
        ascending=False
    )

    # Get top 5 factors
    top_explanations = []

    for _, row in explanation_df.head(5).iterrows():

        feature = feature_name_mapping.get(
            row["feature"],
            row["feature"]
        )

        top_explanations.append({
            "feature": feature,
            "contribution": round(
                float(row["shap_value"]),
                4
            ),
            "direction": (
                "increases risk"
                if row["shap_value"] > 0
                else "decreases risk"
            )
        })


    # ============================
    # READABLE RESULT
    # ============================

    if prediction == 1:
        result = "Heart Disease"
    else:
        result = "No Heart Disease"


    # ============================
    # API RESPONSE
    # ============================

    return {
        "prediction": int(prediction),
        "result": result,
        "heart_disease_probability": round(
            float(probability), 4
        ),
        "heart_disease_probability_percent": (
            f"{probability:.2%}"
        ),
        "explanations": top_explanations
    }