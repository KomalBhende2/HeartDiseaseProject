"""
CardioSense AI - Preprocessing and Model Inference Utilities
Ensures strict feature alignment and preprocessing fidelity with the trained SVM model.
Includes clinical report generation and batch processing utilities.
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import joblib


class MissingArtifactError(FileNotFoundError):
    """Raised when any required model artifact (.pkl) is missing."""
    pass


REQUIRED_ARTIFACTS = {
    "model": "SVM_heart.pkl",
    "scaler": "scaler.pkl",
    "columns": "columns.pkl"
}


def check_artifacts_availability(base_dir: str = ".") -> Tuple[bool, List[str]]:
    """
    Verifies that all three required pickle files exist in the specified directory.
    Returns (True, []) if all exist, or (False, list_of_missing_files) if any are missing.
    """
    missing = []
    for key, filename in REQUIRED_ARTIFACTS.items():
        path = os.path.join(base_dir, filename)
        if not os.path.isfile(path):
            missing.append(filename)
    return (len(missing) == 0, missing)


def load_artifacts(base_dir: str = ".") -> Tuple[Any, Any, List[str]]:
    """
    Loads the trained SVM model, StandardScaler, and expected columns list.
    
    Raises:
        MissingArtifactError: If any artifact file cannot be found.
        Exception: If loading or deserialization fails.
    """
    all_present, missing = check_artifacts_availability(base_dir)
    if not all_present:
        raise MissingArtifactError(
            f"Missing required model artifact(s): {', '.join(missing)}. "
            f"Please verify these files exist in '{os.path.abspath(base_dir)}'."
        )

    model_path = os.path.join(base_dir, REQUIRED_ARTIFACTS["model"])
    scaler_path = os.path.join(base_dir, REQUIRED_ARTIFACTS["scaler"])
    columns_path = os.path.join(base_dir, REQUIRED_ARTIFACTS["columns"])

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    expected_columns = joblib.load(columns_path)

    # Convert expected_columns to standard list if numpy array
    if hasattr(expected_columns, "tolist"):
        expected_columns = expected_columns.tolist()

    return model, scaler, expected_columns


def validate_patient_inputs(data: Dict[str, Any]) -> List[str]:
    """
    Validates clinical and physiological inputs against standard medical plausibility ranges.
    Returns a list of validation error strings. If valid, returns an empty list.
    """
    errors = []

    # Age: 18 - 100
    age = data.get("Age")
    if age is None or not (18 <= age <= 100):
        errors.append("Age must be between 18 and 100 years.")

    # Sex: 'M' or 'F'
    sex = data.get("Sex")
    if sex not in ["M", "F"]:
        errors.append("Sex must be 'M' (Male) or 'F' (Female).")

    # RestingBP: 80 - 200 mm Hg
    resting_bp = data.get("RestingBP")
    if resting_bp is None or not (80 <= resting_bp <= 200):
        errors.append("Resting Blood Pressure must be between 80 and 200 mm Hg.")

    # Cholesterol: 100 - 600 mg/dL
    cholesterol = data.get("Cholesterol")
    if cholesterol is None or not (100 <= cholesterol <= 600):
        errors.append("Serum Cholesterol must be between 100 and 600 mg/dL.")

    # FastingBS: 0 or 1
    fasting_bs = data.get("FastingBS")
    if fasting_bs not in [0, 1]:
        errors.append("Fasting Blood Sugar must be 0 (<= 120 mg/dL) or 1 (> 120 mg/dL).")

    # MaxHR: 60 - 220 bpm
    max_hr = data.get("MaxHR")
    if max_hr is None or not (60 <= max_hr <= 220):
        errors.append("Maximum Heart Rate must be between 60 and 220 bpm.")

    # Oldpeak: 0.0 - 6.5 mm (non-negative)
    oldpeak = data.get("Oldpeak")
    if oldpeak is None or oldpeak < 0.0:
        errors.append("Oldpeak (ST depression) cannot be negative.")
    elif oldpeak > 6.5:
        errors.append("Oldpeak value is physiologically improbable (> 6.5 mm).")

    # ChestPainType: 'TA', 'ATA', 'NAP', 'ASY'
    chest_pain = data.get("ChestPainType")
    if chest_pain not in ["TA", "ATA", "NAP", "ASY"]:
        errors.append("Chest Pain Type must be one of: TA, ATA, NAP, ASY.")

    # RestingECG: 'Normal', 'ST', 'LVH'
    resting_ecg = data.get("RestingECG")
    if resting_ecg not in ["Normal", "ST", "LVH"]:
        errors.append("Resting ECG must be one of: Normal, ST, LVH.")

    # ExerciseAngina: 'Y' or 'N'
    exercise_angina = data.get("ExerciseAngina")
    if exercise_angina not in ["Y", "N"]:
        errors.append("Exercise Angina must be 'Y' (Yes) or 'N' (No).")

    # ST_Slope: 'Up', 'Flat', 'Down'
    st_slope = data.get("ST_Slope")
    if st_slope not in ["Up", "Flat", "Down"]:
        errors.append("ST Slope must be one of: Up, Flat, Down.")

    return errors


def prepare_features(patient_data: Dict[str, Any], expected_columns: List[str]) -> pd.DataFrame:
    """
    Constructs an input DataFrame aligned identically with the features and ordering
    expected by the trained StandardScaler and SVM model.
    """
    feature_dict: Dict[str, float] = {col: 0.0 for col in expected_columns}

    numeric_keys = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
    for key in numeric_keys:
        if key in expected_columns and key in patient_data:
            feature_dict[key] = float(patient_data[key])

    cat_mappings = [
        ("Sex", patient_data.get("Sex")),
        ("ChestPainType", patient_data.get("ChestPainType")),
        ("RestingECG", patient_data.get("RestingECG")),
        ("ExerciseAngina", patient_data.get("ExerciseAngina")),
        ("ST_Slope", patient_data.get("ST_Slope"))
    ]

    for prefix, value in cat_mappings:
        if value is not None:
            dummy_col = f"{prefix}_{value}"
            if dummy_col in feature_dict:
                feature_dict[dummy_col] = 1.0

    df = pd.DataFrame([feature_dict])
    df = df[expected_columns]
    return df


def predict_heart_disease(
    model: Any, 
    scaler: Any, 
    input_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Executes the trained prediction pipeline:
    input_df -> scaler.transform() -> model.predict() & model.predict_proba()
    """
    scaled_input = scaler.transform(input_df)
    pred_class = int(model.predict(scaled_input)[0])

    has_prob = hasattr(model, "predict_proba")
    probability: Optional[float] = None
    if has_prob:
        try:
            proba_arr = model.predict_proba(scaled_input)[0]
            probability = float(proba_arr[1])
        except Exception:
            has_prob = False
            probability = None

    label = "Elevated Risk Detected" if pred_class == 1 else "Lower Risk Detected"

    return {
        "prediction": pred_class,
        "prediction_label": label,
        "probability": probability,
        "has_probability": has_prob,
        "scaled_features": scaled_input
    }


def generate_clinical_report(patient_data: Dict[str, Any], result: Dict[str, Any]) -> str:
    """
    Generates a structured clinical summary report formatted for export/download.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prob_str = f"{result['probability'] * 100:.1f}%" if result.get("probability") is not None else "N/A"
    
    report = f"""================================================================================
                    CARDIOSENSE AI - CLINICAL ASSESSMENT REPORT
================================================================================
Assessment Timestamp: {timestamp}
Classifier:           Support Vector Machine (RBF Kernel)
Model Accuracy:       88.04% | F1 Score: 89.62%
--------------------------------------------------------------------------------
1. PATIENT BIOMETRICS & VITALS
--------------------------------------------------------------------------------
- Age:                         {patient_data.get('Age')} years
- Biological Sex:              {'Male' if patient_data.get('Sex') == 'M' else 'Female'}
- Resting Blood Pressure:      {patient_data.get('RestingBP')} mm Hg
- Serum Cholesterol:           {patient_data.get('Cholesterol')} mg/dL
- Maximum Heart Rate:          {patient_data.get('MaxHR')} bpm
- Fasting Blood Sugar:         {'> 120 mg/dL' if patient_data.get('FastingBS') == 1 else '<= 120 mg/dL'}
- Oldpeak (ST Depression):     {patient_data.get('Oldpeak')} mm

--------------------------------------------------------------------------------
2. DIAGNOSTIC FINDINGS
--------------------------------------------------------------------------------
- Chest Pain Type:             {patient_data.get('ChestPainType')}
- Resting ECG:                 {patient_data.get('RestingECG')}
- Exercise-Induced Angina:     {'Yes' if patient_data.get('ExerciseAngina') == 'Y' else 'No'}
- Peak Exercise ST Slope:      {patient_data.get('ST_Slope')}

--------------------------------------------------------------------------------
3. RISK STRATIFICATION RESULT
--------------------------------------------------------------------------------
Classification:                {result.get('prediction_label', '').upper()}
Estimated Risk Probability:    {prob_str}
Decision Threshold:            50.0%

--------------------------------------------------------------------------------
4. CLINICAL DISCLAIMER
--------------------------------------------------------------------------------
This report is generated by a machine learning model for educational and research
decision-support analysis only. It DOES NOT constitute a definitive medical diagnosis
and must not replace professional judgment by a qualified healthcare professional.
================================================================================
"""
    return report


def process_batch_csv(
    df: pd.DataFrame, 
    model: Any, 
    scaler: Any, 
    expected_columns: List[str]
) -> Tuple[pd.DataFrame, int, int]:
    """
    Processes an uploaded batch CSV DataFrame, validates columns, prepares features,
    and returns an augmented DataFrame with predictions and risk scores.
    """
    results_list = []
    
    for _, row in df.iterrows():
        patient_row = {
            "Age": row.get("Age"),
            "Sex": str(row.get("Sex")).strip().upper(),
            "RestingBP": row.get("RestingBP"),
            "Cholesterol": row.get("Cholesterol"),
            "FastingBS": int(row.get("FastingBS", 0)),
            "RestingECG": str(row.get("RestingECG")).strip(),
            "MaxHR": row.get("MaxHR"),
            "ExerciseAngina": str(row.get("ExerciseAngina")).strip().upper(),
            "Oldpeak": float(row.get("Oldpeak", 0.0)),
            "ChestPainType": str(row.get("ChestPainType")).strip().upper(),
            "ST_Slope": str(row.get("ST_Slope")).strip()
        }
        
        # Build features strictly using the same encoder
        input_df = prepare_features(patient_row, expected_columns)
        res = predict_heart_disease(model, scaler, input_df)
        
        prob_val = round(res["probability"] * 100, 1) if res.get("probability") is not None else None
        risk_class = "Elevated Risk" if res["prediction"] == 1 else "Lower Risk"
        
        results_list.append({
            "Risk_Prediction": risk_class,
            "Risk_Score_Pct": prob_val
        })

    results_df = pd.DataFrame(results_list)
    augmented_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    elevated_count = (results_df["Risk_Prediction"] == "Elevated Risk").sum()
    lower_count = (results_df["Risk_Prediction"] == "Lower Risk").sum()

    return augmented_df, elevated_count, lower_count


def get_sample_csv_template() -> pd.DataFrame:
    """Returns a ready-to-test sample DataFrame for batch assessment."""
    sample_records = [
        {"Age": 35, "Sex": "F", "RestingBP": 115, "Cholesterol": 170, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 175, "ExerciseAngina": "N", "Oldpeak": 0.0, "ChestPainType": "ATA", "ST_Slope": "Up"},
        {"Age": 59, "Sex": "M", "RestingBP": 145, "Cholesterol": 270, "FastingBS": 1, "RestingECG": "ST", "MaxHR": 125, "ExerciseAngina": "Y", "Oldpeak": 2.2, "ChestPainType": "ASY", "ST_Slope": "Flat"},
        {"Age": 48, "Sex": "M", "RestingBP": 130, "Cholesterol": 220, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 0.8, "ChestPainType": "NAP", "ST_Slope": "Flat"},
        {"Age": 63, "Sex": "F", "RestingBP": 160, "Cholesterol": 290, "FastingBS": 0, "RestingECG": "LVH", "MaxHR": 118, "ExerciseAngina": "Y", "Oldpeak": 2.5, "ChestPainType": "ASY", "ST_Slope": "Flat"},
        {"Age": 28, "Sex": "F", "RestingBP": 110, "Cholesterol": 160, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 185, "ExerciseAngina": "N", "Oldpeak": 0.0, "ChestPainType": "TA", "ST_Slope": "Up"}
    ]
    return pd.DataFrame(sample_records)
