import joblib
import pandas as pd
import numpy as np

def predict_diabetes(patient_data_raw):
    # 1. Load the new assets (Ensure these files are in the same folder)
    loaded_model = joblib.load('diabetes_model_xgb.pkl')
    loaded_scaler = joblib.load('diabetes_scaler.pkl')
    loaded_imputer = joblib.load('diabetes_knn_imputer.pkl')

    # 2. Parse the Input

    feature_names = ['Glucose', 'Insulin', 'BloodPressure', 'Age', 'BMI', 'Pregnancies']
    
    # 3. Create DataFrame
    patient_df = pd.DataFrame([patient_data_raw], columns=feature_names)

    # 4. Handle "0" values (The Fix)
    # The Imputer needs 'NaN' to know a value is missing, not '0'.
    # We apply this to Glucose, Insulin, BP, and BMI. (Pregnancies can be 0).
    cols_to_clean = ['Glucose', 'Insulin', 'BloodPressure', 'BMI']
    patient_df[cols_to_clean] = patient_df[cols_to_clean].replace(0, np.nan)

    # 5. Scale the data (Crucial: Must match training scaling)
    patient_scaled = loaded_scaler.transform(patient_df)

    # 6. Impute Missing Values (Crucial: Uses KNN to fill the NaNs we just created)
    patient_final = loaded_imputer.transform(patient_scaled)

    # 7. Get Probability
    probability = loaded_model.predict_proba(patient_final)[0, 1]

    # 7. Determine Diagnosis & Tone
    # --- LOGIC UPDATE ---
    # We now have two thresholds to create a "Middle Ground" (Prediabetic)
    high_threshold = 0.50  # Above this is Diabetic
    low_threshold = 0.30   # Below this is Healthy; Between 0.30 and 0.50 is Prediabetic
    
    diagnosis = ""
    recommendation = ""
    tone = ""

    if probability >= high_threshold:
        diagnosis = "DIABETIC RISK DETECTED"
        recommendation = "Immediate clinical consultation required. Monitor blood sugar closely."
        tone = "Strict, urgent, and serious"
        emoji = "🚨"
        
    elif probability >= low_threshold:
        diagnosis = "PREDIABETIC / WARNING"
        recommendation = "Lifestyle changes recommended. Re-test in 3 months."
        tone = "Supportive, assertive, and guiding"
        emoji = "⚠️"
        
    else:
        diagnosis = "Likely Healthy"
        recommendation = "Maintain current healthy lifestyle."
        tone = "Cheery, encouraging, and positive"
        emoji = "✅"

    # # 8. Output
    # print(f"\n🔬 Analysis Results:")
    # print(f"-------------------")
    # print(f"Features Used:  {feature_names}")
    # print(f"Risk Score:     {probability:.1%}")
    # print(f"Tone Required:  {tone}")
    # print(f"{emoji} DIAGNOSIS:  {diagnosis}")
    # print(f"Recommendation: {recommendation}")
    
    # Return these if you need to pass them to an LLM API later
    return {
        "risk_score": probability,
        "diagnosis": diagnosis,
        "tone": tone,
        "recommendation": recommendation
    }

