"""
CardioSense AI - Advanced Heart Disease Prediction Dashboard
A modern clinical machine learning application powered by Support Vector Machines (SVM).
"""

import os
import sys
from datetime import datetime

# Guarantee project directory and utils directory are in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(PROJECT_DIR, "utils")
for path in [PROJECT_DIR, UTILS_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Preprocessing & inference utilities
try:
    from utils.preprocessing import (
        load_artifacts,
        check_artifacts_availability,
        validate_patient_inputs,
        prepare_features,
        predict_heart_disease,
        generate_clinical_report,
        process_batch_csv,
        get_sample_csv_template,
        MissingArtifactError
    )
except (ImportError, ModuleNotFoundError):
    from preprocessing import (
        load_artifacts,
        check_artifacts_availability,
        validate_patient_inputs,
        prepare_features,
        predict_heart_disease,
        generate_clinical_report,
        process_batch_csv,
        get_sample_csv_template,
        MissingArtifactError
    )

# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioSense AI | Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CSS Styling Loader
# -----------------------------------------------------------------------------
def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

# -----------------------------------------------------------------------------
# Cached Artifact Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained machine learning models...")
def get_cached_artifacts():
    """Loads and caches the trained SVM model, StandardScaler, and feature schema."""
    base_dir = os.path.dirname(__file__)
    return load_artifacts(base_dir)

artifacts_loaded = False
load_error_message = None
model = None
scaler = None
expected_columns = None

try:
    model, scaler, expected_columns = get_cached_artifacts()
    artifacts_loaded = True
except MissingArtifactError as mae:
    load_error_message = str(mae)
except Exception as e:
    load_error_message = f"An unexpected error occurred while loading model artifacts: {str(e)}"

# -----------------------------------------------------------------------------
# Sidebar Navigation & System Diagnostics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 2rem;">❤️</span>
            <div>
                <h3 style="margin: 0; font-weight: 800; color: #0284c7; font-size: 1.3rem;">CardioSense AI</h3>
                <span style="font-size: 0.78rem; color: #64748b; font-weight: 500;">Clinical Decision Support</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    navigation_selection = st.radio(
        "Navigation",
        ["🏠 Dashboard", "❤️ Prediction", "📊 Model Performance", "ℹ️ About"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("#### **System Diagnostics**")
    if artifacts_loaded:
        st.markdown("""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px; font-size: 0.82rem;">
                <span style="color: #166534; font-weight: 700;">● Engine Online</span><br>
                <span style="color: #374151;">Classifier: <b>SVM (RBF Kernel)</b></span><br>
                <span style="color: #374151;">Pipeline: <b>StandardScaler Normalization</b></span><br>
                <span style="color: #374151;">Feature Space: <b>15 Dimensions</b></span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 10px; font-size: 0.82rem; color: #991b1b;">
                <b>● Model Offline</b><br>
                Missing or corrupted model artifacts.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.75rem; color: #94a3b8; line-height: 1.4;">
            <b>Educational & Research Tool</b><br>
            Designed for clinical research. Inferences do not constitute definitive medical advice.
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Fail-Safe Display for Missing Artifacts
# -----------------------------------------------------------------------------
if not artifacts_loaded:
    st.error(f"### ⚠️ Model Loading Error\n\n{load_error_message}")
    st.info(
        "Please ensure `SVM_heart.pkl`, `scaler.pkl`, and `columns.pkl` are present in the project directory:\n"
        f"`{os.path.abspath(os.path.dirname(__file__))}`"
    )
    st.stop()


# =============================================================================
# PAGE 1: 🏠 DASHBOARD
# =============================================================================
if navigation_selection == "🏠 Dashboard":
    st.markdown("""
        <div class="hero-box">
            <h1>❤️ Heart Disease Risk Prediction</h1>
            <p>AI-powered cardiovascular risk assessment driven by non-linear Support Vector Machine learning.</p>
        </div>
    """, unsafe_allow_html=True)

    # Core Metric Highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <span class="metric-label">Selected Model</span>
                <span class="metric-value" style="font-size: 1.55rem; color: #0284c7;">SVM (RBF Kernel)</span>
                <span class="metric-subtext">Max-Margin Non-linear Classifier</span>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <span class="metric-label">Validation Accuracy</span>
                <span class="metric-value" style="color: #059669;">88.04%</span>
                <span class="metric-subtext">Top-performing across 5 tested models</span>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <span class="metric-label">F1-Score</span>
                <span class="metric-value" style="color: #0d9488;">89.62%</span>
                <span class="metric-subtext">Optimized sensitivity for cardiac risk detection</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### 🩺 Overview & Clinical Context")
        st.write(
            "Cardiovascular diseases (CVDs) remain the leading cause of global mortality. "
            "Early risk stratification enables proactive clinical intervention, lifestyle modification, "
            "and timely diagnostic verification. This application operationalizes a trained **Support Vector Machine with a "
            "Radial Basis Function (RBF) Kernel**, delivering calibrated probability estimates across 11 key clinical predictors."
        )

        st.markdown("#### **Inference Architecture**")
        st.markdown("""
            - **Input Layer**: 11 physiological indicators collected via verified clinical inputs.
            - **Encoding**: Dynamic one-hot dummy alignment preserving the exact 15-feature training schema.
            - **Standardization**: Feature projection via fitted `StandardScaler` (zero mean, unit variance).
            - **Kernel Evaluation**: RBF kernel mapping into high-dimensional space ($K(x, x') = \\exp(-\\gamma ||x - x'||^2)$).
            - **Risk Stratification**: Continuous probability score & risk classification output.
        """)

    with col_right:
        st.markdown("### 📋 Monitored Clinical Parameters")
        st.markdown("""
            <div class="clinical-card" style="font-size: 0.9rem; line-height: 1.6;">
                <b>Cardiovascular Vitals:</b>
                <ul style="margin-top: 4px; margin-bottom: 8px;">
                    <li>Age & Biological Sex</li>
                    <li>Resting Blood Pressure (mm Hg)</li>
                    <li>Total Serum Cholesterol (mg/dL)</li>
                    <li>Maximum Heart Rate Achieved (bpm)</li>
                </ul>
                <b>Diagnostic Indicators:</b>
                <ul style="margin-top: 4px; margin-bottom: 0;">
                    <li>Chest Pain Classification (TA, ATA, NAP, ASY)</li>
                    <li>Fasting Blood Sugar (&gt; 120 mg/dL)</li>
                    <li>Resting Electrocardiogram (Normal, ST, LVH)</li>
                    <li>Exercise-Induced Angina (Y / N)</li>
                    <li>ST Slope (Up, Flat, Down) & Oldpeak Depression</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Feature Grid
    st.markdown("### 💡 System Capabilities")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Real-Time Scoring</div>
                <div class="feature-desc">Immediate probability assessment and interactive risk visualization.</div>
            </div>
        """, unsafe_allow_html=True)
    with f_col2:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-icon">📁</div>
                <div class="feature-title">Batch Processing</div>
                <div class="feature-desc">Upload patient cohort CSVs for bulk clinical risk screening.</div>
            </div>
        """, unsafe_allow_html=True)
    with f_col3:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-icon">💡</div>
                <div class="feature-title">What-If Simulator</div>
                <div class="feature-desc">Simulate treatment outcomes by dynamically adjusting vitals.</div>
            </div>
        """, unsafe_allow_html=True)
    with f_col4:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-icon">📥</div>
                <div class="feature-title">Report Export</div>
                <div class="feature-desc">Download structured clinical assessment reports with one click.</div>
            </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
        <div class="disclaimer-card">
            <b>Disclaimer:</b> This application is for educational and research purposes only.
            It is not a medical diagnosis and should not replace advice from a
            qualified healthcare professional.
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: ❤️ PREDICTION
# =============================================================================
elif navigation_selection == "❤️ Prediction":
    st.markdown("## ❤️ Cardiovascular Risk Assessment")
    st.write("Perform individual patient evaluations or process patient cohort data in batch.")

    pred_tab1, pred_tab2 = st.tabs(["🩺 Single Patient Assessment", "📁 Batch Evaluation (CSV)"])

    # -------------------------------------------------------------------------
    # TAB 1: Single Patient Assessment
    # -------------------------------------------------------------------------
    with pred_tab1:
        # Quick Test Profile Loader
        with st.expander("⚡ Quick Test Profiles (Load pre-filled sample patients)", expanded=False):
            col_p1, col_p2, col_p3 = st.columns(3)
            if "preset_data" not in st.session_state:
                st.session_state.preset_data = {}

            with col_p1:
                if st.button("🟢 Load Low-Risk Profile", use_container_width=True):
                    st.session_state.preset_data = {
                        "age": 32,
                        "sex": "Female (F)",
                        "resting_bp": 115,
                        "cholesterol": 175,
                        "max_hr": 172,
                        "oldpeak": 0.0,
                        "chest_pain": "Atypical Angina (ATA)",
                        "fasting_bs": "Normal (<= 120 mg/dL)",
                        "resting_ecg": "Normal",
                        "exercise_angina": "No (N)",
                        "st_slope": "Upsloping (Up)"
                    }
                    st.rerun()

            with col_p2:
                if st.button("🔴 Load Elevated-Risk Profile", use_container_width=True):
                    st.session_state.preset_data = {
                        "age": 58,
                        "sex": "Male (M)",
                        "resting_bp": 145,
                        "cholesterol": 265,
                        "max_hr": 128,
                        "oldpeak": 2.0,
                        "chest_pain": "Asymptomatic (ASY)",
                        "fasting_bs": "Elevated (> 120 mg/dL)",
                        "resting_ecg": "ST-T Wave Abnormality (ST)",
                        "exercise_angina": "Yes (Y)",
                        "st_slope": "Flat"
                    }
                    st.rerun()

            with col_p3:
                if st.button("🔄 Reset Form", use_container_width=True):
                    st.session_state.preset_data = {}
                    st.rerun()

        preset = st.session_state.preset_data

        with st.form("patient_assessment_form"):
            st.markdown("### 1. Patient Demographics")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                age_val = preset.get("age", 50)
                age = st.slider("Patient Age (Years)", 18, 100, int(age_val), help="Range: 18 - 100 years.")
            with col_d2:
                sex_options = ["Male (M)", "Female (F)"]
                default_sex = preset.get("sex", "Male (M)")
                sex_idx = sex_options.index(default_sex) if default_sex in sex_options else 0
                sex_selected = st.selectbox("Biological Sex", sex_options, index=sex_idx)

            st.markdown("---")

            st.markdown("### 2. Cardiovascular Measurements")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                resting_bp = st.number_input(
                    "Resting Blood Pressure (mm Hg)",
                    80, 200, int(preset.get("resting_bp", 125)), 1,
                    help="Normal resting range: < 120 mm Hg."
                )
                max_hr = st.slider(
                    "Maximum Heart Rate Achieved (MaxHR - bpm)",
                    60, 220, int(preset.get("max_hr", 145)), 1,
                    help="Peak exercise heart rate during stress testing."
                )

            with col_m2:
                cholesterol = st.number_input(
                    "Serum Cholesterol (mg/dL)",
                    100, 600, int(preset.get("cholesterol", 210)), 1,
                    help="Desirable cholesterol level: < 200 mg/dL."
                )
                oldpeak = st.number_input(
                    "Oldpeak (ST Depression in mm)",
                    0.0, 6.5, float(preset.get("oldpeak", 0.5)), 0.1, format="%.1f",
                    help="ST depression induced by exercise relative to rest (non-negative)."
                )

            st.markdown("---")

            st.markdown("### 3. Clinical & Electrocardiographic Information")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                cp_options = ["Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"]
                default_cp = preset.get("chest_pain", "Asymptomatic (ASY)")
                cp_idx = cp_options.index(default_cp) if default_cp in cp_options else 3
                chest_pain_selected = st.selectbox("Chest Pain Type", cp_options, index=cp_idx)

                fbs_options = ["Normal (<= 120 mg/dL)", "Elevated (> 120 mg/dL)"]
                default_fbs = preset.get("fasting_bs", "Normal (<= 120 mg/dL)")
                fbs_idx = fbs_options.index(default_fbs) if default_fbs in fbs_options else 0
                fasting_bs_selected = st.selectbox("Fasting Blood Sugar", fbs_options, index=fbs_idx)

                ecg_options = ["Normal", "ST-T Wave Abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"]
                default_ecg = preset.get("resting_ecg", "Normal")
                ecg_idx = ecg_options.index(default_ecg) if default_ecg in ecg_options else 0
                resting_ecg_selected = st.selectbox("Resting Electrocardiogram (Resting ECG)", ecg_options, index=ecg_idx)

            with col_c2:
                angina_options = ["No (N)", "Yes (Y)"]
                default_angina = preset.get("exercise_angina", "No (N)")
                angina_idx = angina_options.index(default_angina) if default_angina in angina_options else 0
                exercise_angina_selected = st.selectbox("Exercise-Induced Angina", angina_options, index=angina_idx)

                slope_options = ["Upsloping (Up)", "Flat", "Downsloping (Down)"]
                default_slope = preset.get("st_slope", "Flat")
                slope_idx = slope_options.index(default_slope) if default_slope in slope_options else 1
                st_slope_selected = st.selectbox("Peak Exercise ST Segment Slope", slope_options, index=slope_idx)

            st.markdown("<br>", unsafe_allow_html=True)
            submit_prediction = st.form_submit_button(
                "🔍 Predict Heart Disease Risk",
                type="primary",
                use_container_width=True
            )

        if submit_prediction:
            sex_code = "M" if "Male" in sex_selected else "F"
            cp_code = {"Typical Angina (TA)": "TA", "Atypical Angina (ATA)": "ATA", "Non-Anginal Pain (NAP)": "NAP", "Asymptomatic (ASY)": "ASY"}[chest_pain_selected]
            fbs_code = 1 if "Elevated" in fasting_bs_selected else 0
            ecg_code = {"Normal": "Normal", "ST-T Wave Abnormality (ST)": "ST", "Left Ventricular Hypertrophy (LVH)": "LVH"}[resting_ecg_selected]
            angina_code = "Y" if "Yes" in exercise_angina_selected else "N"
            slope_code = {"Upsloping (Up)": "Up", "Flat": "Flat", "Downsloping (Down)": "Down"}[st_slope_selected]

            raw_patient_data = {
                "Age": age, "Sex": sex_code, "RestingBP": resting_bp, "Cholesterol": cholesterol,
                "FastingBS": fbs_code, "RestingECG": ecg_code, "MaxHR": max_hr,
                "ExerciseAngina": angina_code, "Oldpeak": oldpeak, "ChestPainType": cp_code, "ST_Slope": slope_code
            }

            validation_errors = validate_patient_inputs(raw_patient_data)
            if validation_errors:
                for err in validation_errors:
                    st.error(f"❌ Input Validation Error: {err}")
            else:
                try:
                    input_df = prepare_features(raw_patient_data, expected_columns)
                    result = predict_heart_disease(model, scaler, input_df)

                    prediction = result["prediction"]
                    probability = result["probability"]
                    has_prob = result["has_probability"]

                    # Save last prediction to session_state for what-if simulation & export
                    st.session_state.last_prediction = {
                        "raw_data": raw_patient_data,
                        "result": result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    st.markdown("---")
                    st.markdown("## 📊 Clinical Assessment Results")

                    if prediction == 1:
                        st.markdown("""
                            <div class="result-card-elevated">
                                <h2 style="margin: 0; color: #991b1b; font-size: 1.8rem;">⚠️ Elevated Risk Detected</h2>
                                <p style="margin-top: 8px; font-size: 1.05rem; line-height: 1.5; color: #7f1d1d;">
                                    <b>Model indicates elevated risk:</b> Patient exhibits biometric signatures correlated 
                                    with coronary heart disease in the validation cohort.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="result-card-low">
                                <h2 style="margin: 0; color: #065f46; font-size: 1.8rem;">✅ Lower Risk Detected</h2>
                                <p style="margin-top: 8px; font-size: 1.05rem; line-height: 1.5; color: #064e3b;">
                                    <b>Model indicates lower risk:</b> Patient clinical markers align with the low-risk 
                                    cardiovascular cohort in the validation dataset.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        st.markdown("### Risk Probability Gauge")
                        if has_prob and probability is not None:
                            prob_pct = probability * 100
                            st.markdown(f"**Estimated Risk Score: {prob_pct:.1f}%**")

                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=prob_pct,
                                number={'suffix': "%", 'font': {'size': 36, 'color': '#0f172a'}},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                                    'bar': {'color': "#dc2626" if prob_pct >= 50 else "#059669", 'thickness': 0.3},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "#e2e8f0",
                                    'steps': [
                                        {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.25)"},
                                        {'range': [40, 60], 'color': "rgba(245, 158, 11, 0.25)"},
                                        {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.25)"}
                                    ],
                                    'threshold': {'line': {'color': "#0f172a", 'width': 3}, 'thickness': 0.8, 'value': 50}
                                }
                            ))
                            fig_gauge.update_layout(
                                height=270,
                                margin=dict(l=20, r=20, t=25, b=20),
                                paper_bgcolor="rgba(0,0,0,0)",
                                font={'color': "#1e293b", 'family': "sans-serif"}
                            )
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        else:
                            st.info("Probability estimation is unavailable for this model configuration.")

                    with res_col2:
                        st.markdown("### Patient Key Biometrics")
                        m_row1_c1, m_row1_c2 = st.columns(2)
                        with m_row1_c1:
                            st.markdown(f"""
                                <div class="metric-card" style="margin-bottom: 10px;">
                                    <span class="metric-label">Age & Sex</span>
                                    <span class="metric-value" style="font-size: 1.4rem;">{age} yrs / {sex_code}</span>
                                    <span class="metric-subtext">Demographic Baseline</span>
                                </div>
                            """, unsafe_allow_html=True)
                        with m_row1_c2:
                            bp_status = "Optimal" if resting_bp < 120 else ("Prehypertension" if resting_bp < 140 else "Hypertensive")
                            bp_color = "#059669" if resting_bp < 120 else ("#d97706" if resting_bp < 140 else "#dc2626")
                            st.markdown(f"""
                                <div class="metric-card" style="margin-bottom: 10px;">
                                    <span class="metric-label">Resting BP</span>
                                    <span class="metric-value" style="font-size: 1.4rem;">{resting_bp} <span style="font-size: 0.9rem; font-weight: 500;">mm Hg</span></span>
                                    <span class="metric-subtext" style="color: {bp_color}; font-weight: 600;">{bp_status}</span>
                                </div>
                            """, unsafe_allow_html=True)

                        m_row2_c1, m_row2_c2 = st.columns(2)
                        with m_row2_c1:
                            chol_status = "Desirable" if cholesterol < 200 else ("Borderline" if cholesterol < 240 else "High")
                            chol_color = "#059669" if cholesterol < 200 else ("#d97706" if cholesterol < 240 else "#dc2626")
                            st.markdown(f"""
                                <div class="metric-card">
                                    <span class="metric-label">Cholesterol</span>
                                    <span class="metric-value" style="font-size: 1.4rem;">{cholesterol} <span style="font-size: 0.9rem; font-weight: 500;">mg/dL</span></span>
                                    <span class="metric-subtext" style="color: {chol_color}; font-weight: 600;">{chol_status}</span>
                                </div>
                            """, unsafe_allow_html=True)
                        with m_row2_c2:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <span class="metric-label">Max Heart Rate</span>
                                    <span class="metric-value" style="font-size: 1.4rem;">{max_hr} <span style="font-size: 0.9rem; font-weight: 500;">bpm</span></span>
                                    <span class="metric-subtext">Exercise Peak HR</span>
                                </div>
                            """, unsafe_allow_html=True)

                        st.markdown(f"""
                            <div class="metric-card" style="margin-top: 10px;">
                                <span class="metric-label">Oldpeak (ST Depression)</span>
                                <span class="metric-value" style="font-size: 1.4rem;">{oldpeak:.1f} <span style="font-size: 0.9rem; font-weight: 500;">mm</span></span>
                                <span class="metric-subtext">Induced ST Segment Depression</span>
                            </div>
                        """, unsafe_allow_html=True)

                    # Export Clinical Report Action
                    st.markdown("### 📥 Export Clinical Assessment Report")
                    report_content = generate_clinical_report(raw_patient_data, result)
                    st.download_button(
                        label="📄 Download Patient Assessment Report (.txt)",
                        data=report_content,
                        file_name=f"cardiosense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"⚠️ Error during feature processing or model inference: {str(e)}")

        # ---------------------------------------------------------------------
        # WHAT-IF SENSITIVITY SIMULATOR (Shows deep ML interactive value)
        # ---------------------------------------------------------------------
        if "last_prediction" in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("💡 Interactive What-If Parameter Simulator (Simulate Clinical Interventions)", expanded=False):
                st.write(
                    "Evaluate how modifying modifiable cardiovascular markers (e.g., blood pressure reduction, "
                    "statin cholesterol lowering, or improved physical conditioning) affects the predicted risk score in real time."
                )

                base_p = st.session_state.last_prediction["raw_data"]
                orig_prob = st.session_state.last_prediction["result"]["probability"]

                sim_c1, sim_c2 = st.columns(2)
                with sim_c1:
                    sim_bp = st.slider("Simulated Resting BP (mm Hg)", 80, 200, int(base_p["RestingBP"]), 5, key="sim_bp")
                    sim_chol = st.slider("Simulated Cholesterol (mg/dL)", 100, 600, int(base_p["Cholesterol"]), 10, key="sim_chol")
                with sim_c2:
                    sim_hr = st.slider("Simulated Max HR (bpm)", 60, 220, int(base_p["MaxHR"]), 5, key="sim_hr")
                    sim_oldpeak = st.slider("Simulated Oldpeak (mm)", 0.0, 6.0, float(base_p["Oldpeak"]), 0.2, key="sim_oldpeak")

                # Compute simulated risk
                sim_data = dict(base_p)
                sim_data["RestingBP"] = sim_bp
                sim_data["Cholesterol"] = sim_chol
                sim_data["MaxHR"] = sim_hr
                sim_data["Oldpeak"] = sim_oldpeak

                sim_df = prepare_features(sim_data, expected_columns)
                sim_result = predict_heart_disease(model, scaler, sim_df)
                sim_prob = sim_result["probability"]

                if orig_prob is not None and sim_prob is not None:
                    delta_prob = (sim_prob - orig_prob) * 100
                    s_col1, s_col2, s_col3 = st.columns(3)
                    with s_col1:
                        st.metric("Original Risk Score", f"{orig_prob * 100:.1f}%")
                    with s_col2:
                        st.metric("Simulated Risk Score", f"{sim_prob * 100:.1f}%", f"{delta_prob:+.1f}%", delta_color="inverse")
                    with s_col3:
                        sim_label = "Elevated Risk" if sim_result["prediction"] == 1 else "Lower Risk"
                        st.metric("Simulated Classification", sim_label)

    # -------------------------------------------------------------------------
    # TAB 2: Batch Assessment via CSV
    # -------------------------------------------------------------------------
    with pred_tab2:
        st.markdown("### 📁 Batch Patient Screening via CSV")
        st.write("Upload a CSV file containing patient records to screen an entire cohort simultaneously.")

        # Download sample template
        sample_df = get_sample_csv_template()
        sample_csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_csv_bytes,
            file_name="cardiosense_sample_cohort.csv",
            mime="text/csv"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_input_df = pd.read_csv(uploaded_file)
                st.write(f"**Loaded Cohort:** {len(batch_input_df)} records identified.")
                st.dataframe(batch_input_df.head(), use_container_width=True)

                if st.button("⚡ Execute Batch Risk Stratification", type="primary"):
                    with st.spinner("Processing batch inferences through StandardScaler & SVM..."):
                        processed_df, elevated_count, lower_count = process_batch_csv(
                            batch_input_df, model, scaler, expected_columns
                        )

                    st.success("Batch classification complete!")
                    
                    b_col1, b_col2, b_col3 = st.columns(3)
                    with b_col1:
                        st.metric("Total Patients Screened", len(processed_df))
                    with b_col2:
                        st.metric("Elevated Risk Identified", elevated_count)
                    with b_col3:
                        st.metric("Lower Risk Identified", lower_count)

                    # Display augmented table
                    st.dataframe(
                        processed_df.style.apply(
                            lambda row: ['background-color: #fee2e2' if row.name == 'Risk_Prediction' and val == 'Elevated Risk' else '' for val in row],
                            axis=1
                        ),
                        use_container_width=True
                    )

                    # Download results
                    result_csv = processed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Stratified Cohort CSV",
                        data=result_csv,
                        file_name=f"cardiosense_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error processing uploaded CSV: {str(e)}. Please verify the column headers match the template.")


# =============================================================================
# PAGE 3: 📊 MODEL PERFORMANCE
# =============================================================================
elif navigation_selection == "📊 Model Performance":
    st.markdown("## 📊 Model Performance & Comparative Benchmark")
    st.write(
        "A rigorous empirical comparison was conducted across 5 machine learning algorithms. "
        "The **Support Vector Machine with Radial Basis Function (RBF) Kernel** demonstrated the highest "
        "generalization accuracy and balanced F1-score on held-out validation cohorts."
    )

    # Top Model Banner
    st.markdown("""
        <div class="clinical-card" style="border-left: 5px solid #0284c7; background: #f0f9ff;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span class="badge-pill" style="background-color: #0284c7; color: white;">Top Performing Algorithm</span>
                    <h3 style="margin: 6px 0 2px 0; color: #0369a1;">Support Vector Machine (RBF Kernel)</h3>
                    <span style="font-size: 0.85rem; color: #475569;">Selected for deployment based on benchmark metrics</span>
                </div>
                <div style="display: flex; gap: 20px;">
                    <div style="text-align: center;">
                        <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">ACCURACY</span><br>
                        <span style="font-size: 1.6rem; font-weight: 800; color: #0284c7;">88.04%</span>
                    </div>
                    <div style="text-align: center;">
                        <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">F1 SCORE</span><br>
                        <span style="font-size: 1.6rem; font-weight: 800; color: #0d9488;">89.62%</span>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Benchmark Data
    model_benchmarks = [
        {"Model": "Logistic Regression", "Accuracy (%)": 85.33, "F1 Score (%)": 86.96},
        {"Model": "K-Nearest Neighbors (KNN)", "Accuracy (%)": 84.78, "F1 Score (%)": 86.41},
        {"Model": "Naive Bayes", "Accuracy (%)": 86.41, "F1 Score (%)": 87.92},
        {"Model": "Decision Tree", "Accuracy (%)": 79.35, "F1 Score (%)": 81.90},
        {"Model": "SVM (RBF Kernel)", "Accuracy (%)": 88.04, "F1 Score (%)": 89.62},
    ]

    benchmark_df = pd.DataFrame(model_benchmarks)

    st.markdown("### 📈 Metric Comparison Across Tested Models")
    
    fig = go.Figure()
    acc_colors = ['#94a3b8', '#94a3b8', '#94a3b8', '#94a3b8', '#0284c7']
    f1_colors = ['#cbd5e1', '#cbd5e1', '#cbd5e1', '#cbd5e1', '#0d9488']

    fig.add_trace(go.Bar(
        x=benchmark_df["Model"],
        y=benchmark_df["Accuracy (%)"],
        name="Accuracy (%)",
        marker_color=acc_colors,
        text=benchmark_df["Accuracy (%)"].apply(lambda v: f"{v:.2f}%"),
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        x=benchmark_df["Model"],
        y=benchmark_df["F1 Score (%)"],
        name="F1 Score (%)",
        marker_color=f1_colors,
        text=benchmark_df["F1 Score (%)"].apply(lambda v: f"{v:.2f}%"),
        textposition="outside"
    ))

    fig.update_layout(
        barmode='group',
        yaxis=dict(title="Percentage (%)", range=[70, 95], gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, sans-serif", 'color': "#1e293b"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Empirical Comparison Table
    st.markdown("### 📋 Empirical Comparison Table")
    st.markdown("""
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 2rem; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <thead>
                <tr style="background-color: #0f172a; color: white; text-align: left;">
                    <th style="padding: 12px 16px;">Model</th>
                    <th style="padding: 12px 16px;">Accuracy</th>
                    <th style="padding: 12px 16px;">F1 Score</th>
                    <th style="padding: 12px 16px;">Outcome</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background-color: #ffffff; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 12px 16px; font-weight: 500;">Logistic Regression</td>
                    <td style="padding: 12px 16px; font-weight: 600;">85.33%</td>
                    <td style="padding: 12px 16px; font-weight: 600;">86.96%</td>
                    <td style="padding: 12px 16px;"><span class="badge-pill badge-neutral">Baseline</span></td>
                </tr>
                <tr style="background-color: #f8fafc; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 12px 16px; font-weight: 500;">K-Nearest Neighbors (KNN)</td>
                    <td style="padding: 12px 16px; font-weight: 600;">84.78%</td>
                    <td style="padding: 12px 16px; font-weight: 600;">86.41%</td>
                    <td style="padding: 12px 16px;"><span class="badge-pill badge-neutral">Evaluated</span></td>
                </tr>
                <tr style="background-color: #ffffff; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 12px 16px; font-weight: 500;">Naive Bayes</td>
                    <td style="padding: 12px 16px; font-weight: 600;">86.41%</td>
                    <td style="padding: 12px 16px; font-weight: 600;">87.92%</td>
                    <td style="padding: 12px 16px;"><span class="badge-pill badge-neutral">Evaluated</span></td>
                </tr>
                <tr style="background-color: #f8fafc; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 12px 16px; font-weight: 500;">Decision Tree</td>
                    <td style="padding: 12px 16px; font-weight: 600;">79.35%</td>
                    <td style="padding: 12px 16px; font-weight: 600;">81.90%</td>
                    <td style="padding: 12px 16px;"><span class="badge-pill badge-neutral">Lowest</span></td>
                </tr>
                <tr style="background-color: #f0f9ff; border-bottom: 2px solid #0284c7; border-left: 4px solid #0284c7;">
                    <td style="padding: 14px 16px; font-weight: 700; color: #0369a1;">SVM (RBF Kernel)</td>
                    <td style="padding: 14px 16px; font-weight: 800; color: #0369a1; font-size: 1.05rem;">88.04%</td>
                    <td style="padding: 14px 16px; font-weight: 800; color: #0d9488; font-size: 1.05rem;">89.62%</td>
                    <td style="padding: 14px 16px;"><span class="badge-pill badge-info" style="font-weight: 700;">⭐ Best Model</span></td>
                </tr>
            </tbody>
        </table>
    """, unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("""
            <div class="clinical-card">
                <h4 style="margin-top: 0; color: #0f172a;">Non-Linear Decision Boundaries</h4>
                <p style="font-size: 0.9rem; color: #475569; line-height: 1.5; margin-bottom: 0;">
                    Cardiovascular interactions (e.g., heart rate reserve vs. ST segment depression) are inherently non-linear. 
                    The RBF kernel projects the 15-dimensional input vectors into an infinite-dimensional feature space, 
                    allowing the classifier to capture intricate boundaries that linear models cannot resolve.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col_r2:
        st.markdown("""
            <div class="clinical-card">
                <h4 style="margin-top: 0; color: #0f172a;">Maximum-Margin Regularization</h4>
                <p style="font-size: 0.9rem; color: #475569; line-height: 1.5; margin-bottom: 0;">
                    While Decision Trees overfit to local sample splits (dropping to 79.35% accuracy), 
                    SVM optimizes a global margin between support vectors of opposing classes. 
                    This structural risk minimization provides robust generalization on unseen patient data.
                </p>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4: ℹ️ ABOUT
# =============================================================================
elif navigation_selection == "ℹ️ About":
    st.markdown("## ℹ️ About CardioSense AI")
    st.write("Technical documentation, methodology, and medical disclaimers.")

    st.markdown("""
        <div class="clinical-card">
            <h3 style="margin-top: 0; color: #0284c7;">🎯 Project Scope</h3>
            <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin-bottom: 0;">
                CardioSense AI is a machine learning clinical decision-support application designed to estimate 
                cardiovascular disease risk using 11 routinely collected patient markers. Built on an empirically 
                validated Support Vector Machine (SVM) pipeline, the system demonstrates how rigorous machine learning 
                can be applied to health informatics.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📐 Metrics & Pipeline Rationale")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
            <div class="clinical-card">
                <h4 style="margin-top: 0; color: #0f172a;">Why F1-Score Matters in Medicine</h4>
                <p style="font-size: 0.9rem; color: #475569; line-height: 1.5; margin-bottom: 0;">
                    Accuracy measures total correct predictions, but in clinical diagnostic support, 
                    <b>False Negatives</b> (missing an at-risk cardiac patient) carry severe real-world consequences. 
                    The F1-Score (89.62%) computes the harmonic mean of Precision and Sensitivity (Recall), 
                    confirming high diagnostic sensitivity while minimizing false alarms.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col_a2:
        st.markdown("""
            <div class="clinical-card">
                <h4 style="margin-top: 0; color: #0f172a;">Why StandardScaler is Mandatory</h4>
                <p style="font-size: 0.9rem; color: #475569; line-height: 1.5; margin-bottom: 0;">
                    The RBF kernel computes geometric distances between feature vectors. Without standardization, 
                    a variable with large numbers like <b>Cholesterol (100–600 mg/dL)</b> would completely overwhelm 
                    features with small numbers like <b>Oldpeak (0.0–6.0 mm)</b> or binary dummy indicators (0 or 1). 
                    The fitted <code>StandardScaler</code> centers each feature at zero mean and unit variance.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### ⚠️ Regulatory & Medical Disclaimer")
    st.markdown("""
        <div class="disclaimer-card" style="border-left-color: #dc2626; background-color: #fef2f2; color: #991b1b;">
            <h4 style="margin-top: 0; color: #991b1b;">Educational & Research Use Only</h4>
            <p style="margin-bottom: 0; font-size: 0.88rem; line-height: 1.5;">
                This software is intended strictly for educational and academic research demonstration. 
                It is not a cleared medical diagnostic device. Predictions are statistical estimates and MUST NOT 
                substitute for professional medical consultation, diagnosis, or clinical evaluation by a certified physician.
            </p>
        </div>
    """, unsafe_allow_html=True)