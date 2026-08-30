import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_diabetes_best_model.joblib"


# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Model tidak ditemukan: {MODEL_PATH.name}. "
            "Pastikan file model berada pada folder yang sama dengan aplikasi."
        )
        st.stop()

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# FEATURE SPECIFICATION
# =========================================================
GENDERS = ["Male", "Female", "Other"]

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad",
    "Chennai", "Kolkata", "Pune", "Surat", "Jaipur",
    "Kanpur", "Nagpur", "Indore", "Lucknow", "Thane",
    "Bhopal", "Visakhapatnam", "Patna"
]

FAMILY_HISTORY = ["No", "Yes"]
PHYSICAL_ACTIVITY = ["Active", "Moderate", "Sedentary"]
DIET_TYPES = ["Non-Vegetarian", "Vegetarian", "Pescatarian", "Vegan"]
SMOKING_STATUS = ["Never", "Former", "Current"]
ALCOHOL_CONSUMPTION = ["Never", "Occasional", "Regular"]
INCOME_BRACKET = ["Low", "Middle", "High"]

FEATURE_COLUMNS = [
    "age",
    "gender",
    "city",
    "bmi",
    "family_history_diabetes",
    "physical_activity_level",
    "diet_type",
    "smoking_status",
    "alcohol_consumption",
    "hours_sleep_per_night",
    "stress_level",
    "fasting_blood_sugar",
    "hba1c_level",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "waist_circumference_cm",
    "income_bracket",
]


# =========================================================
# DEFAULT / RANDOM DATA
# =========================================================
DEFAULT_DATA = {
    "age": 45,
    "gender": "Male",
    "city": "Mumbai",
    "bmi": 22.7,
    "family_history_diabetes": "No",
    "physical_activity_level": "Moderate",
    "diet_type": "Non-Vegetarian",
    "smoking_status": "Never",
    "alcohol_consumption": "Never",
    "hours_sleep_per_night": 7.0,
    "stress_level": 6,
    "fasting_blood_sugar": 165.0,
    "hba1c_level": 6.8,
    "blood_pressure_systolic": 140,
    "blood_pressure_diastolic": 88,
    "waist_circumference_cm": 99.3,
    "income_bracket": "Middle",
}


def clipped_normal(mean, std, low, high, decimals=1):
    value = np.random.normal(mean, std)
    value = np.clip(value, low, high)
    return round(float(value), decimals)


def generate_random_data():
    """Generate data acak yang masih berada dalam rentang dataset training."""
    return {
        "age": int(round(clipped_normal(44.33, 10.38, 18, 80, 0))),
        "gender": random.choices(
            GENDERS,
            weights=[0.511, 0.479, 0.010],
            k=1,
        )[0],
        "city": random.choices(
            CITIES,
            weights=[
                2243, 1737, 1535, 1227, 1039, 933, 872, 756, 734,
                627, 475, 462, 458, 457, 451, 399, 307, 288
            ],
            k=1,
        )[0],
        "bmi": clipped_normal(23.06, 4.05, 15.0, 40.8, 1),
        "family_history_diabetes": random.choices(
            FAMILY_HISTORY,
            weights=[0.6504, 0.3496],
            k=1,
        )[0],
        "physical_activity_level": random.choices(
            PHYSICAL_ACTIVITY,
            weights=[5100, 4950, 4950],
            k=1,
        )[0],
        "diet_type": random.choices(
            DIET_TYPES,
            weights=[8195, 6074, 445, 286],
            k=1,
        )[0],
        "smoking_status": random.choices(
            SMOKING_STATUS,
            weights=[10119, 1463, 2946],
            k=1,
        )[0],
        "alcohol_consumption": random.choices(
            ALCOHOL_CONSUMPTION,
            weights=[5616, 4483, 1113],
            k=1,
        )[0],
        "hours_sleep_per_night": clipped_normal(7.31, 1.11, 3.0, 12.0, 1),
        "stress_level": int(round(clipped_normal(5.94, 2.15, 1, 10, 0))),
        "fasting_blood_sugar": clipped_normal(168.79, 38.37, 72.0, 400.0, 1),
        "hba1c_level": clipped_normal(6.88, 0.84, 4.8, 13.4, 1),
        "blood_pressure_systolic": int(
            round(clipped_normal(139.50, 12.66, 80, 200, 0))
        ),
        "blood_pressure_diastolic": int(
            round(clipped_normal(86.26, 8.39, 50, 120, 0))
        ),
        "waist_circumference_cm": clipped_normal(
            100.21, 10.76, 73.2, 148.2, 1
        ),
        "income_bracket": random.choices(
            INCOME_BRACKET,
            weights=[3388, 7305, 3844],
            k=1,
        )[0],
    }


def initialize_session_state():
    if "input_data" not in st.session_state:
        st.session_state.input_data = DEFAULT_DATA.copy()

    if "prediction" not in st.session_state:
        st.session_state.prediction = None


initialize_session_state()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Input Data")

    input_mode = st.radio(
        "Pilih metode input",
        ["Manual Input", "Generate Random Data"],
        index=0,
    )

    st.divider()

    if input_mode == "Generate Random Data":
        st.caption(
            "Data acak dibangkitkan berdasarkan rentang dan distribusi "
            "fitur pada dataset training."
        )

        if st.button(
            "Generate Data Acak",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.input_data = generate_random_data()
            st.session_state.prediction = None
            st.rerun()

    else:
        st.caption(
            "Masukkan karakteristik pasien secara manual pada formulir utama."
        )

        if st.button("Reset Input", use_container_width=True):
            st.session_state.input_data = DEFAULT_DATA.copy()
            st.session_state.prediction = None
            st.rerun()

    st.divider()
    st.subheader("Model")
    st.write("**Algoritma:** SVM")
    st.write("**Kernel:** Linear")
    st.write("**C:** 0.1")
    st.write("**Class weight:** Balanced")


# =========================================================
# HEADER
# =========================================================
st.title("Diabetes Risk Prediction Dashboard")
st.caption(
    "Prediksi tingkat risiko diabetes menggunakan model Support Vector Machine (SVM)."
)

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Test Accuracy", "76.70%")
metric2.metric("Macro F1", "73.91%")
metric3.metric("Output Classes", "3")


# =========================================================
# INPUT FORM
# =========================================================
st.divider()
st.subheader("Data Pasien")

random_mode = input_mode == "Generate Random Data"
current = st.session_state.input_data


def option_index(options, value):
    try:
        return options.index(value)
    except ValueError:
        return 0


with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(
        [
            "Demografi",
            "Gaya Hidup",
            "Pemeriksaan Klinis",
        ]
    )

    # -----------------------------------------------------
    # DEMOGRAPHIC
    # -----------------------------------------------------
    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=80,
                value=int(current["age"]),
                step=1,
                disabled=random_mode,
            )

            gender = st.selectbox(
                "Gender",
                GENDERS,
                index=option_index(GENDERS, current["gender"]),
                disabled=random_mode,
            )

        with col2:
            city = st.selectbox(
                "City",
                CITIES,
                index=option_index(CITIES, current["city"]),
                disabled=random_mode,
            )

            income_bracket = st.selectbox(
                "Income Bracket",
                INCOME_BRACKET,
                index=option_index(
                    INCOME_BRACKET,
                    current["income_bracket"],
                ),
                disabled=random_mode,
            )

        with col3:
            family_history_diabetes = st.selectbox(
                "Family History of Diabetes",
                FAMILY_HISTORY,
                index=option_index(
                    FAMILY_HISTORY,
                    current["family_history_diabetes"],
                ),
                disabled=random_mode,
            )

            bmi = st.number_input(
                "BMI",
                min_value=15.0,
                max_value=40.8,
                value=float(current["bmi"]),
                step=0.1,
                format="%.1f",
                disabled=random_mode,
            )

    # -----------------------------------------------------
    # LIFESTYLE
    # -----------------------------------------------------
    with tab2:
        col1, col2, col3 = st.columns(3)

        with col1:
            physical_activity_level = st.selectbox(
                "Physical Activity Level",
                PHYSICAL_ACTIVITY,
                index=option_index(
                    PHYSICAL_ACTIVITY,
                    current["physical_activity_level"],
                ),
                disabled=random_mode,
            )

            diet_type = st.selectbox(
                "Diet Type",
                DIET_TYPES,
                index=option_index(
                    DIET_TYPES,
                    current["diet_type"],
                ),
                disabled=random_mode,
            )

        with col2:
            smoking_status = st.selectbox(
                "Smoking Status",
                SMOKING_STATUS,
                index=option_index(
                    SMOKING_STATUS,
                    current["smoking_status"],
                ),
                disabled=random_mode,
            )

            alcohol_consumption = st.selectbox(
                "Alcohol Consumption",
                ALCOHOL_CONSUMPTION,
                index=option_index(
                    ALCOHOL_CONSUMPTION,
                    current["alcohol_consumption"],
                ),
                disabled=random_mode,
            )

        with col3:
            hours_sleep_per_night = st.number_input(
                "Hours Sleep per Night",
                min_value=3.0,
                max_value=12.0,
                value=float(current["hours_sleep_per_night"]),
                step=0.1,
                format="%.1f",
                disabled=random_mode,
            )

            stress_level = st.slider(
                "Stress Level",
                min_value=1,
                max_value=10,
                value=int(current["stress_level"]),
                disabled=random_mode,
            )

    # -----------------------------------------------------
    # CLINICAL
    # -----------------------------------------------------
    with tab3:
        col1, col2, col3 = st.columns(3)

        with col1:
            fasting_blood_sugar = st.number_input(
                "Fasting Blood Sugar",
                min_value=72.0,
                max_value=400.0,
                value=float(current["fasting_blood_sugar"]),
                step=1.0,
                format="%.1f",
                disabled=random_mode,
            )

            hba1c_level = st.number_input(
                "HbA1c Level",
                min_value=4.8,
                max_value=13.4,
                value=float(current["hba1c_level"]),
                step=0.1,
                format="%.1f",
                disabled=random_mode,
            )

        with col2:
            blood_pressure_systolic = st.number_input(
                "Blood Pressure Systolic",
                min_value=80,
                max_value=200,
                value=int(current["blood_pressure_systolic"]),
                step=1,
                disabled=random_mode,
            )

            blood_pressure_diastolic = st.number_input(
                "Blood Pressure Diastolic",
                min_value=50,
                max_value=120,
                value=int(current["blood_pressure_diastolic"]),
                step=1,
                disabled=random_mode,
            )

        with col3:
            waist_circumference_cm = st.number_input(
                "Waist Circumference (cm)",
                min_value=73.2,
                max_value=148.2,
                value=float(current["waist_circumference_cm"]),
                step=0.1,
                format="%.1f",
                disabled=random_mode,
            )

    submit = st.form_submit_button(
        "Prediksi Risiko Diabetes",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# PREDICTION
# =========================================================
if submit:
    input_record = {
        "age": age,
        "gender": gender,
        "city": city,
        "bmi": bmi,
        "family_history_diabetes": family_history_diabetes,
        "physical_activity_level": physical_activity_level,
        "diet_type": diet_type,
        "smoking_status": smoking_status,
        "alcohol_consumption": alcohol_consumption,
        "hours_sleep_per_night": hours_sleep_per_night,
        "stress_level": stress_level,
        "fasting_blood_sugar": fasting_blood_sugar,
        "hba1c_level": hba1c_level,
        "blood_pressure_systolic": blood_pressure_systolic,
        "blood_pressure_diastolic": blood_pressure_diastolic,
        "waist_circumference_cm": waist_circumference_cm,
        "income_bracket": income_bracket,
    }

    input_df = pd.DataFrame(
        [input_record],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(input_df)[0]

    st.session_state.input_data = input_record
    st.session_state.prediction = prediction


# =========================================================
# RESULT
# =========================================================
if st.session_state.prediction is not None:
    st.divider()
    st.subheader("Hasil Prediksi")

    prediction = st.session_state.prediction

    if prediction == "Low":
        st.success("Prediksi Risiko Diabetes: LOW")
        st.caption("Model mengklasifikasikan data pasien ke kelompok risiko rendah.")

    elif prediction == "Moderate":
        st.warning("Prediksi Risiko Diabetes: MODERATE")
        st.caption("Model mengklasifikasikan data pasien ke kelompok risiko sedang.")

    else:
        st.error("Prediksi Risiko Diabetes: HIGH")
        st.caption("Model mengklasifikasikan data pasien ke kelompok risiko tinggi.")

    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        st.metric("Predicted Class", prediction)

        if hasattr(model.named_steps["svm"], "decision_function"):
            input_df = pd.DataFrame(
                [st.session_state.input_data],
                columns=FEATURE_COLUMNS,
            )

            decision_scores = model.decision_function(input_df)[0]

            if np.ndim(decision_scores) > 0:
                score_df = pd.DataFrame(
                    {
                        "Class": model.classes_,
                        "Decision Score": decision_scores,
                    }
                ).sort_values(
                    "Decision Score",
                    ascending=False,
                )

                st.caption(
                    "Decision score adalah skor pemisahan SVM, "
                    "bukan probabilitas."
                )
                st.dataframe(
                    score_df,
                    hide_index=True,
                    use_container_width=True,
                )

    with result_col2:
        display_data = pd.DataFrame(
            {
                "Feature": [
                    "Age",
                    "Gender",
                    "City",
                    "BMI",
                    "Family History",
                    "Physical Activity",
                    "Diet Type",
                    "Smoking Status",
                    "Alcohol Consumption",
                    "Sleep Hours",
                    "Stress Level",
                    "Fasting Blood Sugar",
                    "HbA1c",
                    "Systolic BP",
                    "Diastolic BP",
                    "Waist Circumference",
                    "Income Bracket",
                ],
                "Value": list(st.session_state.input_data.values()),
            }
        )

        st.dataframe(
            display_data,
            hide_index=True,
            use_container_width=True,
        )


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Dashboard ini merupakan demonstrasi model machine learning dan "
    "bukan alat diagnosis medis."
)
