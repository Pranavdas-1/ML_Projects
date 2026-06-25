import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------
st.markdown("""
<style>

.main{
    background:#f6f8fb;
}

.block-container{
    padding-top:1rem;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 2px 10px rgba(0,0,0,0.08);
}

.result-good{
    background:#e8f5e9;
    color:#1b5e20;
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

.result-bad{
    background:#ffebee;
    color:#b71c1c;
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

.stButton>button{
    width:100%;
    background:#2563eb;
    color:white;
    font-size:18px;
    border-radius:10px;
    height:50px;
    border:none;
}

.stButton>button:hover{
    background:#1d4ed8;
}

.sidebar-title{
    font-size:24px;
    font-weight:bold;
    color:#2563eb;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
@st.cache_resource
def load_model():

    model = joblib.load("customer_churn_model.pkl")
    encoders = joblib.load("churn_encoders.pkl")

    return model, encoders


try:
    model, encoders = load_model()

except Exception as e:

    st.error(f"Unable to load model.\n\n{e}")
    st.stop()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.markdown(
    "<div class='sidebar-title'>📊 Churn Dashboard</div>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Model Information",
        "About Project"
    ]
)

# ----------------------------------------------------
# MODEL INFORMATION PAGE
# ----------------------------------------------------
if page == "Model Information":

    st.title("📈 Model Information")

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithm", "XGBoost")

    c2.metric("Training Samples", "7043")

    c3.metric("Accuracy", "~77%")

    st.markdown("---")

    st.subheader("Dataset")

    st.write("""
This application predicts whether a telecom customer is likely to churn.

The model was trained using the Telco Customer Churn dataset after:

- Missing value handling
- Feature encoding
- SMOTE balancing
- XGBoost Classification
""")

    st.subheader("Top Features")

    st.write("""
- Contract Type
- Tenure
- Monthly Charges
- Internet Service
- Payment Method
- Tech Support
- Online Security
""")

    st.stop()

# ----------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------
if page == "About Project":

    st.title("📘 About")

    st.write("""
Customer churn prediction helps companies identify customers who are likely to leave.

This project demonstrates an end-to-end Machine Learning workflow:

• Data Cleaning

• Feature Engineering

• Label Encoding

• SMOTE Balancing

• XGBoost Classification

• Interactive Streamlit Dashboard
""")

    st.info(
        "Developed using Python, Streamlit, Pandas, Plotly and Scikit-learn."
    )

    st.stop()

# ----------------------------------------------------
# MAIN PAGE
# ----------------------------------------------------
st.title("📊 Customer Churn Analytics Dashboard")

st.write(
    "Enter customer information and predict whether the customer is likely to churn."
)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    [
        "👤 Customer",
        "📡 Services",
        "💳 Account"
    ]
)

# ----------------------------------------------------
# TAB 1
# ----------------------------------------------------
with tab1:

    col1, col2 = st.columns(2)

    with col1:

        gender = st.radio(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0,1],
            format_func=lambda x:"Yes" if x==1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes","No"]
        )

    with col2:

        dependents = st.selectbox(
            "Dependents",
            ["Yes","No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

# ----------------------------------------------------
# TAB 2
# ----------------------------------------------------
with tab2:

    c1, c2 = st.columns(2)

    with c1:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes","No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with c2:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )
# ----------------------------------------------------
# TAB 3
# ----------------------------------------------------
with tab3:

    left, right = st.columns(2)

    with left:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with right:

        monthly_charge = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=150.0,
            value=50.0,
            step=5.0
        )

        total_charge = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=float(monthly_charge * tenure),
            step=50.0
        )

st.markdown("---")

predict_btn = st.button("Predict Customer Churn")

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------
if predict_btn:

    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charge,
        "TotalCharges": total_charge
    }

    input_df = pd.DataFrame([input_data])

    # ----------------------------------------------
    # Encode Binary Columns
    # ----------------------------------------------

    input_df["gender"] = input_df["gender"].map(
        {
            "Male":1,
            "Female":0
        }
    )

    binary_columns = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]

    for col in binary_columns:

        if col in input_df.columns:

            input_df[col] = input_df[col].map(
                {
                    "Yes":1,
                    "No":0
                }
            )

    # ----------------------------------------------
    # Encode Remaining Categorical Features
    # ----------------------------------------------

    for column, encoder in encoders.items():

        if column in input_df.columns:

            if input_df[column].dtype == "object":

                input_df[column] = encoder.transform(
                    input_df[column]
                )

    # ----------------------------------------------
    # Predict
    # ----------------------------------------------

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    churn_probability = probabilities[1]
    stay_probability = probabilities[0]

    confidence = max(probabilities)

    if churn_probability >= 0.75:

        risk_level = "High"

    elif churn_probability >= 0.45:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    st.markdown("---")

    # ==================================================
    # RESULTS SECTION
    # ==================================================

    st.header("Prediction Results")

    card1, card2, card3 = st.columns(3)

    # -----------------------------
    # Prediction Card
    # -----------------------------
    with card1:

        if prediction == 1:

            st.markdown(
                """
                <div class="result-bad">
                🚨 WILL CHURN
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-good">
                ✅ WILL STAY
                </div>
                """,
                unsafe_allow_html=True
            )

    # -----------------------------
    # Confidence
    # -----------------------------
    with card2:

        st.metric(
            label="Model Confidence",
            value=f"{confidence*100:.2f}%"
        )

    # -----------------------------
    # Risk Level
    # -----------------------------
    with card3:

        st.metric(
            label="Risk Level",
            value=risk_level
        )

    st.markdown("---")

    # ==================================================
    # GAUGE CHART
    # ==================================================

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=churn_probability*100,

            title={
                "text":"Churn Probability (%)"
            },

            gauge={

                "axis":{
                    "range":[0,100]
                },

                "bar":{
                    "color":"red"
                },

                "steps":[

                    {
                        "range":[0,40],
                        "color":"lightgreen"
                    },

                    {
                        "range":[40,70],
                        "color":"gold"
                    },

                    {
                        "range":[70,100],
                        "color":"tomato"
                    }

                ]
            }

        )

    )

    gauge.update_layout(
        height=350
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # ==================================================
    # PROBABILITY BAR CHART
    # ==================================================

    fig = go.Figure()

    fig.add_trace(

        go.Bar(

            name="Will Stay",

            x=["Prediction"],

            y=[stay_probability],

            marker_color="#2ecc71"

        )

    )

    fig.add_trace(

        go.Bar(

            name="Will Churn",

            x=["Prediction"],

            y=[churn_probability],

            marker_color="#e74c3c"

        )

    )

    fig.update_layout(

        title="Prediction Probabilities",

        yaxis_title="Probability",

        barmode="group",

        height=400

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # CUSTOMER INSIGHT
    # ==================================================

    st.subheader("Customer Insight")

    if prediction == 1:

        st.warning("""

### High Risk Customer

This customer is likely to leave the company.

Recommended actions:

- Offer promotional discounts.
- Recommend a long-term contract.
- Contact the customer personally.
- Provide better technical support.
- Improve customer engagement.

""")

    else:

        st.success("""

### Loyal Customer

This customer is expected to stay.

Recommended actions:

- Continue providing good service.
- Offer loyalty rewards.
- Send satisfaction surveys.
- Recommend premium plans.
- Maintain regular engagement.

""")

    st.markdown("---")

    # ==================================================
    # QUICK SUMMARY
    # ==================================================

    st.subheader("Prediction Summary")

    summary1, summary2 = st.columns(2)

    with summary1:

        st.info(f"""
Prediction : {"Churn" if prediction==1 else "Stay"}

Confidence : {confidence*100:.2f}%

Risk Level : {risk_level}
""")

    with summary2:

        st.info(f"""
Probability of Staying :

{stay_probability*100:.2f}%

Probability of Churning :

{churn_probability*100:.2f}%
""")

    st.markdown("---")

# ==================================================
# FEATURE IMPORTANCE (STATIC DISPLAY)
# ==================================================

st.header("Top Predictive Features")

feature_names = [
    "Contract",
    "Tenure",
    "Monthly Charges",
    "Internet Service",
    "Payment Method",
    "Tech Support",
    "Online Security",
    "Total Charges",
    "Streaming TV",
    "Partner"
]

importance = [
    100,
    90,
    82,
    73,
    60,
    55,
    48,
    43,
    32,
    24
]

importance_fig = go.Figure()

importance_fig.add_trace(
    go.Bar(
        x=importance[::-1],
        y=feature_names[::-1],
        orientation="h",
        marker_color="#2563eb"
    )
)

importance_fig.update_layout(
    title="Most Important Features Used by the Model",
    xaxis_title="Relative Importance",
    yaxis_title="Feature",
    height=500
)

st.plotly_chart(
    importance_fig,
    use_container_width=True
)

st.markdown("---")

# ==================================================
# MODEL SUMMARY
# ==================================================

st.header("Model Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Algorithm",
        "XGBoost"
    )

with c2:
    st.metric(
        "Training Samples",
        "7043"
    )

with c3:
    st.metric(
        "Features",
        "19"
    )

with c4:
    st.metric(
        "Test Accuracy",
        "~77%"
    )

st.markdown("---")

# ==================================================
# DATASET INFORMATION
# ==================================================

st.header("Dataset Information")

dataset_col1, dataset_col2 = st.columns(2)

with dataset_col1:

    st.markdown("""
### Customer Attributes

- Gender
- Senior Citizen
- Partner
- Dependents
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
""")

with dataset_col2:

    st.markdown("""
### Account Attributes

- Tech Support
- Streaming TV
- Streaming Movies
- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges
- Tenure
""")

st.markdown("---")

# ==================================================
# MODEL PIPELINE
# ==================================================

st.header("Machine Learning Pipeline")

st.markdown("""

1. Data Collection

⬇️

2. Data Cleaning

⬇️

3. Feature Encoding

⬇️

4. SMOTE Balancing

⬇️

5. Train XGBoost Classifier

⬇️

6. Save Model using Joblib

⬇️

7. Predict Customer Churn

""")

st.markdown("---")

# ==================================================
# PROJECT DETAILS
# ==================================================

with st.expander("Project Details"):

    st.write("""
### Libraries Used

- Streamlit
- Pandas
- Plotly
- Joblib
- Scikit-learn
- XGBoost

### Model

The model predicts whether a telecom customer is likely to churn based on demographic, service usage, and account information.

### Target Variable

- 0 → Customer will stay
- 1 → Customer will churn

### Input Features

19 customer attributes are used for prediction.
""")

st.markdown("---")

# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
<div class="footer">

Built with ❤️ using

<b>Python</b> •
<b>Streamlit</b> •
<b>Pandas</b> •
<b>Plotly</b> •
<b>Scikit-learn</b> •
<b>XGBoost</b>

<br><br>

Customer Churn Prediction Dashboard

</div>
""",
    unsafe_allow_html=True
)
