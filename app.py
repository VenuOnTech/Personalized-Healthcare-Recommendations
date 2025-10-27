import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- Page Configuration ---
st.set_page_config(
    page_title="Patient Health Dashboard",
    page_icon="🩺",
    layout="wide",
)

# --- Caching Functions (for speed) ---

@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the CSV data."""
    df = pd.read_csv("personalised_dataset.csv")
    # Basic deduplication
    if 'Patient_ID' in df.columns:
        df = df.drop_duplicates(subset=['Patient_ID'], keep='first')
    return df

@st.cache_data
def perform_data_quality_check(_df):
    """Runs data quality checks."""
    report = {}
    report['duplicate_patient_ids'] = int(_df['Patient_ID'].duplicated().sum())
    missing = _df.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    return report

@st.cache_data
def train_baseline_model(_df):
    """Trains the model and returns metrics and predictions."""
    target = 'Predicted_Insurance_Cost'
    numerical_features = ['Age', 'BMI', 'Cholesterol', 'Glucose_Level', 'HbA1c', 'Systolic_BP', 'Diastolic_BP', 'LDL', 'HDL', 'Triglycerides', 'Stress_Level']
    categorical_features = ['Gender', 'Smoking_Status', 'Alcohol_Consumption', 'Physical_Activity_Level', 'Diet_Type', 'Heart_Disease_Risk', 'Diabetes_Risk']
    
    # Filter out potential missing features for robustness
    numerical_features = [col for col in numerical_features if col in _df.columns]
    categorical_features = [col for col in categorical_features if col in _df.columns]
    
    X = _df[numerical_features + categorical_features]
    y = _df[target]

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Create a DataFrame for the "Actual vs. Predicted" plot
    predictions_df = pd.DataFrame({'Actual Cost': y_test, 'Predicted Cost': y_pred})
    
    return r2, rmse, predictions_df

@st.cache_data
def get_smoking_ttest(_df):
    """Performs T-test on smoking vs. cost."""
    cost_smoker = _df[_df['Smoking_Status'] == 'Current smoker']['Predicted_Insurance_Cost']
    cost_nonsmoker = _df[_df['Smoking_Status'] == 'Non-smoker']['Predicted_Insurance_Cost']
    if len(cost_smoker) > 1 and len(cost_nonsmoker) > 1:
        t_stat, p_value = stats.ttest_ind(cost_smoker, cost_nonsmoker, equal_var=False)
        return p_value
    return 1.0 # Return non-significant if data is missing

# --- Sidebar ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-primary-col.svg", width=250)
    st.title("Patient Health Analyzer")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload your `personalised_dataset.csv`",
        type=["csv"],
        help="Upload the patient dataset to begin analysis."
    )
    st.markdown("---")
    st.info("This app performs a full analysis, including data quality checks, predictive modeling, and results visualization. All analysis is done live in your browser.")

# --- Main App Body ---
if uploaded_file is None:
    st.title("🩺 Welcome to the Patient Health Analyzer")
    st.markdown("### Please upload your `personalised_dataset.csv` using the sidebar to begin.")
    st.markdown("The dashboard will appear here once your file is loaded.")
    st.image("https://user-images.githubusercontent.com/10280424/222329380-087a6c24-c3c1-432d-88b9-9f3d67189196.png", caption="Example of a Streamlit dashboard")

else:
    # --- Data Loading and Analysis ---
    with st.spinner('Analyzing your data... This may take a moment.'):
        df = load_data(uploaded_file)
        quality_report = perform_data_quality_check(df)
        r2, rmse, predictions_df = train_baseline_model(df)
        p_value = get_smoking_ttest(df)
        high_risk_count = df[(df['Heart_Disease_Risk'] == 'High') | (df['Diabetes_Risk'] == 'High')].shape[0]

    # --- Dashboard UI ---
    st.title("🩺 Patient Health Analysis Dashboard")
    st.markdown("Here is the complete analysis of your patient dataset.")

    # --- Key Metrics ---
    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model R-Squared ($R^2$)", f"{r2:.3f}", help="Fraction of insurance cost variance explained by the model.")
    col2.metric("Model RMSE (Error)", f"${rmse:.2f}", help="The average dollar error of the model's cost prediction.")
    col3.metric("High-Risk Patients", f"{high_risk_count}", help="Count of patients with 'High' heart disease or diabetes risk.")
    col4.metric("Smoking Cost P-Value", f"{p_value:.2e}", "p-value for T-test comparing smoker vs. non-smoker costs. (p < 0.05 is significant)")

    st.markdown("---")

    # --- Tabs for Analysis ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Model Performance", 
        "💰 Cost Drivers", 
        "❤️ Lifestyle & Risk", 
        "📊 Data Quality"
    ])

    with tab1:
        st.subheader("Baseline Model Performance (Actual vs. Predicted Cost)")
        
        # Interactive Scatter Plot
        fig = px.scatter(
            predictions_df, 
            x='Actual Cost', 
            y='Predicted Cost', 
            title=f"Actual vs. Predicted Insurance Cost (R²: {r2:.3f})",
            labels={'Actual Cost': 'Actual Cost ($)', 'Predicted Cost': 'Predicted Cost ($)'},
            hover_data=[predictions_df.index]
        )
        # Add the 'perfect prediction' line
        fig.add_trace(go.Scatter(
            x=[predictions_df['Actual Cost'].min(), predictions_df['Actual Cost'].max()],
            y=[predictions_df['Actual Cost'].min(), predictions_df['Actual Cost'].max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Analysis of Insurance Cost Drivers")
        
        col_cost1, col_cost2 = st.columns([1, 2])
        
        with col_cost1:
            st.markdown("#### Cost by Smoking Status")
            # Interactive Bar Chart
            smoking_cost_df = df.groupby('Smoking_Status')['Predicted_Insurance_Cost'].mean().reset_index()
            fig = px.bar(
                smoking_cost_df, 
                x='Smoking_Status', 
                y='Predicted_Insurance_Cost',
                title="Average Cost by Smoking Status",
                labels={'Predicted_Insurance_Cost': 'Average Cost ($)'},
                color='Smoking_Status'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_cost2:
            st.markdown("#### Correlation Heatmap (Numerical Features)")
            # Correlation Heatmap
            numerical_features = ['Age', 'BMI', 'Cholesterol', 'Glucose_Level', 'HbA1c', 'Systolic_BP', 'LDL', 'Triglycerides', 'Stress_Level', 'Predicted_Insurance_Cost']
            corr_matrix = df[numerical_features].corr()
            
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Feature Correlation with Insurance Cost")
            st.pyplot(fig)

    with tab3:
        st.subheader("Lifestyle Impact on Health Risk")
        
        # Stacked Bar Chart
        crosstab_perc = pd.crosstab(df['Physical_Activity_Level'], df['Health_Risk'], normalize='index') * 100
        crosstab_df = crosstab_perc.reset_index().melt(id_vars='Physical_Activity_Level', var_name='Health_Risk', value_name='Percentage')
        
        fig = px.bar(
            crosstab_df,
            x="Physical_Activity_Level",
            y="Percentage",
            color="Health_Risk",
            title="Health Risk Distribution by Physical Activity Level",
            barmode="stack",
            category_orders={"Physical_Activity_Level": ["Sedentary", "Lightly Active", "Moderately Active", "Highly Active"]},
            color_discrete_map={'Low': 'green', 'Moderate': 'orange', 'High': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Data Quality & Distribution Analysis")
        st.markdown("#### Data Quality Report")
        st.json(quality_report)
        st.markdown("#### Feature Distributions")
        
        col_dist1, col_dist2, col_dist3 = st.columns(3)
        
        with col_dist1:
            fig = px.histogram(df, x="Age", title="Age Distribution", nbins=30)
            st.plotly_chart(fig, use_container_width=True)
            fig = px.box(df, y="Age", title="Age Outlier Boxplot")
            st.plotly_chart(fig, use_container_width=True)

        with col_dist2:
            fig = px.histogram(df, x="BMI", title="BMI Distribution", nbins=30)
            st.plotly_chart(fig, use_container_width=True)
            fig = px.box(df, y="BMI", title="BMI Outlier Boxplot")
            st.plotly_chart(fig, use_container_width=True)
            
        with col_dist3:
            fig = px.histogram(df, x="Predicted_Insurance_Cost", title="Cost Distribution", nbins=30)
            st.plotly_chart(fig, use_container_width=True)
            fig = px.box(df, y="Predicted_Insurance_Cost", title="Cost Outlier Boxplot")
            st.plotly_chart(fig, use_container_width=True)