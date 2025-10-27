# 🩺 Personalized Patient Health & Cost Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://personalized-healthcare-recommendations-cyxhunqsyjf5lrcdto7c3y.streamlit.app/)

This is a full-stack data science project that analyzes a rich, personalized patient dataset. It performs a comprehensive data analysis, builds a predictive machine learning model, and presents all findings in an interactive web application built with Streamlit.

## 🚀 Live Demo

**Access the live, interactive dashboard here:**

[**Streamlit App**](https://personalized-healthcare-recommendations-cyxhunqsyjf5lrcdto7c3y.streamlit.app/)

---

## 🎯 Project Objectives

This project was designed to go beyond a simple analysis. Our goal was to complete an end-to-end data science workflow.

1.  **Data Quality & Validation:** To ensure the dataset is clean, complete, and free of anomalies or outliers that could skew the analysis.
2.  **Analyze Insurance Cost Drivers:** To identify the key factors (lifestyle, clinical, genetic) that most significantly influence `Predicted_Insurance_Cost`.
3.  **Evaluate Lifestyle Impact:** To quantify the relationship between patient choices (e.g., `Diet_Type`, `Physical_Activity_Level`) and their `Health_Risk` (Low, Moderate, High).
4.  **Profile High-Risk Patients:** To build a demographic and clinical profile of patients who have a 'High' `Heart_Disease_Risk` or `Diabetes_Risk`.
5.  **Assess Mental Health & Sleep:** To explore the correlation between mental health indicators (`Stress_Level`, `Depression_Score`) and sleep metrics (`Sleep_Hours`, `Sleep_Quality`).
6.  **Examine Genetic Factors:** To analyze the prevalence of genetic markers (`APOE_e4_Carrier`, `Family_History_CVD`, etc.) and their association with disease risk.
7.  **Build a Predictive Model:** To train a baseline Linear Regression model to predict insurance costs and evaluate its performance using **$R^2$** and **RMSE**.
8.  **Deploy as an Interactive App:** To present all findings not as a static report, but as a dynamic, user-friendly web application.

---

## 💡 Key Findings & Insights
(What we understood from the data)

Our analysis uncovered several key insights that are presented in the dashboard:

* **Insurance Cost is Highly Predictable:** Cost is not random. It is strongly correlated with `Age`, `BMI`, and most significantly, `Smoking_Status`. The dashboard's T-test proves this difference is statistically significant (p < 0.05), not just random chance.

* **Lifestyle is a Major Risk Factor:** There is a clear, visual link between physical activity and health. The dashboard shows that `Sedentary` individuals have a much higher percentage of 'High' health risk, while `Highly Active` individuals have a significantly higher proportion of 'Low' risk.

* **Mental and Physical Health are Linked:** Poor `Sleep_Quality` is directly correlated with higher `Stress_Level`, `Depression_Score`, and `Anxiety_Score`, demonstrating the deep connection between mental and physical well-being.

* **Genetic Markers Show Real-World Correlation:** The Polygenic Risk Score for cardiometabolic disease (`PRS_Cardiometabolic`) shows a tangible correlation with key clinical biometrics like `LDL` and `Cholesterol`, bridging the gap between genetic predisposition and clinical reality.

* **A Simple Model Works:** Our baseline Linear Regression model successfully explains a significant portion of the variance in insurance costs. The **$R^2$** score (visible in the app) shows that our model is a strong starting point for more complex predictions.

---

## 🖥️ Final Product: The Dashboard Features
(What we present to the user)

This Streamlit application is the final product of our analysis. It's an interactive tool that allows anyone to explore our findings:

* **Dynamic File Uploader:** The app prompts the user to upload the `personalised_dataset.csv`, and all analysis and charts are generated live.

* **Key Performance Indicators (KPIs):** The dashboard homepage features four key metrics: the model's **$R^2$**, **RMSE**, the **count of High-Risk Patients**, and the **P-Value** of the smoking T-test.

* **Tabbed Navigation:** Findings are organized into clean, logical tabs:
    * **Model Performance:** Shows an interactive "Actual vs. Predicted" scatter plot to visually assess model accuracy.
    * **Cost Drivers:** Features a bar chart for smoking costs and a full correlation heatmap for all numerical features.
    * **Lifestyle & Risk:** Presents a stacked bar chart showing the relationship between `Physical_Activity_Level` and `Health_Risk`.
    * **Data Quality:** Includes the full data quality report (duplicates, missing values) and histograms for `Age`, `BMI`, and `Cost` distributions.

---

## 🛠️ Technology Stack

* **Core Framework:** Streamlit
* **Data Analysis:** Pandas, NumPy, SciPy
* **Predictive Modeling:** Scikit-learn
* **Data Visualization:** Plotly, Seaborn, Matplotlib
* **Language:** Python 3.12+

---

## 🏃 How to Run This Project Locally

To run this application on your own machine, follow these steps:

1.  **Clone the Repository:**
   
    git clone [https://github.com/your-username/your-project-name.git](https://github.com/VenuOnTech/Personalized-Healthcare-Recommendations.git)  
    cd Personalized-Healthcare-Recommendations

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

5.  Open your browser and go to `http://localhost:8501`.

---

## 📁 Project Structure

This repository is organized as follows:


Personalized-Healthcare-Recommendations/  
│  
├── app.py                        # The main Streamlit application  
├── requirements.txt              # Required Python libraries for deployment  
│  
├── datasets/                     # Folder for all raw data  
│   └── personalised_dataset.csv  
│
├── notebook/                     # Folder for exploratory data analysis (EDA)  
│   └── Personalized Healthcare Recommendationsk.ipynb  
│  
├── images/                       # Folder for saved plots from the data analysis  
|   └── [Baseline Model Performance](https://github.com/VenuOnTech/Personalized-Healthcare-Recommendations/blob/main/images/baseline_model_performance.png)  
|```   :```  
|```   :```  
|```   :```  
|  
├── Streamlit App Images/         # Folder for the images from the Streamlit App  
|   └── [Streamlit-App](https://github.com/VenuOnTech/Personalized-Healthcare-Recommendations/blob/main/Streamlit%20App%20Images/Streamlit%20app.png)  
|```   :```  
|```   :```  
|```   :```  
│  
└── README.md  
