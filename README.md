# 🦾 ArthroAI – Rheumatoid Arthritis Risk Prediction System

**ArthroAI** is a clinical decision support tool that predicts the risk of Rheumatoid Arthritis (RA) using an XGBoost model trained on NHANES data. The interactive Streamlit web application provides real‑time risk assessment, detailed factor analysis, and a downloadable clinical report.

![ArthroAI Screenshot](screenshot.png) <!-- optional: add a screenshot later -->

## ✨ Features

- **Machine Learning Core** – XGBoost classifier (tuned) achieving research‑grade performance.
- **Interactive Form** – 15 clinical, demographic, and lifestyle inputs with intuitive sliders and selectors.
- **Real‑time Risk Prediction** – Shows probability (0–100%) and class (Low / Moderate / High risk).
- **Clinical Recommendations** – Immediate, evidence‑based advice based on risk level.
- **Detailed Factor Analysis** – Visual breakdown of key contributors (age, BMI, smoking, hypertension).
- **Probability Gauge** – Interactive Plotly gauge chart.
- **PDF Report** – One‑click download of a formatted research report with all inputs and results.
- **Reset & Example** – Buttons to reset all values or load a typical patient case.
- **Responsive Design** – Works on desktop and tablet screens.

## 🧠 Model Details

- **Algorithm**: XGBoost (gradient boosting)
- **Training Data**: NHANES (National Health and Nutrition Examination Survey)
- **Features** (15 total):
  - Age, BRI, BRI Trend, BMI
  - Hypertension, Hyperlipidemia, Diabetes
  - Smoking status, Alcohol consumption
  - Physical activity level
  - Protein, calorie, fiber, carbohydrate intake
  - Education level
- **Output**: Probability of RA (0–1) → risk categories:
  - Low: <30%
  - Moderate: 30%–70%
  - High: >70%

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ArthroAI.git
   cd ArthroAI
