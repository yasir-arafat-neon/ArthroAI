import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
import base64
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

# Try to import xgboost with version detection
try:
    import xgboost as xgb

    XGB_VERSION = xgb.__version__
except ImportError:
    XGB_VERSION = None

st.set_page_config(
    page_title="ArthroAI • Rheumatoid Arthritis Risk Prediction System",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ PROFESSIONAL COLOR THEME ============
COLORS = {
    'primary': '#1e3c72',
    'secondary': '#2a5298',
    'accent': '#6A4C93',
    'success': '#2E8B57',
    'warning': '#FF8C42',
    'danger': '#D32F2F',
    'light_bg': '#F8FAFC',
    'dark_text': '#1E293B',
    'border': '#E2E8F0',
    'info': '#3B82F6',
    'card_bg': '#FFFFFF',
}

# ============ DEFAULT VALUES FOR RESET ============
DEFAULTS = {
    'age': 55,
    'bmi': 28.5,
    'bri': 4.2,
    'bri_trend': 0.5,
    'physical_activity': 2.5,
    'protein': 70,
    'calories': 2200,
    'fiber': 25,
    'carbs': 250,
    'education': 3,
    'hypertension': "Yes",
    'hyperlipidemia': "Yes",
    'diabetes': "No",
    'smoking': "No",
    'drinking': "Yes"
}

EXAMPLE = {
    'age': 55,
    'bmi': 28.5,
    'bri': 4.2,
    'bri_trend': 0.5,
    'physical_activity': 2.5,
    'protein': 70,
    'calories': 2200,
    'fiber': 25,
    'carbs': 250,
    'education': 3,
    'hypertension': "Yes",
    'hyperlipidemia': "Yes",
    'diabetes': "No",
    'smoking': "No",
    'drinking': "Yes"
}

# ============ CSS STYLING ============
st.markdown(f"""
<style>
    .main .block-container {{
        max-width: 1400px;
        padding: 1.5rem 2rem 1.5rem 2rem;
        margin: 0 auto;
    }}
    .research-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(30,60,114,0.15);
        text-align: center;
    }}
    .research-title {{
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    .research-subtitle {{
        font-size: 1.1rem;
        color: rgba(255,255,255,0.95);
        margin-top: 0.5rem;
    }}
    .institution-badge {{
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 1.2rem;
        border-radius: 25px;
        display: inline-block;
        font-size: 0.85rem;
        margin-top: 0.8rem;
        backdrop-filter: blur(5px);
    }}
    .academic-card {{
        background: {COLORS['card_bg']};
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid {COLORS['border']};
        margin-bottom: 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: all 0.3s ease;
    }}
    .academic-card:hover {{
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }}
    .card-title {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {COLORS['primary']};
        border-bottom: 3px solid {COLORS['secondary']};
        padding-bottom: 0.6rem;
        margin-bottom: 1.3rem;
        display: inline-block;
        width: auto;
    }}
    .section-header {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {COLORS['secondary']};
        margin: 0.8rem 0 0.8rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px dashed {COLORS['border']};
    }}
    .result-container {{
        background: linear-gradient(135deg, white 0%, {COLORS['light_bg']} 100%);
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border-left: 6px solid {COLORS['accent']};
        margin: 1.5rem 0;
    }}
    .risk-level {{
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin: 0.3rem 0;
    }}
    .probability {{
        font-size: 1.3rem;
        text-align: center;
        margin: 0.3rem 0;
    }}
    .recommendation-box {{
        background: #FFF8F0;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border-left: 4px solid {COLORS['warning']};
        margin-top: 1rem;
        line-height: 1.5;
    }}
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
        margin: 1.2rem 0;
    }}
    .metric-card {{
        background: {COLORS['card_bg']};
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid {COLORS['border']};
        transition: all 0.3s;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        border-color: {COLORS['secondary']};
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {COLORS['primary']};
        line-height: 1.2;
    }}
    .metric-label {{
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.3rem;
        font-weight: 500;
    }}
    .metric-status {{
        font-size: 0.7rem;
        margin-top: 0.4rem;
        font-weight: 600;
    }}
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        border: none;
        transition: all 0.3s;
        width: 100%;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(30,60,114,0.2);
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30,60,114,0.3);
    }}
    .custom-divider {{
        height: 2px;
        background: linear-gradient(to right, transparent, {COLORS['secondary']}50, transparent);
        margin: 1.5rem 0;
    }}
    .research-footer {{
        text-align: center;
        padding: 2rem 0;
        margin-top: 2rem;
        border-top: 1px solid {COLORS['border']};
        color: #64748B;
        font-size: 0.85rem;
    }}
    .status-badge {{
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }}
    .status-low {{
        background: {COLORS['success']}20;
        color: {COLORS['success']};
    }}
    .status-moderate {{
        background: {COLORS['warning']}20;
        color: {COLORS['warning']};
    }}
    .status-high {{
        background: {COLORS['danger']}20;
        color: {COLORS['danger']};
    }}
    .clinical-alert {{
        background: {COLORS['danger']}10;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['danger']};
        margin: 1rem 0;
        font-size: 0.85rem;
    }}
    .warning-alert {{
        background: {COLORS['warning']}10;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['warning']};
        margin: 1rem 0;
        font-size: 0.85rem;
    }}
    .chart-container {{
        background: {COLORS['card_bg']};
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid {COLORS['border']};
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    @media (max-width: 1200px) {{
        .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 768px) {{
        .metrics-grid {{ grid-template-columns: 1fr; }}
    }}
</style>
""", unsafe_allow_html=True)


# ============ ROBUST MODEL LOADING ============
@st.cache_resource
def load_model():
    model_path = 'Initial_RA_XGBoost_Tuned.pkl'

    # Look for model in multiple locations
    if not os.path.exists(model_path):
        alt_paths = [
            './Initial_RA_XGBoost_Tuned.pkl',
            '../Initial_RA_XGBoost_Tuned.pkl',
            os.path.join(os.path.dirname(__file__), 'Initial_RA_XGBoost_Tuned.pkl'),
            os.path.join(os.getcwd(), 'Initial_RA_XGBoost_Tuned.pkl')
        ]
        for path in alt_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            st.error("❌ Model file not found. Please ensure `Initial_RA_XGBoost_Tuned.pkl` is in the app directory.")
            st.stop()

    # Try joblib first (most common)
    try:
        model = joblib.load(model_path)
        if hasattr(model, 'use_label_encoder'):
            model.use_label_encoder = False
        st.success(f"✅ Model loaded with joblib (XGBoost {XGB_VERSION})")
        return model
    except Exception as e1:
        st.warning(f"Joblib load failed: {str(e1)[:100]}...")

    # Try pickle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        if hasattr(model, 'use_label_encoder'):
            model.use_label_encoder = False
        st.success("✅ Model loaded with pickle")
        return model
    except Exception as e2:
        st.warning(f"Pickle load failed: {str(e2)[:100]}...")

    # Try loading as booster and wrapping
    try:
        bst = xgb.Booster()
        bst.load_model(model_path)
        model = xgb.XGBClassifier()
        model._Booster = bst
        st.success(f"✅ Model loaded as Booster (XGBoost {XGB_VERSION})")
        return model
    except Exception as e3:
        st.error(f"❌ All loading methods failed. Last error: {str(e3)}")
        st.info("💡 Recommended fix:\n"
                "1. `pip uninstall xgboost -y`\n"
                "2. `pip install xgboost==2.0.3`\n"
                "3. Restart the app")
        st.stop()


model = load_model()

# ============ HEADER ============
st.markdown(f"""
<div class="research-header">
    <div class="research-title">🦾 ArthroAI</div>
    <div class="research-subtitle">Rheumatoid Arthritis Risk Prediction System | Clinical Decision Support Tool</div>
    <div class="institution-badge">NHANES Data • XGBoost Model • Research Grade</div>
</div>
""", unsafe_allow_html=True)

# ============ FEATURE DEFINITIONS ============
FEATURES = [
    'Age', 'BRI_Trend', 'BRI', 'Hypertension', 'Hyperlipidemia',
    'BMI', 'SmokingStatus', 'PhysicalActivity', 'ProteinConsumption',
    'EducationLevel', 'Diabetes', 'CalorieConsumption', 'FiberConsumption',
    'CarbohydrateConsumption', 'DrinkingStatus'
]

# ============ INPUT FORM (2‑COLUMN LAYOUT) ============
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="academic-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Clinical Assessment Form</div>', unsafe_allow_html=True)

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown('<div class="section-header">🩺 Demographic & Clinical</div>', unsafe_allow_html=True)
        age = st.slider("Age (years)", 18, 100, value=DEFAULTS['age'], key='age', help="Age in years")
        bmi = st.slider("BMI (kg/m²)", 15.0, 45.0, value=DEFAULTS['bmi'], step=0.5, key='bmi', help="Body Mass Index")
        bri = st.slider("BRI (Body Roundness Index)", 1.0, 8.0, value=DEFAULTS['bri'], step=0.1, key='bri')
        bri_trend = st.slider("BRI Trend", -2.0, 2.0, value=DEFAULTS['bri_trend'], step=0.1, key='bri_trend', help="Change over time")

    with col_l2:
        st.markdown('<div class="section-header">🏥 Medical History</div>', unsafe_allow_html=True)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"], index=0 if DEFAULTS['hypertension']=="No" else 1, key='hypertension')
        hyperlipidemia = st.selectbox("Hyperlipidemia", ["No", "Yes"], index=0 if DEFAULTS['hyperlipidemia']=="No" else 1, key='hyperlipidemia')
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=0 if DEFAULTS['diabetes']=="No" else 1, key='diabetes')
        st.markdown('<div class="section-header" style="margin-top:0.5rem;">🚬 Lifestyle</div>', unsafe_allow_html=True)
        smoking = st.selectbox("Smoking Status", ["No", "Yes"], index=0 if DEFAULTS['smoking']=="No" else 1, key='smoking')
        drinking = st.selectbox("Alcohol Consumption", ["No", "Yes"], index=0 if DEFAULTS['drinking']=="No" else 1, key='drinking')
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="academic-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💪 Health Status Assessment</div>', unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown('<div class="section-header">⚡ Lifestyle & Dietary</div>', unsafe_allow_html=True)
        physical_activity = st.slider("Physical Activity Level", 0.0, 10.0, value=DEFAULTS['physical_activity'], step=0.5, key='physical_activity',
                                      help="0=Sedentary → 10=Highly active")
        protein = st.slider("Protein Intake (g/day)", 30, 150, value=DEFAULTS['protein'], step=5, key='protein')
        calories = st.slider("Calorie Intake (kcal/day)", 1200, 3500, value=DEFAULTS['calories'], step=50, key='calories')

    with col_r2:
        st.markdown('<div class="section-header">🥗 Nutritional Factors</div>', unsafe_allow_html=True)
        fiber = st.slider("Fiber Intake (g/day)", 10, 60, value=DEFAULTS['fiber'], step=2, key='fiber')
        carbs = st.slider("Carbohydrate Intake (g/day)", 100, 500, value=DEFAULTS['carbs'], step=10, key='carbs')
        education = st.slider("Education Level", 1, 5, value=DEFAULTS['education'], step=1, key='education',
                              help="1=Less than high school → 5=Post graduate")

    # Clinical alerts based on BMI
    if bmi >= 30:
        st.markdown(f"""
        <div class="clinical-alert">
            ⚠️ <strong>Clinical Alert:</strong> Obesity detected (BMI = {bmi:.1f}).<br>
            Obesity is a significant risk factor for inflammatory arthritis.
        </div>
        """, unsafe_allow_html=True)
    elif bmi >= 25:
        st.markdown(f"""
        <div class="warning-alert">
            ⚠️ <strong>Clinical Note:</strong> Overweight (BMI = {bmi:.1f}).<br>
            Weight management may reduce inflammatory risk.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============ BUTTON SECTION ============
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_clicked = st.button("🔬 Analyze Rheumatoid Arthritis Risk", type="primary", use_container_width=True)

# ============ PREDICTION ============
if analyze_clicked:
    with st.spinner("🧠 Processing clinical data with XGBoost ensemble model..."):
        # Convert categoricals
        hypertension_val = 1 if hypertension == "Yes" else 0
        hyperlipidemia_val = 1 if hyperlipidemia == "Yes" else 0
        diabetes_val = 1 if diabetes == "Yes" else 0
        smoking_val = 1 if smoking == "Yes" else 0
        drinking_val = 1 if drinking == "Yes" else 0

        # Prepare feature array in the exact order the model expects
        features = np.array([[
            age, bri_trend, bri,
            hypertension_val, hyperlipidemia_val,
            bmi, smoking_val,
            physical_activity, protein,
            education, diabetes_val,
            calories, fiber, carbs,
            drinking_val
        ]])

        # Predict
        try:
            prob = model.predict_proba(features)[0][1]
        except Exception as e:
            # Fallback for compatibility
            if hasattr(model, 'use_label_encoder'):
                model.use_label_encoder = False
            prob = model.predict_proba(features)[0][1]

        risk_pct = prob * 100

        # Risk classification
        if prob < 0.3:
            level, color, advice, risk_class = (
                "Low Risk", COLORS['success'],
                "Continue healthy habits. Annual rheumatology screening recommended for patients with family history or other risk factors. Maintain regular follow-ups with primary care physician.",
                "Low"
            )
        elif prob < 0.7:
            level, color, advice, risk_class = (
                "Moderate Risk", COLORS['warning'],
                "Rheumatology consultation recommended within 3 months. Consider laboratory testing: RF, Anti-CCP, ESR, CRP. Monitor for joint pain, swelling, or morning stiffness >30 minutes. Smoking cessation strongly advised if applicable.",
                "Moderate"
            )
        else:
            level, color, advice, risk_class = (
                "High Risk", COLORS['danger'],
                "URGENT: Immediate rheumatology referral. Comprehensive autoimmune panel required (RF, Anti-CCP, ANA, ESR, CRP). Joint examination and imaging studies recommended. Consider early intervention with DMARDs if clinically indicated.",
                "High"
            )

        # Results display
        st.markdown(f"""
        <div class="result-container">
            <div class="risk-level" style="color:{color}">
                🎯 {level}
                <span class="status-badge status-{risk_class.lower()}">{risk_class}</span>
            </div>
            <div class="probability">
                Predicted Probability: <strong style="color:{color}; font-size:2rem;">{prob:.1%}</strong>
            </div>
            <div class="recommendation-box">
                <strong>📋 Clinical Recommendation:</strong><br>
                {advice}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            title=None,
            delta={'reference': 50, 'increasing': {'color': COLORS['danger']},
                   'decreasing': {'color': COLORS['success']}},
            number={'suffix': "%", 'font': {'size': 60, 'color': color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "#FFFFFF",
                'borderwidth': 2,
                'bordercolor': COLORS['border'],
                'steps': [
                    {'range': [0, 30], 'color': "#DCFCE7"},
                    {'range': [30, 70], 'color': "#FEF9C3"},
                    {'range': [70, 100], 'color': "#FEE2E2"}
                ],
                'threshold': {'line': {'color': COLORS['danger'], 'width': 4}, 'thickness': 0.75, 'value': 70}
            }
        ))
        fig.update_layout(height=400, margin=dict(t=60, b=40, l=40, r=40),
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Detailed factor analysis (4 metrics)
        st.markdown('<div class="academic-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Detailed Factor Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)

        age_risk = "High" if age > 60 else "Moderate" if age > 50 else "Low"
        age_color = COLORS['danger'] if age > 60 else COLORS['warning'] if age > 50 else COLORS['success']
        age_icon = "🔴" if age > 60 else "🟡" if age > 50 else "🟢"

        bmi_risk = "Obese" if bmi >= 30 else "Overweight" if bmi >= 25 else "Normal"
        bmi_color = COLORS['danger'] if bmi >= 30 else COLORS['warning'] if bmi >= 25 else COLORS['success']
        bmi_icon = "⚖️" if bmi >= 30 else "📊" if bmi >= 25 else "✓"

        smoking_risk = "Active Smoker" if smoking == "Yes" else "Non-smoker"
        smoking_color = COLORS['danger'] if smoking == "Yes" else COLORS['success']
        smoking_icon = "🚬" if smoking == "Yes" else "✓"

        htn_status = "Present" if hypertension == "Yes" else "Absent"
        htn_color = COLORS['warning'] if hypertension == "Yes" else COLORS['success']
        htn_icon = "❤️" if hypertension == "Yes" else "✓"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{age_icon} {age}</div>
                <div class="metric-label">Age (years)</div>
                <div class="metric-status" style="color:{age_color}">● {age_risk} Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{bmi_icon} {bmi:.1f}</div>
                <div class="metric-label">BMI (kg/m²)</div>
                <div class="metric-status" style="color:{bmi_color}">● {bmi_risk}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{smoking_icon} {smoking_risk}</div>
                <div class="metric-label">Smoking Status</div>
                <div class="metric-status" style="color:{smoking_color}">● {smoking_risk}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{htn_icon} {htn_status}</div>
                <div class="metric-label">Hypertension</div>
                <div class="metric-status" style="color:{htn_color}">● {htn_status}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Store results for PDF
        st.session_state.result = {
            "prob": prob,
            "level": level,
            "color": color,
            "advice": advice,
            "risk_pct": risk_pct,
            "date": datetime.now(),
            "inputs": {
                "Age": f"{age} years",
                "BMI": f"{bmi:.1f} kg/m²",
                "Body Roundness Index (BRI)": f"{bri:.1f}",
                "BRI Trend": f"{bri_trend:.1f}",
                "Hypertension": hypertension,
                "Hyperlipidemia": hyperlipidemia,
                "Diabetes": diabetes,
                "Smoking Status": smoking,
                "Alcohol Consumption": drinking,
                "Physical Activity": f"{physical_activity:.1f}/10",
                "Protein Intake": f"{protein} g/day",
                "Calorie Intake": f"{calories} kcal/day",
                "Fiber Intake": f"{fiber} g/day",
                "Carbohydrate Intake": f"{carbs} g/day",
                "Education Level": f"{education}/5"
            }
        }

# ============ PDF REPORT GENERATION ============
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def create_research_pdf():
    buffer = BytesIO()
    try:
        doc = SimpleDocTemplate(buffer, pagesize=landscape(A4),
                                leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=2 * cm, bottomMargin=2 * cm)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle(name="Title", fontSize=22, textColor=colors.HexColor(COLORS['primary']),
                                     spaceAfter=10, alignment=1, fontName='Helvetica-Bold')
        subtitle_style = ParagraphStyle(name="Subtitle", fontSize=11, textColor=colors.HexColor(COLORS['secondary']),
                                        alignment=1, spaceAfter=15, fontName='Helvetica')
        section_style = ParagraphStyle(name="Section", fontSize=13, textColor=colors.HexColor(COLORS['primary']),
                                       spaceAfter=8, spaceBefore=12, fontName='Helvetica-Bold')

        story.append(Paragraph("ArthroAI", title_style))
        story.append(Paragraph("Rheumatoid Arthritis Risk Assessment Report", subtitle_style))
        story.append(Paragraph(f"Report ID: ARA-{datetime.now().strftime('%Y%m%d%H%M%S')}", styles["Normal"]))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles["Normal"]))
        story.append(Spacer(1, 0.6 * cm))

        story.append(Paragraph("Clinical Risk Factors", section_style))
        data_items = list(st.session_state.result["inputs"].items())
        data = [["Parameter", "Value"]]
        for k, v in data_items:
            data.append([k, str(v)])
        table = Table(data, colWidths=[7 * cm, 6 * cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLORS['primary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor(COLORS['light_bg'])),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.6 * cm))

        story.append(Paragraph("Risk Assessment Results", section_style))
        story.append(Paragraph(
            f"<b>Risk Classification:</b> <font color='{st.session_state.result['color']}'><b>{st.session_state.result['level']}</b></font>",
            styles["Normal"]))
        story.append(
            Paragraph(f"<b>Probability Score:</b> <b>{st.session_state.result['prob']:.1%}</b>", styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("<b>Clinical Recommendation:</b>", styles["Normal"]))
        story.append(Paragraph(st.session_state.result["advice"],
                               ParagraphStyle(name="Advice", fontSize=10, leading=14, spaceAfter=15)))

        story.append(Paragraph(
            "<b>Disclaimer:</b> For research and educational purposes only. Not a substitute for professional medical advice.",
            ParagraphStyle(name="Disc", fontSize=8, textColor=colors.gray, alignment=1)))
        doc.build(story)
    except Exception as e:
        st.error(f"❌ PDF generation failed: {e}")
        return None
    return buffer.getvalue()


# ============ DOWNLOAD BUTTON ============
if "result" in st.session_state:
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    pdf_data = create_research_pdf()
    if pdf_data:
        b64 = base64.b64encode(pdf_data).decode()
        filename = f"ArthroAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.markdown(f'''
            <div style="text-align: center; margin: 0.5rem 0;">
                <a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration: none;">
                    <button style="background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['primary']}); 
                            color: white; padding: 0.8rem 2rem; border: none; border-radius: 12px; 
                            font-size: 1rem; cursor: pointer; font-weight: 600; width: 100%;
                            box-shadow: 0 2px 8px rgba(106,76,147,0.3);">
                        📄 Download Research Report (PDF)
                    </button>
                </a>
            </div>
            ''', unsafe_allow_html=True)

# ============ FOOTER ============
st.markdown(f"""
<div class="research-footer">
    <strong>ArthroAI</strong> | Rheumatoid Arthritis Risk Prediction System<br>
    <small>Powered by NHANES Data • XGBoost Model • Research Grade</small><br>
    <small>© 2025 • ArthroAI Research Initiative</small>
</div>
""", unsafe_allow_html=True)