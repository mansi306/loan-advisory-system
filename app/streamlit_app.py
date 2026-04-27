"""
streamlit_app.py
UI layer only — collects inputs, calls src/ functions, displays results.
No model logic here.
Run: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
from src.predict import predict_with_explanation
from src.recommend import generate_recommendations

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Loan Advisory System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Advisory System")
st.caption("AI-powered loan approval prediction with explainable decisions")

# ── Sidebar — input form ───────────────────────────────
with st.sidebar:
    st.header("Applicant Details")
    st.divider()

    gender       = st.selectbox("Gender", ["Male", "Female"])
    married      = st.selectbox("Marital status", ["Yes", "No"])
    dependents   = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education    = st.selectbox("Education", ["Graduate", "Not Graduate"])
    employed     = st.selectbox("Employment type", ["Salaried", "Self-Employed"])
    property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

    st.divider()

    app_income   = st.number_input("Applicant monthly income (₹)",
                                    min_value=0, value=5000, step=500)
    co_income    = st.number_input("Co-applicant monthly income (₹)",
                                    min_value=0, value=0, step=500)
    loan_amount  = st.number_input("Loan amount (₹000s)",
                                    min_value=1, value=120, step=10)
    loan_term    = st.selectbox("Loan term (months)",
                                 [60, 120, 180, 240, 300, 360], index=5)
    credit       = st.selectbox("Credit history", ["Yes", "No"])

    st.divider()
    submitted = st.button("Check Eligibility", use_container_width=True,
                           type="primary")

# ── Main panel ─────────────────────────────────────────
if not submitted:
    st.info("Fill in the applicant details in the sidebar and click "
            "**Check Eligibility** to get a prediction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "Random Forest")
    with col2:
        st.metric("Cross-val AUC", "0.749")
    with col3:
        st.metric("Training samples", "491")

else:
    # Build applicant dict
    applicant = {
        "Gender":           gender,
        "Married":          married,
        "Dependents":       dependents,
        "Education":        education,
        "Self_Employed":    "Yes" if employed == "Self-Employed" else "No",
        "ApplicantIncome":  app_income,
        "CoapplicantIncome": co_income,
        "LoanAmount":       loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History":   1 if credit == "Yes" else 0,
        "Property_Area":    property_area
    }

    # Run prediction
    with st.spinner("Analysing application..."):
        result = predict_with_explanation(applicant)
        recs   = generate_recommendations(applicant, result["factors"])

    prob     = result["probability"]
    approved = result["approved"]

    # ── Three tabs ─────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Prediction", "🔍 Why this decision?", "💡 How to improve"
    ])

    # ── Tab 1: Prediction result ───────────────────────
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                label="Approval Probability",
                value=f"{prob:.1%}"
            )
            if approved:
                st.success("✅ Likely Approved")
            elif prob > 0.4:
                st.warning("⚠️ Borderline — small improvements may help")
            else:
                st.error("❌ Likely Rejected")

        with col2:
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#1D9E75" if approved else "#D85A30"},
                    "steps": [
                        {"range": [0, 40],  "color": "#FAECE7"},
                        {"range": [40, 50], "color": "#FAEEDA"},
                        {"range": [50, 100],"color": "#E1F5EE"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 2},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            fig.update_layout(height=220, margin=dict(t=20, b=0, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: SHAP factor breakdown ───────────────────
    with tab2:
        st.subheader("What drove this decision")
        st.caption(
            "SHAP values show each feature's contribution. "
            "Negative = hurts approval chances. Positive = helps."
        )

        factors = result["factors"]
        names   = [f[0].split("__")[-1] for f in factors]
        values  = [f[1] for f in factors]
        colors  = ["#D85A30" if v < 0 else "#1D9E75" for v in values]

        fig2 = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside"
        ))
        fig2.update_layout(
            xaxis_title="SHAP value  (← hurts approval  |  helps approval →)",
            height=320,
            margin=dict(l=10, r=60, t=20, b=40),
            xaxis=dict(zeroline=True, zerolinewidth=1.5,
                       zerolinecolor="black"),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Recommendations ─────────────────────────
    with tab3:
        if approved and prob >= 0.6:
            st.success(
                "✅ This application is in strong shape. "
                "No significant improvement actions needed."
            )
        elif not recs:
            st.info("No actionable improvement factors identified.")
        else:
            st.subheader("Ranked improvement actions")
            st.caption("Ordered by impact — address the top item first.")

            impact_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

            for i, rec in enumerate(recs, 1):
                icon = impact_colors.get(rec["impact"], "⚪")
                with st.expander(
                    f"{i}. {rec['action']}  {icon} {rec['impact']} impact",
                    expanded=True
                ):
                    st.write(rec["detail"])