"""
Streamlit Frontend — Salary Predictor
======================================
Interactive UI that calls the FastAPI backend to predict salary.

Run with:
    streamlit run streamlit_app.py

Make sure the FastAPI backend is running first:
    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor | MLOps Capstone",
    page_icon="💼",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 2rem; }
    .salary-box {
        background: linear-gradient(135deg, #1e3a5f, #1e40af);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin: 16px 0;
        border: 1px solid #3b82f6;
        box-shadow: 0 8px 32px rgba(59,130,246,0.25);
    }
    .salary-label {
        font-size: 0.85rem;
        color: #94a3b8;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .salary-value {
        font-size: 3rem;
        font-weight: 800;
        color: #10b981;
        letter-spacing: -1px;
    }
    .salary-sub {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 6px;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #334155;
        text-align: center;
    }
    .stSlider > div > div > div > div { background: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown("# 💼 Salary Predictor")
st.markdown("**MLOps Certification Course — Capstone Project**")
st.markdown("Predict annual salary based on years of professional experience using a trained Linear Regression model.")
st.divider()

# ── Fetch model metadata ──────────────────────────────────
@st.cache_data(ttl=60)
def fetch_metadata():
    try:
        r = requests.get(f"{API_BASE}/metadata", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

metadata = fetch_metadata()

# Show model stats in columns
if metadata:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",    metadata.get("model_type", "—").replace("Regressor","").strip())
    c2.metric("R² Score", metadata.get("r2_score", "—"))
    c3.metric("RMSE",     f"${metadata.get('rmse', 0):,.0f}")
    c4.metric("Train Samples", metadata.get("training_samples", "—"))
    st.divider()

# ── Input section ─────────────────────────────────────────
st.subheader("Enter Years of Experience")

col_input, col_slider = st.columns([1, 2])

with col_input:
    years = st.number_input(
        "Years",
        min_value=0.1,
        max_value=50.0,
        value=5.0,
        step=0.1,
        format="%.1f",
        label_visibility="collapsed"
    )

with col_slider:
    years_slider = st.slider(
        "Slide to adjust",
        min_value=0.1,
        max_value=20.0,
        value=float(years),
        step=0.1,
        format="%.1f yrs",
        label_visibility="collapsed"
    )

# Keep number input and slider in sync
if years_slider != years:
    years = years_slider

# ── Predict button ────────────────────────────────────────
predict_clicked = st.button("🔮  Predict Salary", use_container_width=True, type="primary")

# ── Call API and show result ──────────────────────────────
if predict_clicked:
    with st.spinner("Calling prediction API..."):
        try:
            response = requests.post(
                f"{API_BASE}/predict",
                json={"years_experience": round(years, 1)},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                salary = data["predicted_salary"]

                st.markdown(f"""
                <div class="salary-box">
                    <div class="salary-label">Estimated Annual Salary</div>
                    <div class="salary-value">${salary:,.0f}</div>
                    <div class="salary-sub">
                        For <strong>{years:.1f} year{"s" if years != 1 else ""}</strong> of experience
                        &nbsp;·&nbsp; Model: {data.get("model_type", "LinearRegression")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif response.status_code == 422:
                st.error("⚠️ Invalid input — years of experience must be between 0.1 and 50.")
            else:
                st.error(f"API returned error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("""
**Cannot reach the backend API.**

Make sure the FastAPI server is running in a separate terminal:
```
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```
            """)
        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.divider()

# ── Salary curve chart ────────────────────────────────────
st.subheader("📈 Salary vs Experience Curve")
st.caption("Batch predictions across 0.5 – 15 years to visualise the model's salary curve.")

@st.cache_data(ttl=300)
def build_salary_curve():
    """Fetch predictions for a range of experience values."""
    exp_range = np.arange(0.5, 15.5, 0.5)
    salaries  = []
    for exp in exp_range:
        try:
            r = requests.post(
                f"{API_BASE}/predict",
                json={"years_experience": round(float(exp), 1)},
                timeout=5
            )
            if r.status_code == 200:
                salaries.append(r.json()["predicted_salary"])
            else:
                salaries.append(None)
        except Exception:
            salaries.append(None)
    return exp_range, salaries

try:
    exp_vals, sal_vals = build_salary_curve()
    valid = [(e, s) for e, s in zip(exp_vals, sal_vals) if s is not None]

    if valid:
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")

        xs, ys = zip(*valid)
        ax.plot(xs, ys, color="#3b82f6", linewidth=2.5, label="Predicted salary")
        ax.fill_between(xs, ys, alpha=0.15, color="#3b82f6")

        # Mark current input
        try:
            r = requests.post(
                f"{API_BASE}/predict",
                json={"years_experience": round(float(years), 1)},
                timeout=5
            )
            if r.status_code == 200:
                mark_sal = r.json()["predicted_salary"]
                ax.scatter([years], [mark_sal], color="#10b981", s=120,
                           zorder=5, label=f"Your input ({years:.1f} yrs → ${mark_sal:,.0f})")
                ax.axvline(x=years, color="#10b981", linestyle="--", alpha=0.4, linewidth=1)
        except Exception:
            pass

        ax.set_xlabel("Years of Experience", color="#94a3b8")
        ax.set_ylabel("Predicted Salary (USD)", color="#94a3b8")
        ax.tick_params(colors="#64748b")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)
        ax.grid(alpha=0.15, color="#334155")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

except requests.exceptions.ConnectionError:
    st.info("Start the backend server to see the salary curve.")

# ── Batch predictor ───────────────────────────────────────
st.divider()
st.subheader("📋 Batch Predictor")
st.caption("Enter multiple experience values (comma-separated) to get predictions for all of them at once.")

batch_input = st.text_input(
    "Years of experience (comma-separated)",
    placeholder="e.g.  1, 2.5, 5, 7, 10, 15",
    label_visibility="visible"
)

if st.button("Run Batch Predictions", use_container_width=True):
    try:
        values = [float(v.strip()) for v in batch_input.split(",") if v.strip()]
        if not values:
            st.warning("Enter at least one value.")
        else:
            rows = []
            for v in values:
                try:
                    r = requests.post(
                        f"{API_BASE}/predict",
                        json={"years_experience": round(v, 1)},
                        timeout=5
                    )
                    if r.status_code == 200:
                        rows.append({
                            "Years of Experience": v,
                            "Predicted Salary (USD)": f"${r.json()['predicted_salary']:,.0f}"
                        })
                    else:
                        rows.append({"Years of Experience": v, "Predicted Salary (USD)": "Error"})
                except Exception as e:
                    rows.append({"Years of Experience": v, "Predicted Salary (USD)": str(e)})

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    except ValueError:
        st.error("Invalid input — use numbers separated by commas, e.g. `1, 2.5, 5`")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("MLOps Certification Course · Capstone Project · Rahul · Model: Linear Regression (R² = 0.9024)")
