"""
EconScope — Education Spending & Graduation Rate Analysis
═════════════════════════════════════════════════════════
A multi-page Streamlit app for econometric analysis.

Hypothesis: "Pure school spending does NOT increase graduation rate."

Data: Best_ECON_DATA_SET_ELSI_Export_.csv  (51 obs — 50 U.S. states + D.C.)

Pages:
  1. Data Explorer         — Browse and visualize the dataset
  2. Multiple Regression   — Variable selection, scatter matrix, correlation
  3. OLS Results           — Full OLS summary, regression equation, diagnostics
  4. Hypothesis Testing    — t-tests, F-test, p-values, final verdict
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import warnings, os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="EconScope — Education Spending Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700;800&family=Outfit:wght@300;400;500;600;700&display=swap');

.stApp { font-family: 'Outfit', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
}

/* Custom boxes */
.hypothesis-box {
    background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%);
    border-left: 5px solid #eab308;
    border-radius: 10px;
    padding: 22px 26px;
    margin: 16px 0;
}
.result-accept {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border-left: 5px solid #22c55e;
    border-radius: 10px;
    padding: 22px 26px;
    margin: 16px 0;
}
.result-reject {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 22px 26px;
    margin: 16px 0;
}
.equation-box {
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 14px;
    padding: 26px 34px;
    margin: 20px 0;
    font-family: 'Courier New', monospace;
    font-size: 1.05rem;
    text-align: center;
    letter-spacing: 0.02em;
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA LOADING & CLEANING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and clean the ECON dataset."""
    possible_paths = [
        "Best_ECON_DATA_SET_ELSI_Export_.csv",
        os.path.join(os.path.dirname(__file__), "Best_ECON_DATA_SET_ELSI_Export_.csv"),
        "/mnt/user-data/uploads/Best_ECON_DATA_SET_ELSI_Export_.csv",
        "/mnt/user-data/outputs/Best_ECON_DATA_SET_ELSI_Export_.csv",
    ]

    df = None
    source = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            source = path
            break

    if df is None:
        st.error("Could not find 'Best_ECON_DATA_SET_ELSI_Export_.csv'. "
                 "Place it in the same folder as app.py.")
        st.stop()

    # Drop empty rows
    df = df.dropna(how="all")
    df = df[df["State Name"].notna() & (df["State Name"].str.strip() != "")]

    # ── Clean currency / comma-formatted columns ──
    def clean_numeric(series):
        return (
            series.astype(str)
            .str.replace(r"[\$,\s]", "", regex=True)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )

    df["Median_Income"]  = clean_numeric(df["Median income (dollars)"])
    df["Mean_Income"]    = clean_numeric(df["Mean income (dollars)"])
    df["Salary_Teacher"] = clean_numeric(df["Salary_Teach"])
    df["Total_Pop"]      = clean_numeric(df["Total"])

    # Graduation rate -> percentage (original is 0-1 decimal)
    df["Grad_Rate_Pct"] = df["grad rate"] * 100
    df["Sample_Grad_Rate_Pct"] = df["sample grad rate"] * 100
    df["Poverty_Rate_Pct"] = df["Poverty Rate"] * 100

    # Rename for cleaner display
    df = df.rename(columns={
        "State Name": "State",
        "Spending": "Per_Pupil_Spending",
        "S/T_rate": "Student_Teacher_Ratio",
        "Schools#": "Num_Schools",
        "Diploma Recipients": "Diploma_Recipients",
        "Grades 9-12 Students [State] 2022-23": "Grades_9_12_Students",
        "Grade 12 Students": "Grade_12_Students",
    })

    # Final analysis columns
    analysis_cols = [
        "State", "Grad_Rate_Pct", "Per_Pupil_Spending", "Median_Income",
        "Mean_Income", "Poverty_Rate_Pct", "Salary_Teacher",
        "Student_Teacher_Ratio", "Num_Schools", "Diploma_Recipients",
        "Grades_9_12_Students", "Grade_12_Students", "Total_Pop",
        "Sample_Grad_Rate_Pct",
    ]
    df_clean = df[[c for c in analysis_cols if c in df.columns]].copy()
    df_clean = df_clean.dropna(subset=["Grad_Rate_Pct", "Per_Pupil_Spending"])

    return df_clean, df, source


df_clean, df_raw, data_source = load_data()
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Pretty labels
LABELS = {
    "Grad_Rate_Pct": "Graduation Rate (%)",
    "Per_Pupil_Spending": "Per-Pupil Spending ($)",
    "Median_Income": "Median Household Income ($)",
    "Mean_Income": "Mean Household Income ($)",
    "Poverty_Rate_Pct": "Poverty Rate (%)",
    "Salary_Teacher": "Avg Teacher Salary ($)",
    "Student_Teacher_Ratio": "Student-Teacher Ratio",
    "Num_Schools": "Number of Schools",
    "Diploma_Recipients": "Diploma Recipients",
    "Grades_9_12_Students": "Grades 9-12 Students",
    "Grade_12_Students": "Grade 12 Students",
    "Total_Pop": "Total Population",
    "Sample_Grad_Rate_Pct": "Sample Grad Rate (%)",
}

def pretty(col):
    return LABELS.get(col, col)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📊 EconScope")
    st.markdown("**Education Spending &  \nGraduation Rate Analysis**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🗂️ Data Explorer",
            "📈 Multiple Regression",
            "📋 OLS Results",
            "🧪 Hypothesis Testing",
        ],
        index=0,
    )
    st.markdown("---")
    st.caption(f"**Source:** ELSI Export — 50 States + D.C.")
    st.caption(f"**Observations:** {len(df_clean)}")
    st.caption(f"**Variables:** {len(df_clean.columns)}")
    st.markdown("---")
    st.markdown(
        """
        <div style='background:rgba(234,179,8,0.15);border-radius:10px;
        padding:16px;font-size:0.82rem;color:#fbbf24;line-height:1.6;'>
        <strong>H₀:</strong> Pure school spending does <em>not</em>
        increase graduation rate.<br>
        <strong>H₁:</strong> School spending <em>does</em>
        significantly affect graduation rate.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# PAGE 1: DATA EXPLORER
# ══════════════════════════════════════════════
if page == "🗂️ Data Explorer":
    st.markdown("# 🗂️ Data Explorer")
    st.markdown("Browse the ELSI education dataset for all 50 U.S. states and D.C.")

    # Key metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Avg Graduation Rate", f"{df_clean['Grad_Rate_Pct'].mean():.1f}%")
    with m2:
        st.metric("Avg Per-Pupil Spending", f"${df_clean['Per_Pupil_Spending'].mean():,.0f}")
    with m3:
        st.metric("Avg Median Income", f"${df_clean['Median_Income'].mean():,.0f}")
    with m4:
        st.metric("Avg Poverty Rate", f"{df_clean['Poverty_Rate_Pct'].mean():.1f}%")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📄 Raw Data", "📊 Summary Statistics", "📉 Distributions"])

    with tab1:
        st.dataframe(
            df_clean.style.format({
                "Grad_Rate_Pct": "{:.1f}%",
                "Per_Pupil_Spending": "${:,.0f}",
                "Median_Income": "${:,.0f}",
                "Mean_Income": "${:,.0f}",
                "Poverty_Rate_Pct": "{:.1f}%",
                "Salary_Teacher": "${:,.0f}",
                "Student_Teacher_Ratio": "{:.1f}",
            }),
            use_container_width=True, height=500,
        )

    with tab2:
        desc = df_clean[numeric_cols].describe().T
        desc.index = [pretty(c) for c in desc.index]
        st.dataframe(
            desc.style.format("{:.2f}").background_gradient(cmap="Blues", axis=1),
            use_container_width=True,
        )

    with tab3:
        sel_col = st.selectbox("Select a variable:", numeric_cols,
                               format_func=pretty, index=0)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df_clean, x=sel_col, nbins=20, marginal="box",
                color_discrete_sequence=["#6366f1"],
                title=f"Distribution of {pretty(sel_col)}",
                labels={sel_col: pretty(sel_col)},
            )
            fig.update_layout(template="plotly_white",
                              font=dict(family="Outfit"), title_font_size=16)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(
                df_clean.sort_values(sel_col, ascending=True),
                x=sel_col, y="State", orientation="h",
                color=sel_col, color_continuous_scale="Viridis",
                title=f"{pretty(sel_col)} by State",
                labels={sel_col: pretty(sel_col)},
                height=900,
            )
            fig2.update_layout(template="plotly_white",
                               font=dict(family="Outfit", size=10),
                               title_font_size=16, yaxis=dict(dtick=1))
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2: MULTIPLE REGRESSION ANALYSIS
# ══════════════════════════════════════════════
elif page == "📈 Multiple Regression":
    st.markdown("# 📈 Multiple Regression Analysis")
    st.markdown("Select variables, explore correlations, and visualize relationships in the data.")
    st.markdown("---")

    col_left, col_right = st.columns([1, 2])

    default_dep = "Grad_Rate_Pct"
    default_indep = [
        "Per_Pupil_Spending", "Median_Income", "Poverty_Rate_Pct",
        "Salary_Teacher", "Student_Teacher_Ratio",
    ]
    default_indep = [c for c in default_indep if c in numeric_cols]

    with col_left:
        st.markdown("### Variable Selection")
        dep_var = st.selectbox(
            "Dependent Variable (Y):", numeric_cols,
            index=numeric_cols.index(default_dep) if default_dep in numeric_cols else 0,
            format_func=pretty,
        )
        available_indep = [c for c in numeric_cols if c != dep_var]
        indep_vars = st.multiselect(
            "Independent Variables (X):", available_indep,
            default=[c for c in default_indep if c != dep_var],
            format_func=pretty,
        )
        if not indep_vars:
            st.warning("Select at least one independent variable.")
            st.stop()

        st.markdown("### Model Summary")
        st.markdown(f"**Y:** {pretty(dep_var)}")
        st.markdown(f"**X variables:** {len(indep_vars)}")
        st.markdown(f"**n =** {len(df_clean.dropna(subset=[dep_var] + indep_vars))}")

    with col_right:
        st.markdown("### Correlation Heatmap")
        corr_cols = [dep_var] + indep_vars
        corr_matrix = df_clean[corr_cols].corr()
        corr_labels = [pretty(c) for c in corr_cols]

        fig_corr = px.imshow(
            corr_matrix.values, text_auto=".2f",
            x=corr_labels, y=corr_labels,
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson Correlation Heatmap",
            aspect="auto",
        )
        fig_corr.update_layout(template="plotly_white",
                               font=dict(family="Outfit"),
                               title_font_size=16, width=700, height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # Scatter plots
    st.markdown("### Scatter Plots — Graduation Rate vs. Each Predictor")
    n_vars = len(indep_vars)
    cols_per_row = min(3, n_vars)
    rows = (n_vars + cols_per_row - 1) // cols_per_row

    for ri in range(rows):
        cols = st.columns(cols_per_row)
        for ci in range(cols_per_row):
            idx = ri * cols_per_row + ci
            if idx < n_vars:
                x_var = indep_vars[idx]
                with cols[ci]:
                    fig_sc = px.scatter(
                        df_clean, x=x_var, y=dep_var,
                        hover_data=["State"],
                        trendline="ols",
                        color_discrete_sequence=["#6366f1"],
                        title=f"{pretty(dep_var)} vs {pretty(x_var)}",
                        labels={x_var: pretty(x_var), dep_var: pretty(dep_var)},
                    )
                    fig_sc.update_traces(marker=dict(size=8, opacity=0.7))
                    fig_sc.update_layout(template="plotly_white",
                                         font=dict(family="Outfit", size=11),
                                         title_font_size=13, height=370)
                    st.plotly_chart(fig_sc, use_container_width=True)

    # Correlation bar
    st.markdown("### Correlation with Dependent Variable")
    corr_y = df_clean[indep_vars].corrwith(df_clean[dep_var]).sort_values(ascending=True)
    corr_y.index = [pretty(c) for c in corr_y.index]
    fig_bar = px.bar(
        x=corr_y.values, y=corr_y.index, orientation="h",
        color=corr_y.values, color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title=f"Correlation of Each Predictor with {pretty(dep_var)}",
        labels={"x": "Pearson r", "y": ""},
    )
    fig_bar.update_layout(template="plotly_white", font=dict(family="Outfit"),
                          title_font_size=16, height=max(320, 55 * len(indep_vars)))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Persist selections
    st.session_state["dep_var"] = dep_var
    st.session_state["indep_vars"] = indep_vars


# ══════════════════════════════════════════════
# PAGE 3: OLS RESULTS & REGRESSION EQUATION
# ══════════════════════════════════════════════
elif page == "📋 OLS Results":
    st.markdown("# 📋 OLS Regression Results")
    st.markdown("Full Ordinary Least Squares summary, regression equation, and diagnostics.")

    dep_var = st.session_state.get("dep_var", "Grad_Rate_Pct")
    indep_vars = st.session_state.get("indep_vars", None)
    if indep_vars is None:
        indep_vars = [
            "Per_Pupil_Spending", "Median_Income", "Poverty_Rate_Pct",
            "Salary_Teacher", "Student_Teacher_Ratio",
        ]
        indep_vars = [c for c in indep_vars if c in numeric_cols and c != dep_var]
        st.info("Using default variables. Visit **Multiple Regression** to customize.")

    # Fit OLS
    subset = df_clean[[dep_var] + indep_vars].dropna()
    Y = subset[dep_var]
    X = subset[indep_vars]
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const).fit()

    st.session_state["ols_model"] = model
    st.session_state["dep_var"] = dep_var
    st.session_state["indep_vars"] = indep_vars

    st.markdown("---")

    # ── Regression Equation ──
    st.markdown("### Regression Equation")
    coefs = model.params
    eq = f"{pretty(dep_var)} = {coefs['const']:.4f}"
    for var in indep_vars:
        sign = "+" if coefs[var] >= 0 else "−"
        eq += f"  {sign} {abs(coefs[var]):.6f} × {pretty(var)}"
    st.markdown(f'<div class="equation-box">{eq}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Key metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("R²", f"{model.rsquared:.4f}")
    with m2:
        st.metric("Adjusted R²", f"{model.rsquared_adj:.4f}")
    with m3:
        st.metric("F-statistic", f"{model.fvalue:.2f}")
    with m4:
        st.metric("Prob (F-stat)", f"{model.f_pvalue:.2e}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["📝 OLS Summary Table", "📊 Coefficient Plot", "🔍 Diagnostics"])

    with tab1:
        st.markdown("### Full OLS Regression Summary")
        st.code(model.summary().as_text(), language=None)

        st.markdown("### Coefficient Details")
        coef_df = pd.DataFrame({
            "Variable": [pretty(v) for v in model.params.index],
            "Coefficient": model.params.values,
            "Std Error": model.bse.values,
            "t-value": model.tvalues.values,
            "p-value": model.pvalues.values,
            "95% CI Lower": model.conf_int()[0].values,
            "95% CI Upper": model.conf_int()[1].values,
        })
        coef_df["Significant (α=0.05)"] = coef_df["p-value"].apply(
            lambda p: "✅ Yes" if p < 0.05 else "❌ No"
        )
        st.dataframe(
            coef_df.style.format({
                "Coefficient": "{:.6f}", "Std Error": "{:.6f}",
                "t-value": "{:.3f}", "p-value": "{:.4e}",
                "95% CI Lower": "{:.6f}", "95% CI Upper": "{:.6f}",
            }).background_gradient(subset=["p-value"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

    with tab2:
        st.markdown("### Coefficient Plot with 95% Confidence Intervals")
        cp = coef_df[coef_df["Variable"] != pretty("const")].copy()
        cp = cp.sort_values("Coefficient")
        colors = ["#ef4444" if p >= 0.05 else "#22c55e" for p in cp["p-value"]]

        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=cp["Coefficient"], y=cp["Variable"], mode="markers",
            marker=dict(size=13, color=colors, line=dict(width=1.5, color="white")),
            error_x=dict(
                type="data", symmetric=False,
                array=cp["95% CI Upper"] - cp["Coefficient"],
                arrayminus=cp["Coefficient"] - cp["95% CI Lower"],
                color="#94a3b8", thickness=2,
            ),
        ))
        fig_c.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
        fig_c.update_layout(
            template="plotly_white", font=dict(family="Outfit"),
            title="🟢 Significant (p < 0.05)  |  🔴 Not Significant",
            title_font_size=14,
            height=max(350, 65 * len(cp)), showlegend=False,
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with tab3:
        st.markdown("### Diagnostic Plots")
        resid = model.resid
        fitted = model.fittedvalues

        d1, d2 = st.columns(2)
        with d1:
            fig_rvf = px.scatter(
                x=fitted, y=resid,
                labels={"x": "Fitted Values", "y": "Residuals"},
                title="Residuals vs Fitted", color_discrete_sequence=["#6366f1"],
            )
            fig_rvf.add_hline(y=0, line_dash="dash", line_color="#ef4444")
            fig_rvf.update_layout(template="plotly_white",
                                  font=dict(family="Outfit"), height=400)
            st.plotly_chart(fig_rvf, use_container_width=True)

        with d2:
            sorted_r = np.sort(resid)
            theo_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_r)))
            fig_qq = px.scatter(
                x=theo_q, y=sorted_r,
                labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
                title="Normal Q-Q Plot", color_discrete_sequence=["#f59e0b"],
            )
            mn = min(theo_q.min(), sorted_r.min())
            mx = max(theo_q.max(), sorted_r.max())
            fig_qq.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                        line=dict(dash="dash", color="#ef4444"),
                                        showlegend=False))
            fig_qq.update_layout(template="plotly_white",
                                 font=dict(family="Outfit"), height=400)
            st.plotly_chart(fig_qq, use_container_width=True)

        d3, d4 = st.columns(2)
        with d3:
            fig_rh = px.histogram(
                x=resid, nbins=18, marginal="rug",
                labels={"x": "Residuals"},
                title="Distribution of Residuals",
                color_discrete_sequence=["#10b981"],
            )
            fig_rh.update_layout(template="plotly_white",
                                 font=dict(family="Outfit"), height=400)
            st.plotly_chart(fig_rh, use_container_width=True)

        with d4:
            fig_ap = px.scatter(
                x=Y.values, y=fitted.values,
                labels={"x": f"Actual {pretty(dep_var)}",
                        "y": f"Predicted {pretty(dep_var)}"},
                title="Actual vs Predicted",
                color_discrete_sequence=["#8b5cf6"],
            )
            vals = np.concatenate([Y.values, fitted.values])
            fig_ap.add_trace(go.Scatter(
                x=[vals.min(), vals.max()], y=[vals.min(), vals.max()],
                mode="lines", line=dict(dash="dash", color="#ef4444"),
                showlegend=False))
            fig_ap.update_layout(template="plotly_white",
                                 font=dict(family="Outfit"), height=400)
            st.plotly_chart(fig_ap, use_container_width=True)

        # Diagnostic stats
        st.markdown("### Statistical Diagnostic Tests")
        dc1, dc2, dc3 = st.columns(3)
        dw = durbin_watson(resid)
        with dc1:
            st.metric("Durbin-Watson", f"{dw:.3f}")
            st.caption("≈ 2 → no autocorrelation")
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_const)
            with dc2:
                st.metric("Breusch-Pagan p-value", f"{bp_p:.4f}")
                st.caption("p > 0.05 → homoscedasticity")
        except Exception:
            with dc2:
                st.metric("Breusch-Pagan", "N/A")
        jb_stat, jb_p = stats.jarque_bera(resid)
        with dc3:
            st.metric("Jarque-Bera p-value", f"{jb_p:.4f}")
            st.caption("p > 0.05 → residuals are normal")


# ══════════════════════════════════════════════
# PAGE 4: HYPOTHESIS TESTING & CONCLUSION
# ══════════════════════════════════════════════
elif page == "🧪 Hypothesis Testing":
    st.markdown("# 🧪 Hypothesis Testing & Conclusion")

    # ── Hypothesis statement ──
    st.markdown('<div class="hypothesis-box">', unsafe_allow_html=True)
    st.markdown("""
    **Research Question:** Does pure school spending increase graduation rates?

    **Null Hypothesis (H₀):** Pure school spending does **not** increase graduation rate.
    Formally: β_Per_Pupil_Spending = 0

    **Alternative Hypothesis (H₁):** School spending **does** significantly affect graduation rate.
    Formally: β_Per_Pupil_Spending ≠ 0

    **Significance Level:** α = 0.05
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Retrieve or fit model
    dep_var = st.session_state.get("dep_var", "Grad_Rate_Pct")
    indep_vars = st.session_state.get("indep_vars", None)
    model = st.session_state.get("ols_model", None)

    if indep_vars is None:
        indep_vars = [
            "Per_Pupil_Spending", "Median_Income", "Poverty_Rate_Pct",
            "Salary_Teacher", "Student_Teacher_Ratio",
        ]
        indep_vars = [c for c in indep_vars if c in numeric_cols and c != dep_var]

    if model is None:
        subset = df_clean[[dep_var] + indep_vars].dropna()
        Y = subset[dep_var]
        X_const = sm.add_constant(subset[indep_vars])
        model = sm.OLS(Y, X_const).fit()
        st.session_state["ols_model"] = model

    # ── Identify spending variable ──
    spending_var = None
    for v in model.params.index:
        if "spend" in v.lower():
            spending_var = v
            break
    if spending_var is None:
        spending_var = [v for v in model.params.index if v != "const"][0]

    alpha = 0.05
    beta = model.params[spending_var]
    se = model.bse[spending_var]
    t_val = model.tvalues[spending_var]
    p_val = model.pvalues[spending_var]
    ci_low, ci_high = model.conf_int().loc[spending_var]
    dof = model.df_resid
    t_crit = stats.t.ppf(1 - alpha / 2, dof)

    # ═══════════════════════════════════════
    # 1. INDIVIDUAL t-TEST
    # ═══════════════════════════════════════
    st.markdown("## 1️⃣  Individual t-Test on Per-Pupil Spending")

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        st.metric("β (Spending)", f"{beta:.6f}")
    with tc2:
        st.metric("t-statistic", f"{t_val:.3f}")
    with tc3:
        st.metric("p-value", f"{p_val:.4e}")
    with tc4:
        st.metric("95% CI", f"[{ci_low:.6f}, {ci_high:.6f}]")

    # t-distribution visual
    x_r = np.linspace(-4.5, 4.5, 600)
    t_y = stats.t.pdf(x_r, dof)

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=x_r, y=t_y, mode="lines",
        line=dict(color="#6366f1", width=2.5), name="t-distribution",
    ))
    xl = x_r[x_r <= -t_crit]
    xr = x_r[x_r >= t_crit]
    fig_t.add_trace(go.Scatter(
        x=np.concatenate([xl, xl[::-1]]),
        y=np.concatenate([stats.t.pdf(xl, dof), np.zeros(len(xl))]),
        fill="toself", fillcolor="rgba(239,68,68,0.25)",
        line=dict(color="rgba(239,68,68,0)"), name="Rejection Region",
    ))
    fig_t.add_trace(go.Scatter(
        x=np.concatenate([xr, xr[::-1]]),
        y=np.concatenate([stats.t.pdf(xr, dof), np.zeros(len(xr))]),
        fill="toself", fillcolor="rgba(239,68,68,0.25)",
        line=dict(color="rgba(239,68,68,0)"), showlegend=False,
    ))
    fig_t.add_vline(x=t_val, line_dash="dash", line_color="#eab308",
                    annotation_text=f"t = {t_val:.3f}",
                    annotation_position="top right",
                    annotation_font_color="#eab308")
    fig_t.add_vline(x=-t_crit, line_dash="dot", line_color="#94a3b8")
    fig_t.add_vline(x=t_crit, line_dash="dot", line_color="#94a3b8")
    fig_t.update_layout(
        template="plotly_white", font=dict(family="Outfit"),
        title=f"t-Test for Per-Pupil Spending  (df = {int(dof)}, α = {alpha})",
        xaxis_title="t-value", yaxis_title="Density",
        height=420, showlegend=True,
    )
    st.plotly_chart(fig_t, use_container_width=True)

    if p_val < alpha:
        st.markdown(
            f"""<div class="result-reject">
            <strong>Result:</strong> p-value ({p_val:.4e}) &lt; α = {alpha}<br>
            |t| = {abs(t_val):.3f} &gt; t-critical = {t_crit:.3f}<br><br>
            <strong>→ REJECT H₀.</strong> There is statistically significant evidence that
            per-pupil spending affects graduation rate at the 5% level.
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""<div class="result-accept">
            <strong>Result:</strong> p-value ({p_val:.4e}) &gt; α = {alpha}<br>
            |t| = {abs(t_val):.3f} ≤ t-critical = {t_crit:.3f}<br><br>
            <strong>→ FAIL TO REJECT H₀.</strong> There is insufficient evidence that
            per-pupil spending alone significantly affects graduation rate at the 5% level.
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════
    # 2. OVERALL F-TEST
    # ═══════════════════════════════════════
    st.markdown("## 2️⃣  Overall F-Test (Joint Model Significance)")

    fc1, fc2, fc3 = st.columns(3)
    f_crit = stats.f.ppf(1 - alpha, model.df_model, model.df_resid)
    with fc1:
        st.metric("F-statistic", f"{model.fvalue:.3f}")
    with fc2:
        st.metric("Prob (F-stat)", f"{model.f_pvalue:.2e}")
    with fc3:
        st.metric("F-critical (α=0.05)", f"{f_crit:.3f}")

    if model.f_pvalue < alpha:
        st.success(
            f"The overall model is significant (F = {model.fvalue:.3f}, "
            f"p = {model.f_pvalue:.2e}). At least one predictor meaningfully "
            f"explains variation in {pretty(dep_var)}.")
    else:
        st.warning(
            f"The overall model is NOT significant (F = {model.fvalue:.3f}, "
            f"p = {model.f_pvalue:.2e}).")

    st.markdown("---")

    # ═══════════════════════════════════════
    # 3. ALL COEFFICIENTS
    # ═══════════════════════════════════════
    st.markdown("## 3️⃣  All Coefficients — Significance Summary")

    sig_df = pd.DataFrame({
        "Variable": [pretty(v) for v in model.params.index],
        "Coefficient": model.params.values,
        "t-value": model.tvalues.values,
        "p-value": model.pvalues.values,
        "Significant?": ["✅ Yes" if p < 0.05 else "❌ No" for p in model.pvalues],
    })
    st.dataframe(
        sig_df.style.format({
            "Coefficient": "{:.6f}", "t-value": "{:.3f}", "p-value": "{:.4e}",
        }).applymap(
            lambda v: "background-color: #dcfce7" if v == "✅ Yes"
            else ("background-color: #fee2e2" if v == "❌ No" else ""),
            subset=["Significant?"],
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # ═══════════════════════════════════════
    # 4. FINAL CONCLUSION
    # ═══════════════════════════════════════
    st.markdown("## 📌 Final Conclusion")

    spending_sig = p_val < alpha
    model_sig = model.f_pvalue < alpha

    st.markdown(f"""
    | Test | Statistic | p-value | Decision |
    |------|-----------|---------|----------|
    | **t-test (Per-Pupil Spending)** | t = {t_val:.3f} | {p_val:.4e} | {"Reject H₀" if spending_sig else "Fail to Reject H₀"} |
    | **F-test (Overall Model)** | F = {model.fvalue:.3f} | {model.f_pvalue:.2e} | {"Significant" if model_sig else "Not Significant"} |
    | **R²** | {model.rsquared:.4f} | — | Model explains {model.rsquared*100:.1f}% of variance |
    | **Adj. R²** | {model.rsquared_adj:.4f} | — | Adjusted for {len(indep_vars)} predictors |
    """)

    if spending_sig:
        st.markdown(
            f"""<div class="result-reject">
            <h3>🔴 We REJECT the Null Hypothesis</h3>
            <p>At α = 0.05, the data provides sufficient evidence to <strong>reject</strong>
            the hypothesis that pure school spending does not increase graduation rate.</p>
            <p>The coefficient on <strong>Per-Pupil Spending</strong> is <strong>{beta:.6f}</strong>
            (t = {t_val:.3f}, p = {p_val:.4e}), meaning each additional dollar in per-pupil spending
            is associated with a <strong>{beta:.6f}</strong> percentage-point change in graduation rate,
            holding all other variables constant.</p>
            <p>The full model explains <strong>{model.rsquared*100:.1f}%</strong> of the variance
            in graduation rate (R² = {model.rsquared:.4f}), confirming that spending — along with
            other socioeconomic factors — plays a statistically detectable role.</p>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""<div class="result-accept">
            <h3>🟢 We FAIL TO REJECT the Null Hypothesis</h3>
            <p>At α = 0.05, the data does <strong>not</strong> provide sufficient evidence to reject
            the hypothesis that pure school spending does not increase graduation rate.</p>
            <p>The coefficient on <strong>Per-Pupil Spending</strong> is <strong>{beta:.6f}</strong>
            (t = {t_val:.3f}, p = {p_val:.4e}), which is <strong>not statistically significant</strong>.</p>
            <p>This does <em>not</em> mean spending has zero effect — it means that after controlling
            for other socioeconomic variables (median income, poverty rate, teacher salary,
            student-teacher ratio), the <em>independent marginal effect</em> of spending alone
            is not statistically distinguishable from zero in this dataset of {len(df_clean)}
            observations (50 states + D.C.).</p>
            <p>The overall model explains <strong>{model.rsquared*100:.1f}%</strong> of variance
            (R² = {model.rsquared:.4f}), meaning other factors like poverty rate and median income
            are likely the primary drivers of graduation rate differences across states.</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "**Note:** This analysis uses OLS regression on cross-sectional state-level data (ELSI Export, 2022-23). "
        "Correlation does not imply causation. Results depend on model specification, "
        "data quality, and the assumption that relevant confounders are included."
    )
