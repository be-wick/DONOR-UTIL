from __future__ import annotations

# -----------------------------
# Imports
# -----------------------------
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Optional deps
HAS_STATSMODELS = True
HAS_SKLEARN = True

try:
    import statsmodels.api as sm
except Exception:
    HAS_STATSMODELS = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except Exception:
    HAS_SKLEARN = False

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="DONOR-UTIL TEST", layout="wide")

# =============================
# Page navigation
# =============================
page = st.sidebar.radio(
    "Navigation page",
    [
        "Home",
        "Cohort Builder",
        "Logistic Regression",
        "Propensity Score Matching",
        "Physiology Comparisons",
        "Export",
    ],
)
HEMO_ANALYTES = {
    "BPDiastolic": "hemo_BPDiastolic",
    "BPSystolic": "hemo_BPSystolic",
    "HeartRate": "hemo_HeartRate",
    "Temperature": "hemo_Temperature",
    "UrineOutput": "hemo_UrineOutput",
}

HEMO_METRICS = [
    "last",
    "min",
    "max",
    "mean",
    "auc",
    "slope_per_hr",
    "n",
]

# =============================
# Data classes
# =============================
@dataclass
class FilterRule:
    category: str
    col: str
    op: str
    a: float | str
    b: float | str | None = None


# =============================
# Utilities
# =============================
def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _is_num(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _coerce_binary(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if _is_num(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) != 0).astype(int)
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["1", "true", "t", "yes", "y", "pos", "positive"]).astype(int)


def _build_threshold_flag(x: pd.Series, op: str, a: float, b: float | None = None) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if op == "<":
        return (x < a).astype(int)
    if op == "<=":
        return (x <= a).astype(int)
    if op == ">":
        return (x > a).astype(int)
    if op == ">=":
        return (x >= a).astype(int)
    if op == "between":
        hi = a if b is None else b
        lo, hi = (a, hi) if a <= hi else (hi, a)
        return ((x >= lo) & (x <= hi)).astype(int)
    raise ValueError(f"Unsupported operator: {op}")


def apply_rule(df: pd.DataFrame, rule: FilterRule) -> pd.Series:
    s = df[rule.col]

    # numeric
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if rule.op == "<":
            return x < float(rule.a)
        if rule.op == "<=":
            return x <= float(rule.a)
        if rule.op == ">":
            return x > float(rule.a)
        if rule.op == ">=":
            return x >= float(rule.a)
        if rule.op == "between":
            lo, hi = sorted([float(rule.a), float(rule.b)])
            return x.between(lo, hi)
        if rule.op == "is missing":
            return x.isna()
        if rule.op == "not missing":
            return ~x.isna()

    # categorical/text
    ss = s.astype(str)
    if rule.op == "in":
        vals = [v.strip() for v in str(rule.a).split(",") if v.strip()]
        return ss.isin(vals)
    if rule.op == "is missing":
        return s.isna()
    if rule.op == "not missing":
        return ~s.isna()

    raise ValueError(f"Unsupported rule: {rule}")


def apply_rules(df: pd.DataFrame, rules: List[FilterRule]) -> pd.DataFrame:
    if not rules:
        return df
    mask = pd.Series(True, index=df.index)
    for r in rules:
        if r.col in df.columns:
            mask &= apply_rule(df, r)
    return df.loc[mask].copy()

def build_hemo_map(df: pd.DataFrame) -> dict:
    """
    Builds:
    {
      HeartRate: {
        low: { last: col, min: col, ... },
        average: {...},
        high: {...}
      },
      BPSystolic: {...}
    }
    """
    hemo_struct = {}

    for c in df.columns:
        if not c.startswith("hemo_"):
            continue

        parts = c.replace("hemo_", "").split("_")
        if len(parts) < 3:
            continue

        analyte = parts[0]
        subtype = parts[1]
        metric = "_".join(parts[2:])

        hemo_struct.setdefault(analyte, {})
        hemo_struct[analyte].setdefault(subtype, {})
        hemo_struct[analyte][subtype][metric] = c

    return hemo_struct


def summarize_hemo_timeseries(
    df_long: pd.DataFrame,
    value_col: str,
    time_col: str = "time_sec",
    id_col: str = "patient_id",
    prefix: str = "hemo_",
) -> pd.DataFrame:
    """
    Summarize a hemodynamic time-series into donor-level features.
    """

    rows = []

    for pid, g in df_long.groupby(id_col):
        g = g.sort_values(time_col)
        x = pd.to_numeric(g[value_col], errors="coerce")
        t = pd.to_numeric(g[time_col], errors="coerce")

        if x.notna().sum() < 2:
            continue

        dt = t.diff().fillna(0)

        rows.append({
            id_col: pid,
            f"{prefix}{value_col}_n": int(x.notna().sum()),
            f"{prefix}{value_col}_first": float(x.iloc[0]),
            f"{prefix}{value_col}_last": float(x.iloc[-1]),
            f"{prefix}{value_col}_min": float(x.min()),
            f"{prefix}{value_col}_max": float(x.max()),
            f"{prefix}{value_col}_auc": float((x * dt).sum()),
            f"{prefix}{value_col}_slope_per_hr":
                float(np.polyfit(t / 3600, x, 1)[0]),
        })

    return pd.DataFrame(rows)

def available_hemo_columns(df, analyte_key):
    """
    Returns available metric columns for a given hemodynamic analyte
    """
    base = HEMO_ANALYTES[analyte_key]
    cols = {}

    for m in HEMO_METRICS:
        col = f"{base}_{m}"
        if col in df.columns:
            cols[m] = col

    return cols


# =============================
# Metric helpers
# =============================
def _pick_metric_cols(obj, prefix: str) -> dict[str, list[str]]:
    cols = list(obj.columns) if hasattr(obj, "columns") else list(obj)
    out: dict[str, list[str]] = {}
    for c in cols:
        if isinstance(c, str) and c.startswith(prefix):
            base = c[len(prefix) :].split("_")[0]
            out.setdefault(base, []).append(c)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _metric_column(prefix: str, base: str, metric: str) -> str:
    return f"{prefix}{base}_{metric}"


def _lab_filter_ui(domain_name: str, prefix: str, var_map: dict[str, list[str]], df: pd.DataFrame, rules_key: str):
    st.subheader(domain_name)
    if not var_map:
        st.info(f"No {domain_name} columns found (prefix {prefix}).")
        return

    base = st.selectbox(f"{domain_name} analyte", options=list(var_map.keys()), key=f"{rules_key}_{domain_name}_base")
    metric_options = []
    for m in ["min", "max", "first", "last", "auc", "n", "slope_per_hr"]:
        col = _metric_column(prefix, base, m)
        if col in df.columns:
            metric_options.append(m)

    if not metric_options:
        st.info("No metrics found for that analyte.")
        return

    metric = st.selectbox("Metric", options=metric_options, key=f"{rules_key}_{domain_name}_metric")
    col = _metric_column(prefix, base, metric)

    op = st.selectbox(
        "Operator",
        options=["<", "<=", ">", ">=", "between", "is missing", "not missing"],
        key=f"{rules_key}_{domain_name}_op",
    )

    if op == "between":
        a = st.number_input("A", value=0.0, key=f"{rules_key}_{domain_name}_a")
        b = st.number_input("B", value=1.0, key=f"{rules_key}_{domain_name}_b")
        if st.button("Add filter", key=f"{rules_key}_{domain_name}_add"):
            st.session_state[rules_key].append(FilterRule(domain_name, col, "between", float(a), float(b)))
    elif op in ("is missing", "not missing"):
        if st.button("Add filter", key=f"{rules_key}_{domain_name}_add"):
            st.session_state[rules_key].append(FilterRule(domain_name, col, op, ""))
    else:
        a = st.number_input("Value", value=0.0, key=f"{rules_key}_{domain_name}_a")
        if st.button("Add filter", key=f"{rules_key}_{domain_name}_add"):
            st.session_state[rules_key].append(FilterRule(domain_name, col, op, float(a)))
            
def _hemo_filter_ui(domain_name: str, df: pd.DataFrame, rules_key: str):
    st.subheader(domain_name)

    # 1) Available analytes
    analytes = sorted({
        c.split("_")[1]
        for c in df.columns
        if isinstance(c, str) and c.startswith("hemo_")
    })

    if not analytes:
        st.info("No hemodynamic analytes found.")
        return

    analyte = st.selectbox(
        "Hemodynamic variable",
        analytes,
        key=f"{rules_key}_hemo_analyte",
    )

    # 2) Available bands for this analyte
    bands = sorted({
        c.split("_")[2]
        for c in df.columns
        if c.startswith(f"hemo_{analyte}_")
    })

    band = st.selectbox(
        "Aggregation band",
        bands,
        key=f"{rules_key}_hemo_band",
        help="Physiologic aggregation (low / average / high / total)",
    )

    # 3) Available metrics
    metrics = [
        m for m in HEMO_METRICS
        if f"hemo_{analyte}_{band}_{m}" in df.columns
    ]

    if not metrics:
        st.info("No metrics available for this selection.")
        return

    metric = st.selectbox(
        "Metric",
        metrics,
        key=f"{rules_key}_hemo_metric",
    )

    col = f"hemo_{analyte}_{band}_{metric}"

    # 4) Operator
    op = st.selectbox(
        "Operator",
        ["<", "<=", ">", ">=", "between", "is missing", "not missing"],
        key=f"{rules_key}_hemo_op",
    )

    if op == "between":
        a = st.number_input("A", value=0.0, key=f"{rules_key}_hemo_a")
        b = st.number_input("B", value=1.0, key=f"{rules_key}_hemo_b")
        if st.button("Add filter", key=f"{rules_key}_hemo_add"):
            st.session_state[rules_key].append(
                FilterRule("Hemodynamics", col, "between", float(a), float(b))
            )
    elif op in ("is missing", "not missing"):
        if st.button("Add filter", key=f"{rules_key}_hemo_add"):
            st.session_state[rules_key].append(
                FilterRule("Hemodynamics", col, op, "")
            )
    else:
        a = st.number_input("Value", value=0.0, key=f"{rules_key}_hemo_a")
        if st.button("Add filter", key=f"{rules_key}_hemo_add"):
            st.session_state[rules_key].append(
                FilterRule("Hemodynamics", col, op, float(a))
            )

def render_derived_metric_definitions():
    rows = []
    for k, v in DERIVED_METRIC_DEFINITIONS.items():
        rows.append({
            "Metric": k,
            "Description": v["definition"],
            "Clinical interpretation": v["clinical"],
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

# =============================
# PSM helpers
# =============================
def mean_and_n(x: pd.Series) -> tuple[float, int]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan, 0
    return float(x.mean()), int(len(x))
def _as_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def standardized_mean_diff(x_t: pd.Series, x_c: pd.Series) -> float:
    x_t = _as_numeric_series(x_t).dropna()
    x_c = _as_numeric_series(x_c).dropna()
    if len(x_t) < 2 or len(x_c) < 2:
        return float("nan")
    m1, m0 = x_t.mean(), x_c.mean()
    s1, s0 = x_t.std(ddof=1), x_c.std(ddof=1)
    sp = np.sqrt((s1**2 + s0**2) / 2)
    if sp == 0 or not np.isfinite(sp):
        return 0.0
    return float((m1 - m0) / sp)

def build_balance_table(pre_cov: pd.DataFrame, post_cov: pd.DataFrame, group_col="__group__") -> pd.DataFrame:
    cov_cols = [c for c in pre_cov.columns if c != group_col and c in post_cov.columns]

    rows = []
    for c in cov_cols:
        # PRE
        pre_t = pre_cov.loc[pre_cov[group_col] == 1, c]
        pre_c = pre_cov.loc[pre_cov[group_col] == 0, c]
        pre_mean_t, pre_n_t = mean_and_n(pre_t)
        pre_mean_c, pre_n_c = mean_and_n(pre_c)
        smd_pre = standardized_mean_diff(pre_t, pre_c)

        # POST
        post_t = post_cov.loc[post_cov[group_col] == 1, c]
        post_c = post_cov.loc[post_cov[group_col] == 0, c]
        post_mean_t, post_n_t = mean_and_n(post_t)
        post_mean_c, post_n_c = mean_and_n(post_c)
        smd_post = standardized_mean_diff(post_t, post_c)

        rows.append({
            "covariate": c,
            "pre_N_treated": pre_n_t,
            "pre_N_control": pre_n_c,
            "pre_mean_treated": pre_mean_t,
            "pre_mean_control": pre_mean_c,
            "SMD_pre": smd_pre,
            "post_N_treated": post_n_t,
            "post_N_control": post_n_c,
            "post_mean_treated": post_mean_t,
            "post_mean_control": post_mean_c,
            "SMD_post": smd_post,
            "abs_SMD_pre": abs(smd_pre) if pd.notna(smd_pre) else np.nan,
            "abs_SMD_post": abs(smd_post) if pd.notna(smd_post) else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("abs_SMD_pre", ascending=False)
    return out


# =============================
# Logistic regression helpers
# =============================
def collapse_rare_categories(s: pd.Series, min_n: int = 50) -> pd.Series:
    vc = s.value_counts(dropna=False)
    rare = vc[vc < min_n].index
    return s.where(~s.isin(rare), other="Other")


def fit_logistic_or(
    df_in: pd.DataFrame,
    y: pd.Series,
    exposure: pd.Series,
    covariates: List[str],
) -> pd.DataFrame:

    if not HAS_STATSMODELS:
        raise RuntimeError("statsmodels not installed")

    d = df_in.copy()

    # Outcome and exposure
    d["__y__"] = _coerce_binary(pd.Series(y)).astype(int)
    d["__x__"] = pd.to_numeric(pd.Series(exposure), errors="coerce")

    covs = [c for c in covariates if c in d.columns]
    d = d[["__y__", "__x__"] + covs].copy()
    
    # One-hot encode categoricals
    cat_cols = [
        c for c in covs
        if d[c].dtype == "object" or pd.api.types.is_categorical_dtype(d[c])
    ]
    for c in cat_cols:
        d[c] = collapse_rare_categories(d[c], min_n=50)

    if cat_cols:
        d = pd.get_dummies(d, columns=cat_cols, drop_first=True, dummy_na=True)
    

    # FORCE everything numeric (this is the critical line)
    d = d.apply(pd.to_numeric, errors="coerce")
    d = d.astype(float)


    # Drop rows with missing data
    d = d.dropna(axis=0)

    if len(d) < 50:
        raise ValueError(f"Too few complete rows after cleaning ({len(d)}).")

    if d["__y__"].nunique() < 2:
        raise ValueError("Outcome has no variation.")

    yv = d["__y__"].astype(int)
    X = d.drop(columns="__y__")

    # Drop constant columns
    X = X.loc[:, X.nunique() > 1]

    # Add intercept
    X = sm.add_constant(X, has_constant="add")

    # 🔍 FINAL SAFETY CHECK (optional but recommended)
    if not all(np.issubdtype(dt, np.number) for dt in X.dtypes):
        bad = X.dtypes[~X.dtypes.apply(lambda x: np.issubdtype(x, np.number))]
        raise ValueError(f"Non-numeric columns remain: {list(bad.index)}")
    if "cause_of_death_opo" in covariates and "opo" in covariates:
        raise ValueError("Do not include COD(OPO) and OPO together — collinear.")
    if "OPO" in covariates and "opo" in covariates:
        raise ValueError("Do not include COD(OPO) and OPO together — collinear.")
    if "cause_of_death_opo" in covariates and "OPO" in covariates:
        raise ValueError("Do not include COD(OPO) and OPO together — collinear.")
    model = sm.Logit(yv, X)
    fit = model.fit(disp=False)

    params = fit.params
    conf = fit.conf_int()
    conf.columns = ["CI_lower", "CI_upper"]

    or_table = pd.DataFrame({
        "variable": params.index,
        "OR": np.exp(params.values),
        "CI_lower": np.exp(conf["CI_lower"].values),
        "CI_upper": np.exp(conf["CI_upper"].values),
        "p_value": fit.pvalues.values,
    })
    
    return or_table.sort_values("OR", ascending=False)

DERIVED_METRIC_DEFINITIONS = {
    "n": {
        "label": "Count (n)",
        "definition": "Total number of recorded measurements for this variable during the donor management period.",
        "clinical": "Higher n often reflects longer donor management or more intensive monitoring."
    },
    "first": {
        "label": "First value",
        "definition": "The earliest recorded measurement after donor identification.",
        "clinical": "Represents initial physiologic state at the start of donor management."
    },
    "last": {
        "label": "Last value",
        "definition": "The final recorded measurement prior to organ recovery or donor cross-clamp.",
        "clinical": "Represents terminal physiologic status immediately before procurement."
    },
    "min": {
        "label": "Minimum",
        "definition": "The lowest recorded value during the donor management period.",
        "clinical": "Captures transient physiologic derangements (e.g., hypoxemia, hypotension)."
    },
    "max": {
        "label": "Maximum",
        "definition": "The highest recorded value during the donor management period.",
        "clinical": "Captures peaks such as hyperoxia, hyperglycemia, or catecholamine effects."
    },
    "auc": {
        "label": "Area under the curve (AUC)",
        "definition": "Time-weighted integral of values across the donor management period.",
        "clinical": "Reflects cumulative physiologic exposure rather than single-time measurements."
    },
    "slope_per_hr": {
        "label": "Slope per hour",
        "definition": "Linear rate of change per hour across all available measurements.",
        "clinical": "Indicates whether physiology is improving or worsening over time."
    },
}



# =============================
# Constants
# =============================
ORGAN_COLS = {
    "Heart": "outcome_heart",
    "Liver": "outcome_liver",
    "Kidney (Left)": "outcome_kidney_left",
    "Kidney (Right)": "outcome_kidney_right",
    "Lung (Left)": "outcome_lung_left",
    "Lung (Right)": "outcome_lung_right",
}


# =============================
# Data loading
# =============================
DEFAULT_PARQUET = Path("features_parquet/donor_analysis_latest.parquet")

@st.cache_data(show_spinner=True)
@st.cache_data(show_spinner=True)
def load_data(source) -> pd.DataFrame:
    df = pd.read_parquet(source)

    # Standardize patient_id
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str)

    # Normalize hemodynamic column naming
    HEMO_RENAME_MAP = {
        "hemo_BPDiastolic": "BPDiastolic",
        "hemo_BPSystolic": "BPSystolic",
        "hemo_HeartRate": "HeartRate",
        "hemo_Temperature": "Temperature",
        "hemo_UrineOutput": "UrineOutput",
    }

    new_cols = {}
    for col in df.columns:
        for base, analyte in HEMO_RENAME_MAP.items():
            if col.startswith(base + "_"):
                suffix = col.replace(base + "_", "")
                new_cols[col] = f"hemo_{analyte}_{suffix}"

    if new_cols:
        df = df.rename(columns=new_cols)

    return df

    # ============================================================
    # Normalize hemodynamic column naming (CRITICAL FIX)
    # ============================================================
    HEMO_RENAME_MAP = {
        "hemo_BPDiastolic": "BPDiastolic",
        "hemo_BPSystolic": "BPSystolic",
        "hemo_HeartRate": "HeartRate",
        "hemo_Temperature": "Temperature",
        "hemo_UrineOutput": "UrineOutput",
    }

    new_cols = {}

    for col in df.columns:
        for base, analyte in HEMO_RENAME_MAP.items():
            if col.startswith(base + "_"):
                suffix = col.replace(base + "_", "")
                new_cols[col] = f"hemo_{analyte}_{suffix}"

    if new_cols:
        df = df.rename(columns=new_cols)

    # Standardize patient_id
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str)

    return df

    


# =============================
# Session state
# =============================
for k in ["rules_A", "rules_B"]:
    if k not in st.session_state:
        st.session_state[k] = []

if "matched_df" not in st.session_state:
    st.session_state["matched_df"] = None

# ============================================================
# Cohort display names (user-defined)
# ============================================================
if "cohort_name_A" not in st.session_state:
    st.session_state["cohort_name_A"] = "Cohort A"

if "cohort_name_B" not in st.session_state:
    st.session_state["cohort_name_B"] = "Cohort B"



# =============================
# Header
# =============================
st.title("DONOR-UTIL vTEST")
st.caption("DONor Outcomes & Recovery - UTILization Analytics")


# =============================
# Sidebar — data source
# =============================
# =============================
# Sidebar — data source (URL-based, server-safe)


        


# -----------------------------
# Load parquet from URL (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_parquet_from_url(url: str) -> pd.DataFrame:
    import requests
    import io

    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    bio = io.BytesIO(r.content)
    df = pd.read_parquet(bio)

    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str)

    return df


# -----------------------------
# Execute load
# -----------------------------
# =============================
parquet_url = (
    "https://github.com/be-wick/DONOR-UTIL/releases/download/v1.0/donor_analysis_latest.parquet"
)
try:
    df = load_parquet_from_url(parquet_url.strip())
except Exception as e:
    st.sidebar.error(f"Failed to load parquet from URL: {e}")
    st.stop()

# Derived metric maps
abg_map = _pick_metric_cols(df, "abg_")
chem_map = _pick_metric_cols(df, "chem_")
cbc_map = _pick_metric_cols(df, "cbc_")
hemo_struct = build_hemo_map(df)


# =============================
# Build cohorts globally (available to all pages)
# =============================
cohort_A = apply_rules(df, st.session_state["rules_A"]).copy()
cohort_B = apply_rules(df, st.session_state["rules_B"]).copy()
cohort_A["__group__"] = 1
cohort_B["__group__"] = 0
cohort_A["__cohort_label__"] = st.session_state["cohort_name_A"]
cohort_B["__cohort_label__"] = st.session_state["cohort_name_B"]



with st.sidebar:
    st.markdown("### Cohort sizes")
    st.write(f"{st.session_state['cohort_name_A']}: {len(cohort_A):,}")
    st.write(f"{st.session_state['cohort_name_B']}: {len(cohort_B):,}")
with st.sidebar:
    st.subheader("Cohort names")

    st.session_state["cohort_name_A"] = st.text_input(
        "Name for Cohort A (treated)",
        value=st.session_state["cohort_name_A"],
        help="This name will be used throughout the app (tables, plots, downloads).",
    )

    st.session_state["cohort_name_B"] = st.text_input(
        "Name for Cohort B (control)",
        value=st.session_state["cohort_name_B"],
        help="This name will be used throughout the app (tables, plots, downloads).",
    )


# =============================
# Page — Home
# =============================
if page == "Home":
    st.markdown(
        """
Welcome to **DONor Outcomes & Recovery - UTILization Analytics**.

**DONOR-UTIL** is an analytics platform for evaluating donor physiology and organ utilization.
Use this tool to explore how physiologic parameters relate to procurement, discard, and transplantation outcomes across organs and centers.

> *DONOR-UTIL does not own any of the data analyzed.*
The data used is derived from the **Organ Retrieval and Collection of Health Information for Donation (ORCHID)** database on **PhysioNet**.

All data was **de-identified** in accordance with HIPAA standards.

**Before you begin**, please accept the data use agreement for the ORCHID database on PhysioNet:
"""
    )
    with st.container(border=True):
        st.page_link("https://physionet.org/content/orchid/2.1.1/", label="Accept Terms of Use")
    st.write("To get started, use the **left sidebar** to build a cohort and run analyses.")





# =============================
# Page — Cohort Builder
# =============================
elif page == "Cohort Builder":
    st.header("Cohort Builder")
    st.write("View a sample of patients from your designed cohorts. \n\nInformation on filter data and construction is also available at the bottom of this page.")
    with st.sidebar:
        st.subheader("Edit which cohort?")
        cohort_target = st.radio(
    "Editing cohort",
    [
        f"{st.session_state['cohort_name_A']} (treated)",
        f"{st.session_state['cohort_name_B']} (control)",
    ],
            horizontal=True,
)

        rules_key = (
    "rules_A"
            if cohort_target.startswith(st.session_state["cohort_name_A"])
            else "rules_B"
)


        cat = st.selectbox(
            "Add a filter",
            [
                "Demographics",
                "Organ outcomes",
                "ABG",
                "Chemistry",
                "CBC",
                "Hemodynamics",
                "Serology",
                "Cause of death",
            ],
            key=f"{rules_key}_cat",
        )
        

        # Demographics
        if cat == "Demographics":
            st.subheader("Demographics")

            if "age" in df.columns:
                age_lo, age_hi = st.slider("Age", 0, 100, (18, 75), key=f"{rules_key}_age")
                if st.button("Add age filter", key=f"{rules_key}_add_age"):
                    st.session_state[rules_key].append(FilterRule("Demographics", "age", "between", age_lo, age_hi))

            if "gender" in df.columns:
                vals = sorted(df["gender"].dropna().astype(str).unique())
                pick = st.multiselect("Gender", vals, key=f"{rules_key}_gender")
                if st.button("Add gender filter", key=f"{rules_key}_add_gender") and pick:
                    st.session_state[rules_key].append(FilterRule("Demographics", "gender", "in", ",".join(pick)))

            if "race" in df.columns:
                vals = sorted(df["race"].dropna().astype(str).unique())
                pick = st.multiselect("Race", vals, key=f"{rules_key}_race")
                if st.button("Add race filter", key=f"{rules_key}_add_race") and pick:
                    st.session_state[rules_key].append(FilterRule("Demographics", "race", "in", ",".join(pick)))
                    
                if "BMI" in df.columns:
                    bmi_vals = df["BMI"].dropna()
                    if len(bmi_vals) > 0:
                        bmi_min = float(np.floor(bmi_vals.min()))
                        bmi_max = float(np.ceil(bmi_vals.max()))

                        bmi_lo, bmi_hi = st.slider(
                            "BMI",
                            min_value=bmi_min,
                            max_value=bmi_max,
                            value=(18.5, 35.0),
                            step=0.5,
                            key=f"{rules_key}_bmi",
                            help="Body Mass Index (kg/m²)"
                        )

                        if st.button("Add BMI filter", key=f"{rules_key}_add_bmi"):
                            st.session_state[rules_key].append(
                                FilterRule("Demographics", "BMI", "between", bmi_lo, bmi_hi)
                            )

        # Organ outcomes
        elif cat == "Organ outcomes":
            st.subheader("Organ outcomes")
            organ_opts = {k: v for k, v in ORGAN_COLS.items() if v in df.columns}
            if not organ_opts:
                st.info("No organ outcome columns found.")
            else:
                organ = st.selectbox("Organ", list(organ_opts.keys()), key=f"{rules_key}_organ")
                col = organ_opts[organ]
                vals = sorted(df[col].dropna().astype(str).unique())
                pick = st.multiselect("Allowed values", vals, key=f"{rules_key}_organ_vals")
                if st.button("Add organ filter", key=f"{rules_key}_add_organ") and pick:
                    st.session_state[rules_key].append(FilterRule("Organ outcomes", col, "in", ",".join(pick)))
    
        # ABG / Chem / CBC / Hemo
        elif cat == "ABG":
            _lab_filter_ui("ABG", "abg_", abg_map, df, rules_key)
        elif cat == "Chemistry":
            _lab_filter_ui("Chemistry", "chem_", chem_map, df, rules_key)
        elif cat == "CBC":
            _lab_filter_ui("CBC", "cbc_", cbc_map, df, rules_key)
        elif cat == "Hemodynamics":
            st.subheader("Hemodynamics")

            if not hemo_struct:
                st.info("No hemodynamic summary columns found.")
                st.stop()

            analyte = st.selectbox(
                "Hemodynamic variable",
                sorted(hemo_struct.keys()),
                key=f"{rules_key}_hemo_analyte"
            )

            subtype = st.selectbox(
                "Subtype",
                sorted(hemo_struct[analyte].keys()),
                key=f"{rules_key}_hemo_subtype"
            )

            metric = st.selectbox(
                "Metric",
                sorted(hemo_struct[analyte][subtype].keys()),
                key=f"{rules_key}_hemo_metric"
            )

            col = hemo_struct[analyte][subtype][metric]

            op = st.selectbox(
                "Operator",
                ["<", "<=", ">", ">=", "between", "is missing", "not missing"],
                key=f"{rules_key}_hemo_op"
            )

            if op == "between":
                a = st.number_input("A", key=f"{rules_key}_hemo_a")
                b = st.number_input("B", key=f"{rules_key}_hemo_b")
                if st.button("Add filter", key=f"{rules_key}_hemo_add"):
                    st.session_state[rules_key].append(
                        FilterRule("Hemodynamics", col, "between", a, b)
                    )
            elif op in ("is missing", "not missing"):
                if st.button("Add filter", key=f"{rules_key}_hemo_add"):
                    st.session_state[rules_key].append(
                        FilterRule("Hemodynamics", col, op, "")
                    )
            else:
                a = st.number_input("Value", key=f"{rules_key}_hemo_a")
                if st.button("Add filter", key=f"{rules_key}_hemo_add"):
                    st.session_state[rules_key].append(
                        FilterRule("Hemodynamics", col, op, a)
                    )


        # Serology
        elif cat == "Serology":
            st.subheader("Serology")
            sero_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("sero_")]
            if not sero_cols:
                st.info("No serology columns found.")
            else:
                col = st.selectbox("Serology column", sero_cols, key=f"{rules_key}_sero_col")
                vals = sorted(df[col].dropna().astype(str).unique())
                pick = st.multiselect("Allowed values", vals, key=f"{rules_key}_sero_vals")
                if st.button("Add serology filter", key=f"{rules_key}_add_sero") and pick:
                    st.session_state[rules_key].append(FilterRule("Serology", col, "in", ",".join(pick)))

        # Cause of death
        elif cat == "Cause of death":
            st.subheader("Cause of death")
            cod_cols = [c for c in df.columns if isinstance(c, str) and ("cause" in c.lower() or "cod" in c.lower())]
            if not cod_cols:
                st.info("No cause-of-death columns found.")
            else:
                col = st.selectbox("Cause-of-death column", cod_cols, key=f"{rules_key}_cod_col")
                vals = sorted(df[col].dropna().astype(str).unique())
                pick = st.multiselect("Allowed values", vals, key=f"{rules_key}_cod_vals")
                if st.button("Add COD filter", key=f"{rules_key}_add_cod") and pick:
                    st.session_state[rules_key].append(FilterRule("Cause of death", col, "in", ",".join(pick)))
        
        st.divider()
        st.subheader(f"Active filters – {st.session_state['cohort_name_A']}")

        rules_A = list(st.session_state["rules_A"])  # 🔑 COPY, not reference

        if not rules_A:
            st.caption("No active filters.")
        else:
            for r in rules_A:
                c1, c2 = st.columns([8, 1])

                label = f"[{r.category}] {r.col} {r.op} {r.a}"
                if r.op == "between":
                    label += f"–{r.b}"

                c1.write(label)

                if c2.button("✕", key=f"remove_A_{id(r)}"):
                    st.session_state["rules_A"] = [
                        x for x in st.session_state["rules_A"] if x is not r
                    ]
                    st.rerun()

        # Clear button (UNIQUE key + rerun)
        if st.button(
            "Clear Cohort A",
            key="clear_cohort_A_button",
        ):
            st.session_state["rules_A"] = []
            st.rerun()
            
                    
        
        
        st.subheader(f"Active filters – {st.session_state['cohort_name_B']}")

        rules_B = list(st.session_state["rules_B"])

        if not rules_B:
            st.caption("No active filters.")
        else:
            for r in rules_B:
                c1, c2 = st.columns([8, 1])

                label = f"[{r.category}] {r.col} {r.op} {r.a}"
                if r.op == "between":
                    label += f"–{r.b}"

                c1.write(label)

                if c2.button("✕", key=f"remove_B_{id(r)}"):
                    st.session_state["rules_B"] = [
                        x for x in st.session_state["rules_B"] if x is not r
                    ]
                    st.rerun()

        if st.button(
            "Clear Cohort B",
            key="clear_cohort_B_button",
        ):
            st.session_state["rules_B"] = []
            st.rerun()

        
    
      
    st.subheader(f"Preview Cohort A: '{st.session_state['cohort_name_A']}'")
    st.dataframe(cohort_A.head(200), use_container_width=True)
    st.subheader(f"Preview Cohort B: '{st.session_state['cohort_name_B']}'")
    st.dataframe(cohort_B.head(200), use_container_width=True)
    st.subheader("Cohort Design")
    st.write("Filters use AND logic for stacking. OR logic is currently not available. To add a filter to a cohort, please be sure to select 'Add filter'. \n\nMany of the data in DONOR-UTIL are physiological lab values collected over time. To help with analysis, certain values were extracted and calculated from the data set. Currently, time-to-event data (i.e. labs 'x' hours before 'y' organ outcome) cannot be calculated.") 
    with st.expander("Derived variable definitions", expanded=True):
            st.caption(
                "These summary metrics are computed from raw time-series physiologic data "
                "during donor management."
            )
            render_derived_metric_definitions()

# =============================
# Page — Logistic Regression
# =============================
elif page == "Logistic Regression":
    st.header("Logistic Regression")

    if not HAS_STATSMODELS:
        st.error("statsmodels not installed. Install with: `python -m pip install statsmodels`")
        st.stop()

    cohort_choice = st.radio(
        "Select a cohort to analysis. For most analyses analyzing only a single cohort at a time is recommended.",
        [f"{st.session_state['cohort_name_A']}", f"{st.session_state['cohort_name_B']}", f" Both {st.session_state['cohort_name_A']} and {st.session_state['cohort_name_B']} (Prior to Matching)", "Matched (if available)"],
        horizontal=True,
    )

    if cohort_choice == "Cohort A":
        cohort = cohort_A.copy()
    elif cohort_choice == "Cohort B":
        cohort = cohort_B.copy()
    elif cohort_choice == "Matched (if available)":
        if st.session_state["matched_df"] is None:
            st.warning("No matched cohort yet. Run PSM first.")
            st.stop()
        cohort = st.session_state["matched_df"].copy()
    else:
        cohort = pd.concat([cohort_A, cohort_B], axis=0).copy()

    organ_opts = {k: v for k, v in ORGAN_COLS.items() if v in cohort.columns}
    if not organ_opts:
        st.info("No organ outcome columns found in selected cohort.")
        st.stop()

    organ_pick = st.selectbox("Organ outcome column", options=list(organ_opts.keys()))
    outcome_col = organ_opts[organ_pick]
    outcome_vals = sorted(cohort[outcome_col].dropna().astype(str).unique().tolist())
    if not outcome_vals:
        st.warning("No outcome values available in this cohort.")
        st.stop()

    pos_label = st.selectbox("Define POSITIVE as", options=outcome_vals, index=0)
    y = (cohort[outcome_col].astype(str).str.strip() == str(pos_label)).astype(int)

    numeric_cols = sorted([c for c in cohort.columns if c != "patient_id" and pd.api.types.is_numeric_dtype(cohort[c])])
    if not numeric_cols:
        st.warning("No numeric columns available for exposure.")
        st.stop()

    x_col = st.selectbox("Exposure (numeric column)", options=numeric_cols)

    exposure_mode = st.radio("Exposure type", ["Continuous", "Threshold flag (binary)"], horizontal=True)
    x = _safe_numeric(cohort[x_col])

    if exposure_mode == "Continuous":
        exposure = x
        exposure_label = f"{x_col} (continuous)"
    else:
        op = st.selectbox("Threshold operator", options=["<", "<=", ">", ">=", "between"])
        a = st.number_input("A", value=float(np.nanmedian(x.dropna())) if x.notna().any() else 0.0, step=0.1)
        b = st.number_input("B (between)", value=float(a + 1.0), step=0.1)
        exposure = _build_threshold_flag(x, op, float(a), float(b)).astype(int)
        exposure_label = f"{x_col} {op} {a}" + (f" and {b}" if op == "between" else "")

    cat_cols = sorted(
        [
            c for c in cohort.columns
            if c != "patient_id"
            and (pd.api.types.is_object_dtype(cohort[c]) or isinstance(cohort[c].dtype, pd.CategoricalDtype))
        ]
    )

    default_covs = [c for c in ["age", "gender", "race"] if c in cohort.columns]
    covariates = st.multiselect(
        "Covariates (adjusters)",
        options=sorted(set(default_covs + numeric_cols + cat_cols)),
        default=default_covs,
    )

    st.write(f"Outcome events: **{int(y.sum()):,} / {len(y):,} ({(y.mean()*100):.1f}%)**")
    st.write(f"Exposure: **{exposure_label}**")

    results_box = st.container(border=True)
    if st.button("Run regression"):
        try:
           
            analysis_df = cohort.copy()

            analysis_df["__y__"] = y.values
            analysis_df["__x__"] = pd.to_numeric(exposure, errors="coerce")

            model_cols = ["__x__"] + covariates

            X = analysis_df[model_cols].copy()

            # One-hot encode categoricals
            X = pd.get_dummies(
                X,
                drop_first=True,
                dummy_na=True
            )

            # Force numeric
            X = X.apply(pd.to_numeric, errors="coerce")

            # Drop rows with any missing values
            valid = X.notna().all(axis=1) & analysis_df["__y__"].notna()

            X = X.loc[valid]
            y_clean = analysis_df.loc[valid, "__y__"].astype(int)

            res = fit_logistic_or(
                df_in=analysis_df.loc[valid],
                y=y_clean,
                exposure=analysis_df.loc[valid, "__x__"],
                covariates=covariates,
)

        except Exception as e:
            st.error(f"Regression failed: {e}")
            st.stop()

        with results_box:
            st.markdown("## Results")
            st.dataframe(res, use_container_width=True)
            st.download_button(
                "Download regression results (CSV)",
                data=res.to_csv(index=False).encode("utf-8"),
                file_name="logistic_regression_results.csv",
                mime="text/csv",
            )


# =============================
# Page — PSM
# =============================
elif page == "Propensity Score Matching":
    st.header(f"Propensity Score Matching {st.session_state['cohort_name_A']} vs {st.session_state['cohort_name_B']}")

    if not HAS_SKLEARN:
        st.error("scikit-learn not installed. Install with: `python -m pip install scikit-learn`")
        st.stop()

    st.caption(
        f"Matches donors in {st.session_state['cohort_name_A']} vs {st.session_state['cohort_name_B']} using propensity scores. "
        "Outcomes are evaluated AFTER matching."
    )

    # -------------------------
    # PS model covariates
    # -------------------------
    # Covariate candidates: numeric + categorical
    numeric_covs = [
        c for c in df.columns
        if c not in ["patient_id"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    cat_covs = [
        c for c in df.columns
        if c not in ["patient_id"]
        and (pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype))
]

    covariate_candidates = sorted(set(numeric_covs + cat_covs))

    default_covs = [c for c in ["age", "gender", "race"] if c in covariate_candidates]

    covariates = st.multiselect(
        "Covariates for PS model (numeric + categorical)",
        options=covariate_candidates,
        default=default_covs,
    )


    caliper = st.slider(
        "Caliper (absolute PS distance)",
        0.0, 0.5, 0.05, 0.01,
    help="A smaller caliper value allows for a more accurate comparison of the treatment and control group by making sure that they are as similar as possible before the treatment is administered. A larger caliper value allows for matching more control patients to the treatment group.")

    no_replacement = st.checkbox(
        "Match without replacement (unique controls)",
        value=True,
        help="If enabled, each control donor can be used only once (true 1:1 matching).",
    )

    results_box = st.container(border=True)

    # -------------------------
    # Run matching
    # -------------------------
    if st.button("Run matching"):
        if len(cohort_A) == 0 or len(cohort_B) == 0:
            st.error("One of the cohorts is empty.")
            st.stop()

        if not covariates:
            st.error("Pick at least one covariate for the PS model.")
            st.stop()

        d = pd.concat([cohort_A, cohort_B], axis=0).reset_index(drop=True)

        X = d[covariates].copy()

        # Convert bools to int (just in case)
        for c in X.columns:
            if pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype(int)

        # One-hot encode categoricals
        X = pd.get_dummies(X, drop_first=True, dummy_na=True)
        # --- Save PRE-matching datasets ---
        pre_df = d.copy()

        pre_df = d.copy()

        pre_raw = d[[ "__group__" ] + covariates].copy()

        pre_X = pd.get_dummies(
            pre_raw.drop(columns=["__group__"]),
            drop_first=True,
            dummy_na=True
        )

        # force numeric for safety (keeps everything numeric for SMD/means)
        pre_X = pre_X.apply(pd.to_numeric, errors="coerce")

        pre_covariates = pd.concat(
            [pre_raw["__group__"].reset_index(drop=True), pre_X.reset_index(drop=True)],
            axis=1
        )

        st.session_state["psm_pre_full"] = pre_df
        st.session_state["psm_pre_cov"] = pre_covariates


        st.session_state["psm_pre_full"] = pre_df
        st.session_state["psm_pre_cov"] = pre_covariates

        # Force numeric & fill missing
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        
        t = d["__group__"].astype(int).values

        ps_model = LogisticRegression(max_iter=2000)
        ps_model.fit(X, t)
        ps = ps_model.predict_proba(X)[:, 1]

        treated_idx = np.where(t == 1)[0]
        control_idx = np.where(t == 0)[0]

        ps_t = ps[treated_idx]
        ps_c = ps[control_idx]

        # -------------------------
        # Matching logic
        # -------------------------
        if not no_replacement:
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(ps_c.reshape(-1, 1))
            dist, neigh = nn.kneighbors(ps_t.reshape(-1, 1))
            dist = dist.flatten()
            neigh = neigh.flatten()

            keep = dist <= caliper
            matched_t = treated_idx[keep]
            matched_c = control_idx[neigh[keep]]

        else:
            # Greedy 1:1 matching without replacement
            order = np.argsort(ps_t)
            used = np.zeros(len(control_idx), dtype=bool)

            mt, mc = [], []
            for j in order:
                diffs = np.abs(ps_c - ps_t[j])
                diffs[used] = np.inf
                k = int(np.argmin(diffs))
                if np.isfinite(diffs[k]) and diffs[k] <= caliper:
                    used[k] = True
                    mt.append(treated_idx[j])
                    mc.append(control_idx[k])

            matched_t = np.array(mt, dtype=int)
            matched_c = np.array(mc, dtype=int)

        matched = pd.concat(
            [d.iloc[matched_t], d.iloc[matched_c]],
            axis=0,
        ).reset_index(drop=True)

        st.session_state["matched_df"] = matched
        post_raw = matched[["__group__"] + covariates].copy()

        post_X = pd.get_dummies(
            post_raw.drop(columns=["__group__"]),
            drop_first=True,
            dummy_na=True
        )

        post_X = post_X.apply(pd.to_numeric, errors="coerce")

        post_covariates = pd.concat(
            [post_raw["__group__"].reset_index(drop=True), post_X.reset_index(drop=True)],
            axis=1
        )

        st.session_state["psm_post_cov"] = post_covariates
        st.session_state["psm_post_full"] = matched


        st.session_state["psm_post_cov"] = post_covariates
        st.session_state["psm_post_full"] = matched

        # -------------------------
        # Display matching results
        # -------------------------
        with results_box:
            st.markdown("## Matching results")

            t_rows = matched[matched["__group__"] == 1]
            c_rows = matched[matched["__group__"] == 0]

            st.write(
                f"Matched rows: **{len(matched):,}** "
                f"({st.session_state['cohort_name_A']}={len(t_rows):,}, "
                f"{st.session_state['cohort_name_B']}={len(c_rows):,})"
            )
            # -------------------------
            # Balance / QC table (pre vs post)
            # -------------------------
            try:
                balance_df = build_balance_table(
                    st.session_state["psm_pre_cov"],
                    st.session_state["psm_post_cov"],
                    group_col="__group__"
                )
                st.session_state["psm_balance"] = balance_df

                st.markdown("### Covariate balance + data availability (pre vs post)")
                st.dataframe(
                    balance_df.style.format({
                        "pre_mean_treated": "{:.3f}",
                        "pre_mean_control": "{:.3f}",
                        "post_mean_treated": "{:.3f}",
                        "post_mean_control": "{:.3f}",
                        "SMD_pre": "{:.3f}",
                        "SMD_post": "{:.3f}",
                    }),
                    use_container_width=True
                )

                # quick warnings for low post-match data
                low = balance_df[
                    (balance_df["post_N_treated"] < 25) | (balance_df["post_N_control"] < 25)
                ]
                if len(low) > 0:
                    st.warning(
                        f"{len(low)} covariates have <25 donors with data after matching. "
                        "Those balance stats are unstable."
                    )

                st.download_button(
                    "Download balance table (N/means/SMD pre+post) (CSV)",
                    data=balance_df.to_csv(index=False).encode("utf-8"),
                    file_name="psm_balance_table_pre_post.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Failed to compute balance/QC table: {e}")

            st.write(
                f"Unique donors: "
                f"{st.session_state['cohort_name_A']}={t_rows['patient_id'].nunique():,}, "
                f"{st.session_state['cohort_name_B']}={c_rows['patient_id'].nunique():,}"
            )

            if no_replacement and len(cohort_A) > len(cohort_B):
                st.warning(
                    "Treated cohort is larger than control cohort. "
                    "Some treated donors could not be matched without replacement."
                )

            st.dataframe(matched.head(200), use_container_width=True)

            st.download_button(
                "Download matched cohort (CSV)",
                data=matched.to_csv(index=False).encode("utf-8"),
                file_name="matched_cohort.csv",
                mime="text/csv",
            )

    # -------------------------
    # Outcomes AFTER matching
    # -------------------------
    if st.session_state.get("matched_df") is not None:
        st.divider()
        st.subheader("Outcomes after matching")

        matched = st.session_state["matched_df"]
        organ_opts = {k: v for k, v in ORGAN_COLS.items() if v in matched.columns}
        if not organ_opts:
            st.info("No outcome columns found in matched cohort.")
            st.stop()

        organ = st.selectbox("Outcome organ", list(organ_opts.keys()))
        out_col = organ_opts[organ]

        vals = sorted(matched[out_col].dropna().astype(str).unique())
        if not vals:
            st.warning("No outcome values available.")
            st.stop()

        pos = st.selectbox("Define POSITIVE as", vals)
        y = (matched[out_col].astype(str) == str(pos)).astype(int)
        t = matched["__group__"].astype(int)

        a = int(((t == 1) & (y == 1)).sum())
        b = int(((t == 1) & (y == 0)).sum())
        c = int(((t == 0) & (y == 1)).sum())
        d0 = int(((t == 0) & (y == 0)).sum())

                # -------------------------
        # 95% CI for OR (Wald, log scale)
        # -------------------------
        


        OR = ((a + 0.5) * (d0 + 0.5)) / ((b + 0.5) * (c + 0.5))
        se_log_or = np.sqrt(
            1 / (a + 0.5) +
            1 / (b + 0.5) +
            1 / (c + 0.5) +
            1 / (d0 + 0.5)
        )

        log_or = np.log(OR)
        z = 1.96  # 95% CI

        ci_low = np.exp(log_or - z * se_log_or)
        ci_high = np.exp(log_or + z * se_log_or)

        st.write(f"Treated positive rate: **{y[t==1].mean()*100:.1f}%**")
        st.write(f"Control positive rate: **{y[t==0].mean()*100:.1f}%**")
        st.success(
            f"Outcome comparison: {st.session_state['cohort_name_A']} (treated) vs {st.session_state['cohort_name_B']} (control) \n\nMatched Patients = **{len(matched):,}** at caliper width of **{caliper}** \n\nOdds Ratio: **{OR:.3f}** "

                   f"(95% Confidence Interval =    [{ci_low:.2f}–{ci_high:.2f}])"
        )
        
        st.divider()
        st.subheader("PSM exports")

        if "psm_pre_cov" in st.session_state:
            st.download_button(
                "Download PRE-matching covariates (CSV)",
                data=st.session_state["psm_pre_cov"].to_csv(index=False).encode("utf-8"),
                file_name="psm_pre_matching_covariates.csv",
                mime="text/csv",
            )

        if "psm_post_cov" in st.session_state:
            st.download_button(
                "Download POST-matching covariates (CSV)",
                data=st.session_state["psm_post_cov"].to_csv(index=False).encode("utf-8"),
                file_name="psm_post_matching_covariates.csv",
                mime="text/csv",
            )

        if "psm_pre_full" in st.session_state:
            st.download_button(
                "Download PRE-matching full donor profiles (CSV)",
                data=st.session_state["psm_pre_full"].to_csv(index=False).encode("utf-8"),
                file_name="psm_pre_matching_full_profiles.csv",
                mime="text/csv",
            )

        if "psm_post_full" in st.session_state:
            st.download_button(
                "Download POST-matching full donor profiles (CSV)",
                data=st.session_state["psm_post_full"].to_csv(index=False).encode("utf-8"),
                file_name="psm_post_matching_full_profiles.csv",
                mime="text/csv",
            )




# ============================================================
# PAGE — Physiology Comparisons
# ============================================================
# ============================================================
# PAGE — Physiology Comparisons
# ============================================================
elif page == "Physiology Comparisons":
    st.header("Physiology Comparisons Between Cohorts")

    st.markdown(
        """
        This section provides **descriptive comparisons of physiologic and laboratory-derived features**
        across cohorts or outcomes.

        These analyses are intended for:
        - Baseline characterization
        - Balance assessment after matching
        - Physiologic plausibility checks

        No causal inference is implied.
        """
    )

    # --------------------------------------------------------
    # 1) Comparison mode (stable keys)
    # --------------------------------------------------------
    comparison_mode = st.radio(
        "Comparison mode",
        options=["outcome", "cohorts", "matched"],
        format_func=lambda x: {
            "outcome": "Outcome-based (within a cohort)",
            "cohorts": f"{st.session_state['cohort_name_A']} vs {st.session_state['cohort_name_B']}",
            "matched": f"{st.session_state['cohort_name_A']} vs {st.session_state['cohort_name_B']} (post-PSM)",
        }[x],
        horizontal=True,
    )

    # --------------------------------------------------------
    # 2) Select data + grouping
    # --------------------------------------------------------
    if comparison_mode == "outcome":
        source_key = st.radio(
            "Cohort source",
            ["A", "B"],
            format_func=lambda k: st.session_state["cohort_name_A"] if k == "A" else st.session_state["cohort_name_B"],
            horizontal=True,
        )

        base_df = cohort_A if source_key == "A" else cohort_B
        if base_df.empty:
            st.warning("Selected cohort is empty.")
            st.stop()

        organ_opts = {k: v for k, v in ORGAN_COLS.items() if v in base_df.columns}
        if not organ_opts:
            st.warning("No organ outcome columns found.")
            st.stop()

        organ = st.selectbox("Outcome organ", list(organ_opts.keys()))
        out_col = organ_opts[organ]

        vals = sorted(base_df[out_col].dropna().astype(str).unique())
        if not vals:
            st.warning("No outcome values available.")
            st.stop()

        pos = st.selectbox("Define POSITIVE as", vals)
        df = base_df.copy()
        group = (df[out_col].astype(str) == str(pos)).astype(int)

        group_labels = ("POSITIVE outcome", "Other outcomes")

    elif comparison_mode == "cohorts":
        if cohort_A.empty or cohort_B.empty:
            st.warning("One of the cohorts is empty.")
            st.stop()

        df = pd.concat([cohort_A, cohort_B], axis=0).copy()
        group = df["__group__"].astype(int)

        group_labels = (
            st.session_state["cohort_name_A"],
            st.session_state["cohort_name_B"],
        )

    else:  # matched
        if "matched_df" not in st.session_state:
            st.warning("No matched cohort found. Run PSM first.")
            st.stop()

        df = st.session_state["matched_df"].copy()
        group = df["__group__"].astype(int)

        group_labels = (
            f"Matched {st.session_state['cohort_name_A']}",
            f"Matched {st.session_state['cohort_name_B']}",
        )

    # --------------------------------------------------------
    # 3) Physiology domains
    # --------------------------------------------------------
    domains = st.multiselect(
        "Physiology domains",
        ["ABG", "Chemistry", "CBC", "Hemodynamics"],
        default=["ABG", "Chemistry"],
    )

    domain_maps = {
        "ABG": abg_map,
        "Chemistry": chem_map,
        "CBC": cbc_map,
        "Hemodynamics": hemo_struct,
    }

    # --------------------------------------------------------
    # 4) Summary metric
    # --------------------------------------------------------
    metric = st.selectbox(
        "Summary metric",
        ["last", "first", "min", "max", "auc", "slope_per_hr", "n"],
        index=0,
        help="Single summary metric used for all variables for interpretability.",
    )

    # --------------------------------------------------------
    # 5) Collect physiology columns
    # --------------------------------------------------------
    phys_cols = []
    for dname in domains:
        for base, cols in domain_maps[dname].items():
            for c in cols:
                if c.endswith(f"_{metric}") and c in df.columns:
                    phys_cols.append(c)

    phys_cols = sorted(set(phys_cols))

    if not phys_cols:
        st.warning("No physiology variables match your selections.")
        st.stop()

    # --------------------------------------------------------
    # 6) Build comparison table
    # --------------------------------------------------------
    rows = []
    for c in phys_cols:
        x1 = df.loc[group == 1, c]
        x0 = df.loc[group == 0, c]

        rows.append({
            "Variable": c,
            f"{group_labels[0]} n": x1.notna().sum(),
            f"{group_labels[0]} mean": x1.mean(),
            f"{group_labels[0]} median": x1.median(),
            f"{group_labels[1]} n": x0.notna().sum(),
            f"{group_labels[1]} mean": x0.mean(),
            f"{group_labels[1]} median": x0.median(),
            "Std Diff": standardized_mean_diff(x1, x0),
            "% Missing": df[c].isna().mean() * 100,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("Std Diff", key=lambda s: s.abs(), ascending=False)

    # --------------------------------------------------------
    # 7) Display + export table
    # --------------------------------------------------------
    st.subheader("Physiology comparison results")
    st.dataframe(result, use_container_width=True)

    st.download_button(
        "Download physiology comparison table (CSV)",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="physiology_comparison.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # 8) Optional visualization
    # --------------------------------------------------------
    if st.checkbox("Show boxplots for top 5 variables"):
        st.subheader("Top physiology variables (boxplots)")

        top_vars = result.head(5)["Variable"].tolist()

        for v in top_vars:
            plot_df = df[[v]].copy()
            plot_df["Group"] = group.map({
                1: group_labels[0],
                0: group_labels[1],
            })
            plot_df = plot_df.dropna()

            chart = (
                alt.Chart(plot_df)
                .mark_boxplot(size=40)
                .encode(
                    x=alt.X("Group:N", title="Group"),
                    y=alt.Y(f"{v}:Q", title=v),
                )
                .properties(height=300)
            )

            st.altair_chart(chart, use_container_width=True)

        st.info(
            """
            **How to read these plots**
            - Boxes show interquartile range (IQR)
            - Center line is the median
            - Whiskers indicate spread excluding extreme outliers
            - Large residual separation after matching suggests incomplete balance
            """
        )

    # --------------------------------------------------------
    # 9) Export raw physiology values
    # --------------------------------------------------------
    st.divider()
    st.subheader("Export physiology-level data")

    export_df = df[["patient_id", "__group__"] + phys_cols].copy()
    export_df["cohort"] = export_df["__group__"].map({
        1: st.session_state["cohort_name_A"],
        0: st.session_state["cohort_name_B"],
    })
    export_df = export_df.drop(columns="__group__")
    export_df = export_df.dropna(subset=phys_cols, how="all")

    st.caption(f"Export contains {len(export_df):,} donors with ≥1 selected physiologic value.")
    st.dataframe(export_df.head(200), use_container_width=True)

    st.download_button(
        "Download physiology-level data (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="physiology_comparison_data.csv",
        mime="text/csv",
    )

       




# =============================
# Page — Export
# =============================
elif page == "Export":
    st.header("Export cohort")

    cohort_choice = st.radio(
        "Export which cohort?",
        [f"{st.session_state['cohort_name_A']}", f"{st.session_state['cohort_name_B']}", f"Combined ({st.session_state['cohort_name_A']} + {st.session_state['cohort_name_B']})", "Matched (if available)"],
        horizontal=True,
    )

    if cohort_choice == "Cohort A":
        cohort = cohort_A.copy()
    elif cohort_choice == "Cohort B":
        cohort = cohort_B.copy()
    elif cohort_choice == "Matched (if available)":
        if st.session_state["matched_df"] is None:
            st.warning("No matched cohort yet. Run PSM first.")
            st.stop()
        cohort = st.session_state["matched_df"].copy()
    else:
        cohort = pd.concat([cohort_A, cohort_B], axis=0).copy()

    export_options = sorted([c for c in cohort.columns if c != "patient_id"])
    default_export = [c for c in ["age", "gender", "race"] if c in export_options]

    export_cols = st.multiselect(
        "Columns to export (patient_id always included if present)",
        options=export_options,
        default=default_export,
    )

    cols = []
    if "patient_id" in cohort.columns:
        cols.append("patient_id")
    cols += export_cols

    out = cohort[cols].copy()
    st.dataframe(out.head(200), use_container_width=True)

    st.download_button(
        "Download cohort (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="cohort_export.csv",
        mime="text/csv",
    )

