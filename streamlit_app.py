"""
Altion Streamlit UI — AI Portfolio Rebalancer

Frontend only: uploads a portfolio file, selects risk profile, builds
the prompt via backend helpers, and displays the LLM response with
helpful visuals.

Assumptions:
- invoke_altion(prompt: str) is available
- build_prompt(df, risk) is available
- pandas, streamlit, dotenv are installed
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Backend helpers (defined in this repo). If dependencies are missing,
# show a friendly message instead of a raw ImportError page.
try:
    from altion_portfolio import parse_portfolio, build_prompt, invoke_altion
except ImportError as e:
    import sys
    st.error(
        "Missing dependency while loading backend modules.\n\n"
        "If the error mentions ChatGroq or langchain_groq, install it in the same Python environment that runs Streamlit:"
    )
    st.code(
        "# Option A: use project venv\n"
        ".\\venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
        ".\\venv\\Scripts\\python.exe -m streamlit run streamlit_app.py\n\n"
        "# Option B: install into current interpreter\n"
        "python -m pip install langchain-groq\n",
        language="bash",
    )
    st.caption(f"Streamlit Python: {sys.executable}")
    st.stop()


# -------------------------------------------------------------
# Page config and custom CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Altion – AI Portfolio Rebalancer",
    page_icon="📈",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* Overall polish */
.main .block-container{padding-top:2rem;padding-bottom:4rem;}

/* Header styles */
.altion-header {
  display: flex; align-items: center; gap: 1rem;
  padding: 1rem 1.25rem; border-radius: 14px;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #334155 100%);
  color: #e2e8f0; box-shadow: 0 4px 24px rgba(0,0,0,0.2);
}
.altion-title {font-size: 1.6rem; font-weight: 700; margin: 0;}
.altion-subtitle {font-size: 0.95rem; color:#cbd5e1; margin: 0;}

/* Card-like sections */
.altion-card {background: #0b1220; border:1px solid #1f2937; border-radius: 12px; padding: 1rem;}
.altion-card h3{margin-top:0;}
.altion-card.altion-response{font-size:1.05rem; line-height:1.65;}

/* Fine-tune Streamlit defaults */
section[data-testid="stSidebar"] {border-right: 1px solid #1f2937;}
.stButton>button {border-radius:10px; padding:.6rem 1.1rem; font-weight:600;}
.stMarkdown p {line-height:1.55;}

/* Make tables compact */
table {font-size: 0.92rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _clean_portfolio_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize portfolio DataFrame to ['Stock','Quantity','Current Value'].

    Mirrors the cleaning logic in parse_portfolio for CSV/direct DataFrames.
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("Uploaded file appears empty.")

    norm = {str(c).strip().lower().replace("_", " "): c for c in df_raw.columns}

    def has(names: list[str]) -> Optional[str]:
        for name in names:
            original = norm.get(name)
            if original is not None:
                return original
        return None

    stock_col = has(["stock", "ticker", "symbol"])  # -> Stock
    qty_col = has(["quantity", "qty", "shares", "quantity available"])  # -> Quantity
    curr_val_col = has(["current value", "value", "market value", "present value"])  # -> Current Value
    price_col = has(
        [
            "previous closing price",
            "prev close",
            "last traded price",
            "ltp",
            "price",
            "average price",
        ]
    )  # optional

    if stock_col is None or qty_col is None:
        raise ValueError("File must contain Stock and Quantity columns.")

    df = df_raw.copy()
    df.rename(
        columns={
            stock_col: "Stock",
            qty_col: "Quantity",
            **({curr_val_col: "Current Value"} if curr_val_col else {}),
            **({price_col: "Price"} if price_col else {}),
        },
        inplace=True,
    )

    df["Stock"] = df["Stock"].astype(str).str.strip().str.upper()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    if "Current Value" in df.columns:
        df["Current Value"] = pd.to_numeric(df["Current Value"], errors="coerce")
    if "Current Value" not in df.columns and "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Current Value"] = (df["Price"] * df["Quantity"]).round(2)

    df = df[(df["Stock"].notna()) & (df["Stock"].str.len() > 0)]
    df = df[(df["Quantity"].notna()) & (df["Quantity"] > 0)]
    if "Current Value" not in df.columns:
        df["Current Value"] = pd.NA

    df = df[["Stock", "Quantity", "Current Value"]]
    df.sort_values("Stock", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_uploaded_portfolio(uploaded) -> Optional[pd.DataFrame]:
    """Load uploaded Excel/CSV into a normalized portfolio DataFrame."""
    if uploaded is None:
        return None

    name = uploaded.name.lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            # Write to temp and reuse backend cleaner
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(name)[1], delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            df = parse_portfolio(tmp_path)
            return df
        elif name.endswith(".csv"):
            data = uploaded.read()
            df_raw = pd.read_csv(io.BytesIO(data))
            return _clean_portfolio_df(df_raw)
        else:
            raise ValueError("Unsupported file type. Please upload CSV or Excel.")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None


def portfolio_summary(df: pd.DataFrame) -> Dict[str, float | int]:
    """Compute simple summary stats."""
    n = len(df)
    total_value = pd.to_numeric(df.get("Current Value", pd.Series([pd.NA]*n)), errors="coerce").sum(min_count=1)
    return {
        "holdings": int(n),
        "total_value": float(total_value) if pd.notna(total_value) else 0.0,
    }


def plot_current_allocation(df: pd.DataFrame):
    """Plot a pie chart of current allocation by Current Value (or Quantity)."""
    has_values = df["Current Value"].notna().any()
    values_col = "Current Value" if has_values else "Quantity"
    title = "Current Allocation (by Value)" if has_values else "Current Allocation (by Quantity)"

    plot_df = df.copy()
    # Avoid zeros in pie charts; add epsilon to pure zero rows when all zeros
    if not has_values and (plot_df[values_col] == 0).all():
        plot_df[values_col] = 1.0

    fig = px.pie(
        plot_df,
        names="Stock",
        values=values_col,
        title=title,
        hole=0.35,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def try_parse_suggested_allocation(text: str) -> Optional[pd.DataFrame]:
    """Parse an optional suggested allocation from the LLM response.

    Heuristics:
    - JSON with mapping {"AAPL": 20, ...}
    - Lines like "AAPL: 20%"
    - Markdown table with columns like Ticker|Target (% or Weight)
    Returns a DataFrame with columns ["Asset", "Target %"], or None.
    """
    if not text:
        return None

    # JSON-like block
    try:
        # Extract first {...} block
        json_match = re.search(r"\{[\s\S]*?\}", text)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                rows = []
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        rows.append({"Asset": str(k).upper(), "Target %": float(v)})
                if rows:
                    return pd.DataFrame(rows)
    except Exception:
        pass

    # Key: value%
    lines = text.splitlines()
    rows = []
    for ln in lines:
        m = re.search(r"([A-Za-z0-9_\-\.]{1,15})\s*[:\-]\s*(\d+(?:\.\d+)?)\s*%", ln)
        if m:
            rows.append({"Asset": m.group(1).upper(), "Target %": float(m.group(2))})
    if rows:
        return pd.DataFrame(rows)

    # Minimal Markdown table parse (header must contain % or weight)
    if any("|" in ln for ln in lines):
        try:
            tbl_text = "\n".join(ln for ln in lines if "|" in ln)
            df_tbl = pd.read_csv(io.StringIO(tbl_text.replace("|", ",")))
            cols = [c.strip().lower() for c in df_tbl.columns]
            if any("%" in c or "weight" in c for c in cols):
                # Find likely ticker/asset col
                name_col = next((c for c in df_tbl.columns if str(c).strip().lower() in {"ticker","symbol","asset","name"}), df_tbl.columns[0])
                pct_col = next((c for c in df_tbl.columns if "%" in str(c).lower() or "weight" in str(c).lower()), df_tbl.columns[-1])
                out = df_tbl[[name_col, pct_col]].copy()
                out.columns = ["Asset", "Target %"]
                out["Asset"] = out["Asset"].astype(str).str.upper()
                out["Target %"] = pd.to_numeric(out["Target %"], errors="coerce")
                out = out.dropna()
                if not out.empty:
                    return out
        except Exception:
            pass

    return None


def build_stock_outlook_prompt(
    base_prompt: str,
    ticker: str,
    horizon_label: str,
    risk_level: str,
) -> str:
    """Extend the core portfolio prompt to focus on a single ticker outlook."""
    ticker = ticker.upper()
    horizon_label = horizon_label.lower()
    return (
        f"{base_prompt}\n\n"
        f"Focus specifically on {ticker}. Provide a concise outlook for the next {horizon_label}. "
        f"Discuss expected catalysts, downside risks, and recommended positioning that fits a {risk_level} risk profile. "
        "Close with actionable guidance (hold/add/trim) and any key metrics to monitor."
    )


# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------
with st.sidebar:
    load_dotenv()
    st.header("About Altion")
    st.write(
        "Altion is an AI-powered portfolio rebalancing agent. Upload your holdings, select a risk profile, and let the LLM suggest a clean, diversified allocation."
    )
    # Optional: surface environment hints
    if os.getenv("GROQ_API_KEY"):
        st.caption("Groq key detected ✔")
    else:
        st.caption("Groq key not found – set GROQ_API_KEY in .env")


# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.markdown(
    """
    <div class="altion-header">
      <div>
        <h1 class="altion-title">Altion</h1>
        <p class="altion-subtitle">AI Portfolio Rebalancer</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# Inputs
# -------------------------------------------------------------
with st.container():
    st.subheader("Upload Portfolio")
    uploaded = st.file_uploader(
        "Upload an Excel (.xlsx/.xls) or CSV file with columns Stock, Quantity, and (optionally) Current Value.",
        type=["xlsx", "xls", "csv"],
        help="Columns accepted: Stock/Ticker/Symbol; Quantity/Qty/Shares; Current Value/Value/Market Value; optional Price",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        risk = st.selectbox("Risk Profile", options=["Low", "Medium", "High"], index=1)
    with col2:
        submit = st.button("Rebalance Portfolio", type="primary")


df: Optional[pd.DataFrame] = None
if uploaded is not None:
    df = load_uploaded_portfolio(uploaded)
    if df is not None and not df.empty:
        st.markdown("<div class=\"altion-card\">", unsafe_allow_html=True)
        st.write("Preview")
        st.dataframe(df, use_container_width=True, hide_index=True)
        smry = portfolio_summary(df)
        st.caption(f"Holdings: {smry['holdings']} • Total Value: ₹{smry['total_value']:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# Action: Build prompt and invoke LLM
# -------------------------------------------------------------
if submit:
    if df is None or df.empty:
        st.error("Please upload a valid portfolio file before rebalancing.")
    else:
        core_prompt = build_prompt(df, risk.lower())
        unique_tickers = sorted({str(t).strip().upper() for t in df["Stock"].dropna().unique()})
        prompt = core_prompt

        with st.spinner("Generating suggested allocation..."):
            try:
                response = invoke_altion(prompt)
            except Exception as e:
                response = f"Error contacting LLM: {e}"

        # Show LLM output
        st.subheader("Suggested Rebalance Plan")
        st.markdown("<div class=\"altion-card altion-response\">", unsafe_allow_html=True)
        st.markdown(response if isinstance(response, str) else str(response), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Charts
        st.subheader("Visualizations")
        parsed = None
        c1, c2 = st.columns(2)
        with c1:
            plot_current_allocation(df)
        with c2:
            parsed = try_parse_suggested_allocation(response if isinstance(response, str) else str(response))
            if parsed is not None and not parsed.empty:
                fig2 = px.pie(
                    parsed,
                    names="Asset",
                    values="Target %",
                    title="Suggested Allocation",
                    hole=0.35,
                    color_discrete_sequence=px.colors.sequential.PuBu,
                )
                fig2.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
            else:
                st.info("Upload a portfolio and click Rebalance to see suggestions. If the model returns target weights (e.g., 'AAPL: 20%'), they will be parsed here into a pie chart.")

        # Persist context for downstream features
        st.session_state["core_prompt"] = core_prompt
        st.session_state["risk_level"] = risk.lower()
        st.session_state["holdings_tickers"] = unique_tickers
        suggested_from_model = (
            parsed["Asset"].dropna().astype(str).str.upper().tolist()
            if parsed is not None
            else []
        )
        st.session_state["suggested_tickers"] = suggested_from_model or unique_tickers

outlook_universe = st.session_state.get("suggested_tickers") or st.session_state.get("holdings_tickers")
if outlook_universe:
    st.subheader("Stock Outlook Explorer")
    col_stock, col_horizon = st.columns([2, 1])
    with col_stock:
        selected_stock = st.selectbox(
            "Select a stock to research",
            outlook_universe,
            key="stock_outlook_select",
        )
    with col_horizon:
        horizon_choice = st.radio(
            "Time horizon",
            ["1 month", "6 months", "1 year"],
            key="stock_outlook_horizon",
            horizontal=True,
        )
    if st.button("Generate Outlook", key="stock_outlook_btn"):
        base_prompt = st.session_state.get("core_prompt", "")
        risk_level = st.session_state.get("risk_level", "medium")
        outlook_prompt = build_stock_outlook_prompt(
            base_prompt,
            selected_stock,
            horizon_choice,
            risk_level,
        )
        with st.spinner(f"Gathering {horizon_choice} outlook for {selected_stock}..."):
            try:
                outlook_response = invoke_altion(outlook_prompt)
            except Exception as e:
                outlook_response = f"Error contacting LLM: {e}"
        st.markdown(
            outlook_response if isinstance(outlook_response, str) else str(outlook_response),
            unsafe_allow_html=True,
        )
