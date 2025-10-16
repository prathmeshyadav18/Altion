"""
Altion portfolio input layer.

Functions:
- parse_portfolio(file_path): Load and clean a stock portfolio from Excel.
- get_risk_profile(): Get user's risk tolerance from .env or prompt.
- build_prompt(portfolio_df, risk_level): Build a readable LLM prompt.
- invoke_altion(prompt): Call provider LLM via existing get_llm.

Dependencies: pandas, openpyxl, python-dotenv

This module stays UI-agnostic for easy integration with other layers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

# We reuse the LLM factory configured in your repo
from altion_agent import get_llm


def _extract_embedded_table(file_path: Path) -> pd.DataFrame | None:
    """Handle broker exports that embed the holdings table below metadata."""
    try:
        df = pd.read_excel(file_path, header=None)
    except Exception:
        return None

    for idx, row in df.iterrows():
        values = [
            str(v).strip()
            if pd.notna(v)
            else ""
            for v in row.tolist()
        ]
        lowered = [v.lower() for v in values]
        if "symbol" in lowered and any("quantity" in v for v in lowered):
            table = df.iloc[idx + 1 :].copy()
            table.columns = values
            valid_cols = [
                col
                for col in table.columns
                if col
                and str(col).strip()
                and str(col).strip().lower() != "nan"
            ]
            table = table[valid_cols]
            table = table.dropna(how="all")
            table.reset_index(drop=True, inplace=True)
            return table
    return None


def parse_portfolio(file_path: str | os.PathLike) -> pd.DataFrame:
    """Read an Excel portfolio and return a clean DataFrame.

    Expected columns (case/spacing-insensitive, aliases supported):
    - Stock | Ticker | Symbol -> 'Stock' (string)
    - Quantity | Qty | Shares -> 'Quantity' (float)
    - Current Value | Value | Market Value -> 'Current Value' (float, optional)
    - Price (optional; used to compute Current Value if missing)

    Returns a DataFrame with standardized columns: ['Stock', 'Quantity', 'Current Value']
    and only valid rows (non-empty ticker, quantity > 0).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Portfolio file not found: {file_path}")

    # Load Excel (ensure openpyxl is available)
    try:
        df_raw = pd.read_excel(file_path)  # engine auto-detected (needs openpyxl for .xlsx)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "openpyxl is required to read .xlsx files. Install with: pip install openpyxl"
        ) from e

    if df_raw.empty:
        # Some broker exports put the table below headers; try extraction path.
        extracted = _extract_embedded_table(file_path)
        if extracted is None or extracted.empty:
            raise ValueError("Portfolio file is empty")
        df_raw = extracted

    # Normalize column names to simplify mapping
    norm = {str(c).strip().lower().replace("_", " "): c for c in df_raw.columns}

    def has(col_name_variants: List[str]) -> str | None:
        for variant in col_name_variants:
            original = norm.get(variant)
            if original is not None:
                return original
        return None

    stock_col = has(["stock", "ticker", "symbol"])  # canonical -> 'Stock'
    qty_col = has(["quantity", "qty", "shares", "quantity available"])  # canonical -> 'Quantity'
    curr_val_col = has(["current value", "value", "market value", "present value"])  # -> 'Current Value'
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
        extracted = _extract_embedded_table(file_path)
        if extracted is not None and not extracted.empty:
            df_raw = extracted
            norm = {str(c).strip().lower().replace("_", " "): c for c in df_raw.columns}
            stock_col = has(["stock", "ticker", "symbol"])
            qty_col = has(["quantity", "qty", "shares", "quantity available"])
            curr_val_col = has(["current value", "value", "market value", "present value"])
            price_col = has(
                [
                    "previous closing price",
                    "prev close",
                    "last traded price",
                    "ltp",
                    "price",
                    "average price",
                ]
            )

    if stock_col is None or qty_col is None:
        raise ValueError(
            "Portfolio must include stock and quantity columns (e.g., 'Stock' and 'Quantity')."
        )

    df = df_raw.copy()
    # Standardize core fields
    df.rename(
        columns={
            stock_col: "Stock",
            qty_col: "Quantity",
            **({curr_val_col: "Current Value"} if curr_val_col else {}),
            **({price_col: "Price"} if price_col else {}),
        },
        inplace=True,
    )

    # Clean values
    df["Stock"] = df["Stock"].astype(str).str.strip().str.upper()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    if "Current Value" in df.columns:
        df["Current Value"] = pd.to_numeric(df["Current Value"], errors="coerce")
    # Compute current value if not provided but price is present
    if "Current Value" not in df.columns and "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Current Value"] = (df["Price"] * df["Quantity"]).round(2)

    # Drop invalid rows
    df = df[(df["Stock"].notna()) & (df["Stock"].str.len() > 0)]
    df = df[(df["Quantity"].notna()) & (df["Quantity"] > 0)]

    # Ensure the column exists even if not provided/computed
    if "Current Value" not in df.columns:
        df["Current Value"] = pd.NA

    # Final ordering and tidy up
    df = df[["Stock", "Quantity", "Current Value"]]
    df.sort_values("Stock", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_risk_profile() -> str:
    """Return user's risk profile: 'low', 'medium', or 'high'.

    Priority:
    1) .env value RISK_PROFILE (case-insensitive)
    2) Interactive prompt fallback (press Enter for 'medium')
    """
    load_dotenv()
    env_val = os.getenv("RISK_PROFILE", "").strip().lower()
    valid = {"low", "medium", "high"}

    if env_val in valid:
        return env_val

    try:
        user_in = input("Enter risk profile [low/medium/high] (default: medium): ").strip().lower()
    except Exception:
        # Non-interactive environments
        user_in = ""

    if user_in in valid:
        return user_in
    return "medium"


def _format_portfolio_list(df: pd.DataFrame) -> str:
    """Format portfolio holdings into a compact English list."""
    parts: List[str] = []
    for _, row in df.iterrows():
        qty = float(row["Quantity"]) if pd.notna(row["Quantity"]) else 0
        ticker = str(row["Stock"]).upper()
        val = row.get("Current Value", pd.NA)
        if pd.notna(val):
            parts.append(f"{qty:g} shares of {ticker} (₹{float(val):,.2f})")
        else:
            parts.append(f"{qty:g} shares of {ticker}")

    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def build_prompt(portfolio_df: pd.DataFrame, risk_level: str) -> str:
    """Build a rich LLM prompt that enforces allocation constraints and risk context."""
    risk_level = (risk_level or "").strip().lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "medium"

    df = portfolio_df.copy()
    df["Quantity"] = pd.to_numeric(df.get("Quantity"), errors="coerce").fillna(0)
    if "Current Value" in df.columns:
        df["Current Value"] = pd.to_numeric(df["Current Value"], errors="coerce")
    else:
        df["Current Value"] = pd.NA

    total_value = float(df["Current Value"].fillna(0).sum())
    holdings_lines: List[str] = []
    for _, row in df.iterrows():
        ticker = str(row.get("Stock", "")).strip().upper()
        if not ticker:
            continue
        qty = float(row.get("Quantity", 0) or 0)
        current_value = row.get("Current Value", pd.NA)
        if pd.notna(current_value) and total_value > 0:
            weight = (float(current_value) / total_value) * 100
            holdings_lines.append(
                f"- {ticker}: {qty:g} shares, current value INR {float(current_value):,.2f} ({weight:.2f}% of portfolio)"
            )
        elif pd.notna(current_value):
            holdings_lines.append(
                f"- {ticker}: {qty:g} shares, current value INR {float(current_value):,.2f}"
            )
        else:
            holdings_lines.append(f"- {ticker}: {qty:g} shares (current value unavailable)")

    holdings_summary = "\n".join(holdings_lines) if holdings_lines else "- No active equity positions recorded."
    listing = _format_portfolio_list(df)

    risk_guidance = {
        "low": (
            "Prioritise low-volatility, high-liquidity large and mid caps. Limit exposure to speculative or penny names. "
            "Still, ensure the allocation can outperform inflation with some growth sectors."
        ),
        "medium": (
            "Blend resilient large/mid caps with a sleeve of high-growth or thematic ideas that trade with strong daily volume. "
            "You may allocate up to 5% collectively to penny/small-cap ideas that have clear catalysts."
        ),
        "high": (
            "Lean into high-beta growth sectors (tech, renewables, defence, specialty manufacturing) and mid/small caps with demonstrated breakout patterns and heavy trading volume. "
            "Actively recycle capital from legacy blue chips into these names where justified. "
            "Include a dedicated shortlist of penny/small-cap swing ideas (<=3% weight each) that fit the theme."
        ),
    }[risk_level]

    penny_section_rule = (
        "Include a `## Penny Stock Ideas` section with 2-3 liquid small/micro-cap tickers (daily turnover > INR 5 crore). "
        "Limit each to <=2% weight and justify catalysts, liquidity, and risk controls."
        if risk_level in {"medium", "high"}
        else "Skip any penny-stock ideas for this risk level."
    )

    capital_constraint = (
        f"- Net deployed capital (after sells) must stay within +/-1% of INR {total_value:,.2f}. Facilitate buys by specifying which current holdings to trim or exit."
        if total_value > 0
        else "- Keep the total allocation internally consistent so target weights sum to 100% even if current market values are unavailable."
    )

    prompt = f"""
You are an equity portfolio strategist tasked with rebalancing a client portfolio within existing capital.

Client risk tolerance: **{risk_level.upper()}**
Current equity holdings summary:
{holdings_summary}

Total investable capital (current value): INR {total_value:,.2f}

Constraints:
{capital_constraint}
- If you recommend selling or trimming a position, include it in the allocation table with a reduced/zero target weight and call out the action.
- Use only NSE/BSE-listed equities with healthy liquidity; favour stocks exhibiting sustained high trading volume and strong growth trajectories relative to the risk profile.
- Express the final allocation as percentages that sum to 100%.
- Ensure qualitative rationale references catalysts, valuation, liquidity, and downside management.

Risk posture guidance:
{risk_guidance}

{penny_section_rule}

Output format (Markdown):
## Target Allocation
| Ticker | Action (Buy/Sell/Hold) | Target % | Thesis / Liquidity Notes |
| --- | --- | --- | --- |
...populate one row per holding or new suggestion...

## Rebalance Actions
- Bullet list summarising what to sell, hold, and accumulate, including approximate INR amounts based on current capital.

## Penny Stock Ideas
- Provide the mandated section (or explicitly state it is omitted for low risk). Each item: `Ticker – Target % – Daily volume and catalyst`.

## Risk Management
- Detail hedging, stop-loss, or monitoring guidance matched to the {risk_level} profile.

Portfolio snapshot reference for you: {listing if listing else "(empty)"}
"""
    return prompt.strip()


def invoke_altion(prompt: str) -> str:
    """Call Groq LLM using the project's get_llm(model_name).

    - Model is taken from GROQ_MODEL; if missing/invalid, try common fallbacks.
    """
    load_dotenv()

    def _try_invoke(model_name: str) -> str:
        llm = get_llm(model_name)
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp))

    env_model = os.getenv("GROQ_MODEL")
    candidates = [
        # Groq often exposes these model IDs; include OSS variant too
        "openai/gpt-oss-20b",
        "mixtral-8x7b-32768",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "llama-3.2-90b-text-preview",
        "llama-3.2-11b-text-preview",
        "llama-3.1-8b-instant",
    ]

    last_err: Exception | None = None

    if env_model:
        try:
            return _try_invoke(env_model)
        except Exception as e:
            last_err = e
            # If it's an obviously wrong/retired model, fall through to candidates.
            msg = str(e).lower()
            retryable = any(s in msg for s in [
                "decommissioned",
                "model not found",
                "unsupported model",
                "unknown model",
                "invalid model",
            ])
            if not retryable:
                raise

    for m in candidates:
        try:
            return _try_invoke(m)
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise RuntimeError("Unable to contact Groq or invalid configuration.")


def create_sample_portfolio(file_path: str | os.PathLike) -> Path:
    """Create a sample Excel portfolio for testing."""
    sample = pd.DataFrame(
        {
            "Stock": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
            "Quantity": [25, 10, 8, 5, 6],
            "Current Value": [
                25 * 195.2,
                10 * 412.6,
                8 * 171.0,
                5 * 950.0,
                6 * 245.3,
            ],
        }
    )

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure openpyxl is available when writing. Pandas will raise if missing.
    sample.to_excel(file_path, index=False)
    return file_path


if __name__ == "__main__":
    # Simple demo: generate sample file, parse, build prompt, and invoke LLM
    load_dotenv()
    os.environ.setdefault("RISK_PROFILE", "medium")  # avoid interactive prompt during demo

    data_path = Path("data")
    xlsx_path = data_path / "sample_portfolio.xlsx"
    create_sample_portfolio(xlsx_path)

    df = parse_portfolio(xlsx_path)
    risk = get_risk_profile()
    prompt = build_prompt(df, risk)

    print("--- Built Prompt ---")
    print(prompt)
    print("--------------------\n")

    try:
        resp = invoke_altion(prompt)
        print("--- LLM Response ---")
        try:
            print(resp)
        except Exception:
            # Fallback for Windows console encoding edge cases
            print(str(resp).encode("utf-8", "ignore").decode("utf-8"))
    except Exception as e:
        # Fallback simulated output if LLM call fails (e.g., missing key, network)
        print("--- LLM Response (simulated) ---")
        print(
            "Sample guidance: Target a diversified allocation based on your medium risk profile,\n"
            "e.g., 60% large-cap (AAPL/MSFT/GOOGL), 25% growth/AI (NVDA/TSLA), 10% international,\n"
            "5% cash. Rebalance by trimming overweight NVDA and adding to underweight positions."
        )
        print(f"(Note: Real call failed: {e})")
