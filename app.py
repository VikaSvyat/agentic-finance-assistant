
import os
import asyncio
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.agent import run_agent
from agent.config import MAX_TRACE_OUTPUT_LENGTH

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR_RESOLVED = DATA_DIR.resolve()

st.set_page_config(page_title="Finance MCP Agent", page_icon="💸", layout="wide")

st.markdown("""
<style>
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1180px; }
h1 { font-size: 1.75rem !important; line-height: 1.2 !important; margin-bottom: 0.25rem !important; }
h2 { font-size: 1.2rem !important; margin-top: 0.7rem !important; margin-bottom: 0.45rem !important; }
h3 { font-size: 1.05rem !important; margin-top: 0.5rem !important; margin-bottom: 0.4rem !important; }
.stButton > button {
    background-color: white; color: #31333f; border: 1px solid rgba(49, 51, 63, 0.25);
    border-radius: 0.45rem; padding: 0.42rem 0.9rem; font-weight: 600;
}
.stButton > button:hover { border-color: rgba(49, 51, 63, 0.45); color: #111827; background-color: #fafafa; }
div[data-testid="stDataFrame"] { font-size: 0.82rem; }
.kpi-card {
    padding: 0.8rem 0.9rem; border: 1px solid rgba(49, 51, 63, 0.12);
    border-radius: 0.85rem; background: #ffffff; min-height: 86px;
}
.kpi-label { font-size: 0.78rem; color: #6b7280; margin-bottom: 0.35rem; }
.kpi-value { font-size: 1.35rem; line-height: 1.15; font-weight: 650; color: #262730; }
.business-card {
    padding: 0.85rem 0.95rem; border: 1px solid rgba(49, 51, 63, 0.12);
    border-radius: 0.85rem; background: #fafafa; margin-bottom: 0.7rem;
    font-size: 0.92rem; line-height: 1.55;
}
.section-divider {
    margin-top: 0.9rem; margin-bottom: 0.9rem;
    border-top: 1px solid rgba(49, 51, 63, 0.12);
}
</style>
""", unsafe_allow_html=True)

st.title("💸 Finance MCP Agent")
st.caption("Business dashboard powered by MCP tools + LLM orchestration")

input_mode = st.radio("Input mode", ["Single CSV", "Directory"], horizontal=True)

default_goal_single = """
Analyze this monthly bank statement.
Find spending by category, unusual expenses, savings advice,
save the report, and store summary in SQLite.
"""

default_goal_directory = """
Analyze all CSV bank statements in the selected directory as one combined household financial view.
First discover the CSV files using filesystem tools.
Merge the statements, analyze spending by category, find unusual expenses, generate savings advice,
save the report, and store summary in SQLite.
"""

if input_mode == "Single CSV":
    uploaded_file = st.file_uploader("Upload CSV bank statement", type=["csv"])
    statements_dir = ""
    goal = st.text_area("User goal", value=default_goal_single, height=105)
else:
    uploaded_file = None
    statements_dir = st.text_input(
        "Statements directory",
        value=str(DATA_DIR),
        help="Example: data/statements or /absolute/path/to/statements",
    )
    goal = st.text_area("User goal", value=default_goal_directory, height=125)


def extract_json_from_step(steps, tool_name):
    for step in steps:
        if step.get("tool") == tool_name:
            obs = step.get("observation", {})
            if not isinstance(obs, dict):
                continue
            raw = obs.get("full_output") or obs.get("output") or obs.get("result") or obs.get("output_preview") or ""
            if not raw:
                continue
            try:
                return json.loads(raw)
            except Exception as e:
                st.warning(f"Could not parse JSON for {tool_name}: {e}")
                st.code(str(raw)[:2000])
                return None
    return None


def extract_text_from_step(steps, tool_name):
    for step in steps:
        if step.get("tool") != tool_name:
            continue

        if step.get("output"):
            return str(step["output"])

        obs = step.get("observation", {})
        if not isinstance(obs, dict):
            continue

        raw = obs.get("full_output") or obs.get("output") or obs.get("result") or obs.get("output_preview") or ""
        if raw:
            return str(raw)

    return ""


def parse_report_fallback(report: str):
    summary = {}
    total = re.search(r"Total spent:\s*([\d,.]+)", report)
    txns = re.search(r"Transactions:\s*(\d+)", report)
    avg = re.search(r"Average transaction:\s*([\d,.]+)", report)
    if total:
        summary["total_spent"] = float(total.group(1).replace(",", ""))
    if txns:
        summary["transactions_count"] = int(txns.group(1))
    if avg:
        summary["average_transaction"] = float(avg.group(1).replace(",", ""))
    return summary


def step_status(step):
    obs = step.get("observation", {})
    if step.get("tool") == "final_answer":
        return "🏁 Final"
    if isinstance(obs, dict) and obs.get("ok") is True:
        return "✅ Success"
    if isinstance(obs, dict) and obs.get("ok") is False:
        return "❌ Error"
    return "⚪ Unknown"


def format_money(value):
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "0.00"


def kpi_card(label, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def truncate_trace_value(value, limit=MAX_TRACE_OUTPUT_LENGTH):
    text = str(value)

    if len(text) <= limit:
        return value

    omitted = len(text) - limit
    return f"{text[:limit]}\n... [truncated {omitted} chars]"


def truncate_trace_payload(value):
    if isinstance(value, dict):
        return {
            key: truncate_trace_payload(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            truncate_trace_payload(item)
            for item in value
        ]

    if isinstance(value, str):
        return truncate_trace_value(value)

    return value


def prepare_categories_df(analysis):
    categories = analysis.get("top_categories", []) if analysis else []
    if not categories:
        return pd.DataFrame()
    df = pd.DataFrame(categories).rename(columns={
        "category_en": "Category",
        "total_amount": "Amount (NIS)",
        "transactions_count": "Transactions",
        "average_transaction": "Avg Transaction",
    })
    cols = ["Category", "Amount (NIS)", "Transactions", "Avg Transaction"]
    return df[[c for c in cols if c in df.columns]]


def prepare_merchants_df(analysis):
    merchants = analysis.get("top_merchants", []) if analysis else []
    if not merchants:
        return pd.DataFrame()
    df = pd.DataFrame(merchants).rename(columns={
        "merchant": "Merchant",
        "total_amount": "Amount (NIS)",
        "transactions_count": "Transactions",
        "average_transaction": "Avg Transaction",
    })
    cols = ["Merchant", "Amount (NIS)", "Transactions", "Avg Transaction"]
    return df[[c for c in cols if c in df.columns]]


def prepare_expenses_df(unusual):
    large_expenses = unusual.get("large_expenses_to_review", []) if unusual else []
    if not large_expenses:
        return pd.DataFrame()
    df = pd.DataFrame(large_expenses).rename(columns={
        "transaction_date": "Date",
        "merchant": "Merchant",
        "amount": "Amount (NIS)",
        "normalized_category_en": "Category",
        "threshold": "Threshold",
        "threshold_method": "Method",
    })
    cols = ["Date", "Merchant", "Amount (NIS)", "Category", "Threshold", "Method"]
    return df[[c for c in cols if c in df.columns]]


def is_path_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_uploaded_csv_path(uploaded_name: str) -> tuple[Path | None, str]:
    filename = Path(uploaded_name).name

    if not filename or filename in {".", ".."}:
        return None, "Uploaded file name is not valid."

    if Path(filename).suffix.lower() != ".csv":
        return None, "Only CSV files are supported."

    csv_path = (DATA_DIR / filename).resolve()

    if not is_path_inside(csv_path, DATA_DIR_RESOLVED):
        return None, "Uploaded file path is outside the data directory."

    return csv_path, ""


def validate_statements_directory(raw_path: str) -> tuple[Path | None, str]:
    path_text = raw_path.strip()

    if not path_text:
        return None, "Enter a statements directory first."

    if path_text.startswith("file://"):
        return None, "Use a local path without file:// prefix."

    directory = Path(path_text).expanduser().resolve()

    if not directory.exists():
        return None, "Directory does not exist."

    if not directory.is_dir():
        return None, "Path is not a directory."

    if not is_path_inside(directory, PROJECT_ROOT):
        return None, f"Directory must be inside the project folder: {PROJECT_ROOT}"

    csv_files = list(directory.glob("*.csv"))

    if not csv_files:
        return None, "Directory does not contain CSV files."

    return directory, ""


def run_selected_agent():
    if input_mode == "Single CSV":
        if not uploaded_file:
            st.warning("Upload a CSV file first.")
            return None

        csv_path, error = safe_uploaded_csv_path(uploaded_file.name)
        if error:
            st.error(error)
            return None

        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if not csv_path.exists():
            st.error("Uploaded file was not saved correctly.")
            return None

        st.success(f"File saved to: `{csv_path}`")
        return asyncio.run(run_agent(goal, str(csv_path)))

    directory, error = validate_statements_directory(statements_dir)

    if error:
        st.warning(error)
        return None

    st.info(f"Directory selected: `{directory}`")
    return asyncio.run(run_agent(goal, csv_path="", statements_dir=str(directory)))


can_run = uploaded_file is not None if input_mode == "Single CSV" else bool(statements_dir.strip())

if st.button("🚀 Run Agent", disabled=not can_run):
    with st.spinner("Agent is working..."):
        result = run_selected_agent()

    if result:
        answer = result.get("answer", "")
        steps = result.get("steps", [])

        analysis = extract_json_from_step(steps, "finance.analyze_statement")
        unusual = extract_json_from_step(steps, "finance.find_unusual_expenses")
        ai_insights = extract_text_from_step(steps, "finance.generate_ai_financial_insights")

        if not analysis:
            analysis = parse_report_fallback(answer)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("Financial Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_card("Total Spent", f"{format_money(analysis.get('total_spent', 0))} NIS")
        with col2:
            kpi_card("Transactions", f"{analysis.get('transactions_count', 0)}")
        with col3:
            kpi_card("Average Transaction", f"{format_money(analysis.get('average_transaction', 0))} NIS")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        left, right = st.columns(2)

        with left:
            st.subheader("🏆 Top Categories")
            categories_df = prepare_categories_df(analysis)
            if not categories_df.empty:
                st.dataframe(categories_df, use_container_width=True, hide_index=True)
            else:
                st.info("No category data found.")

        with right:
            st.subheader("🛒 Top Merchants")
            merchants_df = prepare_merchants_df(analysis)
            if not merchants_df.empty:
                st.dataframe(merchants_df, use_container_width=True, hide_index=True)
            else:
                st.info("No merchant data found.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("⚠️ Large Expenses to Review")
        expenses_df = prepare_expenses_df(unusual)
        if not expenses_df.empty:
            st.dataframe(expenses_df, use_container_width=True, hide_index=True)
        else:
            st.success("No large expenses detected.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("AI Insights")

        if ai_insights:
            st.markdown(ai_insights)
        else:
            st.info("AI insights were not generated for this run.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        with st.expander("🔧 Technical Agent Trace", expanded=False):
            st.caption("This section demonstrates MCP tool orchestration.")

            rows = []
            for step in steps:
                obs = step.get("observation", {})
                timing = step.get("timing", {})
                rows.append({
                    "Step": step.get("step"),
                    "Tool": step.get("tool"),
                    "Status": step_status(step),
                    "LLM (s)": timing.get("llm_response_seconds", ""),
                    "Tool (s)": timing.get("tool_execution_seconds", ""),
                    "Total (s)": timing.get("total_step_seconds", ""),
                    "Reason": step.get("reason", ""),
                    "Error": obs.get("error", "") if isinstance(obs, dict) else "",
                })

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.subheader("Raw Step Details")
            for step in steps:
                with st.expander(f"{step_status(step)} | Step {step.get('step')}: {step.get('tool')}", expanded=False):
                    st.write("Reason:")
                    st.write(step.get("reason", ""))

                    if "args" in step:
                        st.write("Args:")
                        st.json(truncate_trace_payload(step["args"]))

                    if "observation" in step:
                        st.write("Observation:")
                        st.json(truncate_trace_payload(step["observation"]))

                    if "output" in step:
                        st.write("Output:")
                        st.markdown(truncate_trace_value(step["output"]))
else:
    if input_mode == "Single CSV":
        st.info("Upload a CSV file first.")
    else:
        st.info("Enter a directory path containing CSV statements.")
