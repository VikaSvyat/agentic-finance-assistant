
import os
import asyncio
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.runtime_config import PROJECT_ROOT, load_runtime_config

load_runtime_config()

from agent.agent import run_agent, run_financial_question
from agent.config import MAX_TRACE_OUTPUT_LENGTH
from finance.display import format_merchant_display
from servers.finance_server import (
    default_db_path,
    load_spending_transactions_from_db,
    mark_duplicate_transactions_in_db,
    parse_transactions_from_file,
    save_transactions_to_db,
    summarize_transaction_import,
)

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
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

default_goal_single = """Analyze this monthly bank statement.
Find spending by category, unusual expenses, savings advice,
save the report, and store summary in SQLite.
"""

default_goal_directory = """Analyze all CSV bank statements in the selected directory as one combined household financial view.
First discover the CSV files using filesystem tools.
Merge the statements, analyze spending by category, find unusual expenses, generate savings advice,
save the report, and store summary in SQLite.
"""

setup_left, setup_right = st.columns([1, 2], gap="large")

with setup_left:
    st.header("Upload")
    input_mode = st.radio("Input mode", ["Single CSV", "Directory"], horizontal=True)

    if input_mode == "Single CSV":
        uploaded_file = st.file_uploader("Upload finance file", type=["csv", "xlsx", "xls"])
        statements_dir = ""
    else:
        uploaded_file = None
        statements_dir = st.text_input(
            "Statements directory",
            value=str(DATA_DIR),
            help="Example: data/statements or /absolute/path/to/statements",
        )

with setup_right:
    st.header("User goal")
    if input_mode == "Single CSV":
        goal = st.text_area("User goal", value=default_goal_single, height=105)
    else:
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


def format_report_month(value):
    if not value or value == "Not selected":
        return "Not selected"

    try:
        return datetime.strptime(str(value), "%Y-%m").strftime("%B %Y")
    except ValueError:
        return str(value)


def friendly_summary_label(kind, value):
    labels = {
        "month_field": {
            "transaction_month": "Purchase/activity month",
            "cashflow_month": "Bank posting/cashflow month",
            "statement_month": "Statement month",
        },
        "analysis_source": {
            "transactions_table": "Saved transactions database",
            "csv_fallback": "Uploaded CSV",
            "empty": "No transactions found",
        },
    }
    return labels.get(kind, {}).get(value, value or "Unknown")


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
    if "Merchant" in df.columns:
        df["Merchant"] = df["Merchant"].apply(format_merchant_display)
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
    if "Merchant" in df.columns:
        df["Merchant"] = df["Merchant"].apply(format_merchant_display)
    cols = ["Date", "Merchant", "Amount (NIS)", "Category", "Threshold", "Method"]
    return df[[c for c in cols if c in df.columns]]


def selected_row_index(selection_state):
    try:
        rows = selection_state.selection.rows
    except Exception:
        try:
            rows = selection_state.get("selection", {}).get("rows", [])
        except Exception:
            rows = []

    return rows[0] if rows else None


def report_transactions_df(report_month: str) -> pd.DataFrame:
    if not report_month or report_month == "Not selected":
        return pd.DataFrame()

    try:
        return load_spending_transactions_from_db(analysis_month=report_month)
    except Exception as e:
        st.warning(f"Could not load transaction details: {e}")
        return pd.DataFrame()


def prepare_transaction_details_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    details = df.copy()
    details = details.rename(columns={
        "transaction_date": "Transaction Date",
        "posting_date": "Posting Date",
        "charge_date": "Charge Date",
        "transaction_month": "Transaction Month",
        "cashflow_month": "Cashflow Month",
        "merchant": "Merchant",
        "description": "Description",
        "card_last4": "Card Last4",
        "account_last4": "Account Last4",
        "account_id": "Account ID",
        "counterparty": "Counterparty",
        "bank_reference": "Bank Reference",
        "category_en": "Stored Category",
        "normalized_category_en": "Analytics Category",
        "signed_amount": "Signed Amount",
        "amount": "Amount Abs",
        "currency": "Currency",
        "original_amount": "Original Amount",
        "original_currency": "Original Currency",
        "direction": "Direction",
        "source_type": "Source Type",
        "transaction_type": "Transaction Type",
    })
    if "Merchant" in details.columns:
        details["Merchant"] = details["Merchant"].apply(format_merchant_display)
    cols = [
        "Transaction Date",
        "Posting Date",
        "Charge Date",
        "Transaction Month",
        "Cashflow Month",
        "Merchant",
        "Description",
        "Card Last4",
        "Account Last4",
        "Account ID",
        "Counterparty",
        "Bank Reference",
        "Stored Category",
        "Analytics Category",
        "Signed Amount",
        "Amount Abs",
        "Currency",
        "Original Amount",
        "Original Currency",
        "Direction",
        "Source Type",
        "Transaction Type",
    ]
    existing_cols = [col for col in cols if col in details.columns]
    return details[existing_cols].sort_values(
        [col for col in ["Transaction Date", "Posting Date", "Merchant"] if col in existing_cols]
    )


def render_transaction_details(title: str, details_df: pd.DataFrame):
    if details_df.empty:
        st.info("No matching transaction details found.")
        return

    st.markdown(
        f"<div class='business-card'><b>{title}</b><br>"
        f"Transactions: {len(details_df)}</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        prepare_transaction_details_df(details_df),
        use_container_width=True,
        hide_index=True,
    )


SUPPORTED_TRANSACTION_SUFFIXES = {".csv", ".xlsx", ".xls"}


def save_transactions_first_step(paths: list[Path]):
    summaries = []
    total_received_count = 0
    inserted_count = 0
    skipped_duplicates_count = 0
    all_dates = []
    all_transaction_months = []
    all_cashflow_months = []
    missing_transaction_month_count = 0
    missing_cashflow_month_count = 0
    all_card_last4 = set()
    errors = []
    detected_file_kinds = set()
    source_types = set()

    for path in paths:
        try:
            transactions, detected_file_kind = parse_transactions_from_file(str(path))
            save_result = save_transactions_to_db(transactions, default_db_path())
            summary = summarize_transaction_import(transactions, save_result)
            summary["source_file"] = str(path)
            summary["detected_file_kind"] = detected_file_kind
            summary["source_types"] = sorted({item.get("source_type") for item in transactions})
            summaries.append(summary)
            detected_file_kinds.add(detected_file_kind)
            source_types.update(summary["source_types"])
        except Exception as e:
            errors.append(
                {
                    "source_file": str(path),
                    "error": str(e),
                }
            )
            continue

        total_received_count += summary.get("total_received_count", 0)
        inserted_count += summary.get("inserted_count", 0)
        skipped_duplicates_count += summary.get("skipped_duplicates_count", 0)

        for date_key in ["min_date", "max_date"]:
            if summary.get(date_key):
                all_dates.append(summary[date_key])

        # Keep separate month ranges for the two future analytics modes:
        # spending by purchase/activity month and cash flow by bank movement month.
        for month_key, target in [
            ("min_transaction_month", all_transaction_months),
            ("max_transaction_month", all_transaction_months),
            ("min_cashflow_month", all_cashflow_months),
            ("max_cashflow_month", all_cashflow_months),
        ]:
            if summary.get(month_key):
                target.append(summary[month_key])

        missing_transaction_month_count += summary.get("missing_transaction_month_count", 0)
        missing_cashflow_month_count += summary.get("missing_cashflow_month_count", 0)
        all_card_last4.update(summary.get("distinct_card_last4", []))

    duplicate_marking = mark_duplicate_transactions_in_db(default_db_path())

    return {
        "files_count": len(paths),
        "total_received_count": total_received_count,
        "inserted_count": inserted_count,
        "skipped_duplicates_count": skipped_duplicates_count,
        "min_date": min(all_dates) if all_dates else None,
        "max_date": max(all_dates) if all_dates else None,
        "min_transaction_month": min(all_transaction_months) if all_transaction_months else None,
        "max_transaction_month": max(all_transaction_months) if all_transaction_months else None,
        "min_cashflow_month": min(all_cashflow_months) if all_cashflow_months else None,
        "max_cashflow_month": max(all_cashflow_months) if all_cashflow_months else None,
        "missing_transaction_month_count": missing_transaction_month_count,
        "missing_cashflow_month_count": missing_cashflow_month_count,
        "distinct_card_last4": sorted(all_card_last4),
        "distinct_card_last4_count": len(all_card_last4),
        "detected_file_kinds": sorted(detected_file_kinds),
        "source_types": sorted(source_types),
        "duplicate_marking": duplicate_marking,
        "files": summaries,
        "errors": errors,
    }


def render_transaction_import_summary(summary):
    if not summary:
        return

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Transaction Import")

    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("Rows Received", summary.get("total_received_count", 0))
    with col2:
        kpi_card("Rows Inserted", summary.get("inserted_count", 0))
    with col3:
        kpi_card("Duplicates Skipped", summary.get("skipped_duplicates_count", 0))

    date_text = "No dates found"
    if summary.get("min_date") and summary.get("max_date"):
        date_text = f"{summary['min_date']} to {summary['max_date']}"

    transaction_month_text = "No transaction months found"
    if summary.get("min_transaction_month") and summary.get("max_transaction_month"):
        transaction_month_text = (
            f"{summary['min_transaction_month']} to {summary['max_transaction_month']}"
        )

    cashflow_month_text = "No cashflow months found"
    if summary.get("min_cashflow_month") and summary.get("max_cashflow_month"):
        cashflow_month_text = (
            f"{summary['min_cashflow_month']} to {summary['max_cashflow_month']}"
        )

    cards = summary.get("distinct_card_last4", [])
    cards_text = ", ".join(cards) if cards else "No card/account identifiers found"

    st.markdown(f"""
    <div class="business-card">
        <b>Files processed:</b> {summary.get("files_count", 0)}<br>
        <b>Date range:</b> {date_text}<br>
        <b>Transaction month range:</b> {transaction_month_text}<br>
        <b>Cashflow month range:</b> {cashflow_month_text}<br>
        <b>Missing transaction months:</b> {summary.get("missing_transaction_month_count", 0)}<br>
        <b>Missing cashflow months:</b> {summary.get("missing_cashflow_month_count", 0)}<br>
        <b>Distinct cards/accounts:</b> {cards_text}<br>
        <b>Detected file kinds:</b> {", ".join(summary.get("detected_file_kinds", [])) or "None"}<br>
        <b>Source types:</b> {", ".join(summary.get("source_types", [])) or "None"}<br>
        <b>Analytics excluded after marking:</b> {summary.get("duplicate_marking", {}).get("analytics_excluded_total", 0)}
    </div>
    """, unsafe_allow_html=True)

    if summary.get("errors"):
        with st.expander("Transaction import warnings", expanded=False):
            for item in summary["errors"]:
                st.warning(f"{item['source_file']}: {item['error']}")


def is_path_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_uploaded_finance_path(uploaded_name: str) -> tuple[Path | None, str]:
    filename = Path(uploaded_name).name

    if not filename or filename in {".", ".."}:
        return None, "Uploaded file name is not valid."

    if Path(filename).suffix.lower() not in SUPPORTED_TRANSACTION_SUFFIXES:
        return None, "Only CSV, XLSX, and XLS files are supported."

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

    finance_files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_TRANSACTION_SUFFIXES
    ]

    if not finance_files:
        return None, "Directory does not contain supported finance files."

    return directory, ""


def run_selected_agent():
    if input_mode == "Single CSV":
        if not uploaded_file:
            st.info("No new file selected. Running the agent on saved transactions.")
            return asyncio.run(run_agent(goal, csv_path=""))

        uploaded_path, error = safe_uploaded_finance_path(uploaded_file.name)
        if error:
            st.error(error)
            return None

        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if not uploaded_path.exists():
            st.error("Uploaded file was not saved correctly.")
            return None

        transaction_import = save_transactions_first_step([uploaded_path])
        st.success(f"File saved to: `{uploaded_path}`")

        if uploaded_path.suffix.lower() == ".csv":
            result = asyncio.run(run_agent(goal, str(uploaded_path)))
        else:
            result = asyncio.run(run_agent(goal, csv_path=""))

        result["transaction_import"] = transaction_import
        return result

    directory, error = validate_statements_directory(statements_dir)

    if error:
        st.warning(error)
        return None

    finance_files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_TRANSACTION_SUFFIXES
    )
    csv_files = [path for path in finance_files if path.suffix.lower() == ".csv"]
    transaction_import = save_transactions_first_step(finance_files)
    st.info(f"Directory selected: `{directory}`")

    if csv_files:
        result = asyncio.run(run_agent(goal, csv_path="", statements_dir=str(directory)))
    else:
        result = {
            "answer": "",
            "steps": [],
            "monthly_analysis_skipped": "Monthly report agent currently analyzes CSV files only.",
        }

    result["transaction_import"] = transaction_import
    return result


can_run = True if input_mode == "Single CSV" else bool(statements_dir.strip())

def render_agent_result(result):
    answer = result.get("answer", "")
    steps = result.get("steps", [])
    transaction_import = result.get("transaction_import")

    analysis = extract_json_from_step(steps, "finance.analyze_statement")
    unusual = extract_json_from_step(steps, "finance.find_unusual_expenses")
    ai_insights = extract_text_from_step(steps, "finance.generate_ai_financial_insights")

    if not analysis:
        analysis = parse_report_fallback(answer)

    if result.get("monthly_analysis_skipped"):
        st.info(result["monthly_analysis_skipped"])
        return

    report_month = format_report_month(analysis.get("analysis_month") or "Not selected")
    report_source = friendly_summary_label("analysis_source", analysis.get("analysis_source"))
    month_field = friendly_summary_label("month_field", analysis.get("month_field"))

    overview_tab, analytics_tab, ai_tab, technical_tab = st.tabs(
        ["Overview", "Analytics", "AI Insights", "Technical"]
    )

    with overview_tab:
        render_transaction_import_summary(transaction_import)
        st.header("Financial Summary")

        summary_left, summary_right = st.columns([1, 2], gap="large")
        with summary_left:
            st.markdown(f"""
            <div class="business-card">
                <b>Report month:</b> {report_month}<br>
                <b>Spending grouped by:</b> {month_field}<br>
                <b>Data source:</b> {report_source}
            </div>
            """, unsafe_allow_html=True)
        with summary_right:
            col1, col2, col3 = st.columns(3)
            with col1:
                kpi_card("Total Spent", f"{format_money(analysis.get('total_spent', 0))} NIS")
            with col2:
                kpi_card("Transactions", f"{analysis.get('transactions_count', 0)}")
            with col3:
                kpi_card("Average Transaction", f"{format_money(analysis.get('average_transaction', 0))} NIS")

    with analytics_tab:
        left, right = st.columns(2, gap="large")

        with left:
            st.subheader("🏆 Top Categories")
            categories_df = prepare_categories_df(analysis)
            if not categories_df.empty:
                category_selection = st.dataframe(
                    categories_df,
                    use_container_width=True,
                    hide_index=True,
                    key="top_categories_table",
                    on_select="rerun",
                    selection_mode="single-row",
                )
                category_index = selected_row_index(category_selection)
                if category_index is not None:
                    category_items = analysis.get("top_categories", [])
                    if category_index < len(category_items):
                        selected_category = category_items[category_index]
                        details = report_transactions_df(report_month)
                        details = details[
                            details["normalized_category"] == selected_category.get("category")
                        ]
                        render_transaction_details(
                            f"Details for {selected_category.get('category_en', 'selected category')}",
                            details,
                        )
            else:
                st.info("No category data found.")

        with right:
            st.subheader("🛒 Top Merchants")
            merchants_df = prepare_merchants_df(analysis)
            if not merchants_df.empty:
                merchant_selection = st.dataframe(
                    merchants_df,
                    use_container_width=True,
                    hide_index=True,
                    key="top_merchants_table",
                    on_select="rerun",
                    selection_mode="single-row",
                )
                merchant_index = selected_row_index(merchant_selection)
                if merchant_index is not None:
                    merchant_items = analysis.get("top_merchants", [])
                    if merchant_index < len(merchant_items):
                        selected_merchant = merchant_items[merchant_index]
                        details = report_transactions_df(report_month)
                        details = details[
                            details["merchant"] == selected_merchant.get("merchant")
                        ]
                        render_transaction_details(
                            f"Details for {format_merchant_display(selected_merchant.get('merchant', 'selected merchant'))}",
                            details,
                        )
            else:
                st.info("No merchant data found.")

        st.subheader("⚠️ Large Expenses to Review")
        expenses_df = prepare_expenses_df(unusual)
        if not expenses_df.empty:
            expense_selection = st.dataframe(
                expenses_df,
                use_container_width=True,
                hide_index=True,
                key="large_expenses_table",
                on_select="rerun",
                selection_mode="single-row",
            )
            expense_index = selected_row_index(expense_selection)
            if expense_index is not None:
                expense_items = unusual.get("large_expenses_to_review", []) if unusual else []
                if expense_index < len(expense_items):
                    selected_expense = expense_items[expense_index]
                    details = report_transactions_df(report_month)
                    selected_amount = float(selected_expense.get("amount", 0) or 0)
                    details = details[
                        (details["transaction_date"] == selected_expense.get("transaction_date")) &
                        (details["merchant"] == selected_expense.get("merchant")) &
                        ((details["amount"] - selected_amount).abs() <= 0.01)
                    ]
                    render_transaction_details(
                        f"Details for {format_merchant_display(selected_expense.get('merchant', 'selected expense'))}",
                        details,
                    )
        else:
            st.success("No large expenses detected.")

    with ai_tab:
        st.header("AI Insights")
        if ai_insights:
            st.markdown(ai_insights)
        else:
            st.info("AI insights were not generated for this run.")

    with technical_tab:
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


def render_financial_qa():
    st.header("Financial Questions")
    st.caption("Ask a question about your saved transaction history.")

    spending_insight_examples = [
        "Where is most of my money actually going?",
        "What changed the most between May and June?",
        "Which expenses deserve a second look?",
    ]

    recurring_commitment_examples = [
        "Which recurring payments are real financial obligations and which are just payment methods?",
        "What are my biggest recurring financial commitments?",
        "Which recurring payments would you never recommend cutting?",
    ]

    ai_review_examples = [
        "If you could give me only one piece of financial advice, what would it be?",
        "What spending habit would save me the most money over time?",
        "Which expenses look expensive, but are actually perfectly normal?",
        "If you knew nothing about me except these transactions, what would you guess about my lifestyle?",
    ]

    def example_selector(label: str, options: list[str], key: str):
        st.caption(label)
        if hasattr(st, "pills"):
            return st.pills(
                label,
                options,
                selection_mode="single",
                key=key,
                label_visibility="collapsed",
            )
        return st.selectbox(
            label,
            options,
            index=None,
            key=key,
            placeholder="Choose an example",
        )

    selected_spending_example = example_selector(
        "📊 Spending Insights",
        spending_insight_examples,
        "spending_insight_question_example",
    )
    selected_commitment_example = example_selector(
        "💳 Recurring Commitments",
        recurring_commitment_examples,
        "recurring_commitment_question_example",
    )
    selected_ai_review_example = example_selector(
        "🤖 AI Financial Review",
        ai_review_examples,
        "ai_review_question_example",
    )
    selected_example = (
        selected_ai_review_example
        or selected_commitment_example
        or selected_spending_example
    )
    if selected_example:
        st.session_state["financial_qa_question"] = selected_example

    qa_question = st.text_input(
        "Ask a question about your transaction history",
        key="financial_qa_question",
        placeholder="Compare May and June spending.",
    )

    if st.button("Ask", disabled=not qa_question.strip()):
        with st.spinner("Answering from transaction history..."):
            st.session_state["last_financial_qa"] = asyncio.run(
                run_financial_question(qa_question.strip())
            )

    if st.session_state.get("last_financial_qa"):
        qa_result = st.session_state["last_financial_qa"]
        response_mode = qa_result.get("response_mode", "interpretive")
        answer_labels = {
            "deterministic": "📊 Deterministic Answer",
            "interpretive": "🤖 AI Interpretation",
            "insight": "🤖 AI Interpretation",
        }
        answer_label = answer_labels.get(response_mode, "🤖 AI Interpretation")
        st.markdown(f"""
        <div class="business-card">
            <b>Question:</b> {qa_result.get("question", "")}<br>
            <b>Tool selected:</b> {qa_result.get("tool", "")}<br>
            <b>Answer mode:</b> {answer_label}<br>
            <b>Insight subtype:</b> {qa_result.get("insight_subtype", "") or "n/a"}
        </div>
        """, unsafe_allow_html=True)
        st.subheader(answer_label)
        key_facts = qa_result.get("key_facts", "")
        if response_mode == "insight" and key_facts:
            st.markdown(key_facts)
            st.markdown("**AI Commentary**")
        st.markdown(qa_result.get("answer", ""))

        # Show the NLP decision path for demos, debugging, and project review.
        with st.expander("NLP debug path", expanded=False):
            nlp_debug = qa_result.get("nlp_debug", {})
            st.write(f"deterministic_router = {nlp_debug.get('deterministic_router', 'n/a')}")
            st.write(f"llm_classifier_used = {nlp_debug.get('llm_classifier_used', 'n/a')}")
            st.write(f"subtype = {nlp_debug.get('subtype', qa_result.get('insight_subtype') or 'n/a')}")
            st.write(f"focused_context_keys = {nlp_debug.get('focused_context_keys', [])}")
            st.write(f"guard = {nlp_debug.get('guard', 'n/a')}")
            st.json(nlp_debug)

        with st.expander("Raw JSON result", expanded=False):
            raw_result = qa_result.get("raw_result", "")
            try:
                st.json(json.loads(raw_result))
            except Exception:
                st.code(str(raw_result))


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
run_col, status_col = st.columns([1, 3])
with run_col:
    run_clicked = st.button("🚀 Run Agent", disabled=not can_run, use_container_width=True)
with status_col:
    if st.session_state.get("last_agent_result"):
        st.success("Agent result is loaded. Use the tabs below to review it.")
    elif input_mode == "Single CSV":
        st.info("Run the agent with saved transactions, or upload a new finance file first.")
    else:
        st.info("Enter a directory path containing finance statements.")

if run_clicked:
    with st.spinner("Agent is working..."):
        result = run_selected_agent()

    if result:
        st.session_state["last_agent_result"] = result

overview_area, qa_area = st.tabs(["Dashboard", "Financial Q&A"])

with overview_area:
    if st.session_state.get("last_agent_result"):
        render_agent_result(st.session_state["last_agent_result"])
    elif input_mode == "Single CSV":
        st.info("Run the agent with saved transactions, or upload a new finance file first.")
    else:
        st.info("Enter a directory path containing finance statements.")

with qa_area:
    render_financial_qa()
