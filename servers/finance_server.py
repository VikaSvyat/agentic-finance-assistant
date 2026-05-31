from mcp.server.fastmcp import FastMCP
import pandas as pd
import json
from pathlib import Path

mcp = FastMCP("finance-mcp")


def load_bank_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    raw = pd.read_csv(path, sep=";", header=None)

    header_row = None
    for i, row in raw.iterrows():
        values = row.astype(str).tolist()
        if "תאריך עסקה" in values and "שם בית העסק" in values:
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not find Hebrew bank statement header row")

    df = pd.read_csv(path, sep=";", header=header_row)

    df = df.rename(
        columns={
            "תאריך עסקה": "transaction_date",
            "שם בית העסק": "merchant",
            "קטגוריה": "category",
            "סכום חיוב": "amount",
            "מטבע חיוב": "currency",
            "תאריך חיוב": "charge_date",
            "סוג עסקה": "transaction_type",
            "4 ספרות אחרונות של כרטיס האשראי": "card_last_4",
        }
    )

    required = ["transaction_date", "merchant", "category", "amount"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required + [c for c in ["currency", "charge_date", "transaction_type", "card_last_4"] if c in df.columns]]

    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace("₪", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")    

 
    df = df.dropna(subset=["amount"])
    df = df[df["merchant"].notna()]

    return df

def add_normalized_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized_category and normalized_category_en columns.

    Bank category 'דלק, חשמל וגז' is too broad, so we split it into:
    - Fuel
    - Electricity
    - Gas
    - Utilities
    """

    df = df.copy()

    def normalize(row):
        original_category = str(row["category"])
        merchant = str(row["merchant"])

        if original_category == "דלק, חשמל וגז":
            fuel_keywords = [
                "פז",
                "YELLOW",
                "yellow",
                "סונול",
                "דור אלון",
                "דלק",
            ]

            electricity_keywords = [
                "חשמל",
                "חברת חשמל",
            ]

            gas_keywords = [
                "גז",
                "אמישראגז",
                "פזגז",
            ]

            if any(keyword in merchant for keyword in fuel_keywords):
                return "fuel"

            if any(keyword in merchant for keyword in electricity_keywords):
                return "electricity"

            if any(keyword in merchant for keyword in gas_keywords):
                return "gas"

            return "utilities"

        category_map = {
            "מזון וצריכה": "food_and_groceries",
            "רפואה ובתי מרקחת": "health_and_pharmacies",
            "מסעדות, קפה וברים": "restaurants_cafes_bars",
            "תחבורה ורכבים": "transport_and_vehicles",
            "עיצוב הבית": "home_and_furniture",
            "עירייה וממשלה": "government_and_municipality",
            "שונות": "other",
        }

        return category_map.get(original_category, "other")

    category_labels = {
        "fuel": "Fuel",
        "electricity": "Electricity",
        "gas": "Gas",
        "utilities": "Utilities",
        "food_and_groceries": "Food and groceries",
        "health_and_pharmacies": "Health and pharmacies",
        "restaurants_cafes_bars": "Restaurants, cafes and bars",
        "transport_and_vehicles": "Transport and vehicles",
        "home_and_furniture": "Home and furniture",
        "government_and_municipality": "Government and municipality",
        "other": "Other",
    }

    df["normalized_category"] = df.apply(normalize, axis=1)
    df["normalized_category_en"] = df["normalized_category"].map(category_labels)

    return df

@mcp.tool()
def categorize_transactions(csv_path: str) -> str:
    """
    Return transactions with original bank category and normalized category.
    """

    df = load_bank_csv(csv_path)
    df = add_normalized_categories(df)

    result = df[
        [
            "transaction_date",
            "merchant",
            "category",
            "normalized_category",
            "normalized_category_en",
            "amount",
        ]
    ].to_dict(orient="records")

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def analyze_statement(csv_path: str) -> str:
    """
    Analyze monthly bank statement and return a clean structured summary.
    Merchant names stay in original Hebrew.
    Categories are shown in Hebrew and English.
    """

    df = load_bank_csv(csv_path)

    df = add_normalized_categories(df)

    category_translation = {
        "מזון וצריכה": "Food and groceries",
        "דלק, חשמל וגז": "Fuel, electricity and gas",
        "רפואה ובתי מרקחת": "Health and pharmacies",
        "מסעדות, קפה וברים": "Restaurants, cafes and bars",
        "תחבורה ורכבים": "Transport and vehicles",
        "עיצוב הבית": "Home and furniture",
        "עירייה וממשלה": "Government and municipality",
        "שונות": "Other",
    }

    total_spent = round(float(df["amount"].sum()), 2)
    transactions_count = int(len(df))
    average_transaction = round(float(df["amount"].mean()), 2)

    categories_grouped = (
        df.groupby(["normalized_category", "normalized_category_en"])["amount"]
        .agg(["sum", "count", "mean"])
        .sort_values("sum", ascending=False)
        .head(10)
        .reset_index()
    )

    top_categories = []

    for _, row in categories_grouped.iterrows():
        top_categories.append(
            {
                "category": row["normalized_category"],
                "category_en": row["normalized_category_en"],
                "total_amount": round(float(row["sum"]), 2),
                "transactions_count": int(row["count"]),
                "average_transaction": round(float(row["mean"]), 2),
            }
        )

    merchants_grouped = (
        df.groupby("merchant")["amount"]
        .agg(["sum", "count", "mean"])
        .sort_values("sum", ascending=False)
        .head(10)
        .reset_index()
    )

    top_merchants = []

    for _, row in merchants_grouped.iterrows():
        top_merchants.append(
            {
                "merchant": row["merchant"],
                "total_amount": round(float(row["sum"]), 2),
                "transactions_count": int(row["count"]),
                "average_transaction": round(float(row["mean"]), 2),
            }
        )

    result = {
        "currency": "NIS",
        "total_spent": total_spent,
        "transactions_count": transactions_count,
        "average_transaction": average_transaction,
        "top_categories": top_categories,
        "top_merchants": top_merchants,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.tool()
def get_category_breakdown(csv_path: str) -> str:
    """
    Return spending grouped by normalized category.
    """

    df = load_bank_csv(csv_path)
    df = add_normalized_categories(df)

    grouped = (
        df.groupby(["normalized_category", "normalized_category_en"])["amount"]
        .agg(["sum", "count", "mean"])
        .sort_values("sum", ascending=False)
        .reset_index()
    )

    result = []

    for _, row in grouped.iterrows():
        result.append(
            {
                "category": row["normalized_category"],
                "category_en": row["normalized_category_en"],
                "total_amount": round(float(row["sum"]), 2),
                "transactions_count": int(row["count"]),
                "average_transaction": round(float(row["mean"]), 2),
            }
        )

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_top_merchants(csv_path: str, limit: int = 10) -> str:
    """
    Return top merchants by total spending.
    """
    df = load_bank_csv(csv_path)

    result = (
        df.groupby("merchant")["amount"]
        .agg(["sum", "count", "mean"])
        .sort_values("sum", ascending=False)
        .head(limit)
        .reset_index()
        .to_dict(orient="records")
    )

    return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.tool()
def generate_monthly_report(analysis_json: str, unusual_json: str, advice_text: str) -> str:
    """
    Generate a clean monthly finance report from tool outputs.
    """

    analysis = json.loads(analysis_json)
    unusual = json.loads(unusual_json)

    lines = []

    lines.append("# Monthly Finance Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total spent: {analysis.get('total_spent', 0):.2f} NIS")
    lines.append(f"- Transactions: {analysis.get('transactions_count', 0)}")
    lines.append(f"- Average transaction: {analysis.get('average_transaction', 0):.2f} NIS")
    lines.append("")

    lines.append("## Top Categories")
    for item in analysis.get("top_categories", [])[:5]:
        lines.append(
            f"- {item.get('category_en')}: {item.get('total_amount', 0):.2f} NIS "
            f"({item.get('transactions_count', 0)} transactions)"
        )

    lines.append("")
    lines.append("## Top Merchants")
    for item in analysis.get("top_merchants", [])[:5]:
        lines.append(
            f"- {item.get('merchant')}: {item.get('total_amount', 0):.2f} NIS "
            f"({item.get('transactions_count', 0)} transactions)"
        )

    large_expenses = unusual.get("large_expenses_to_review", [])

    lines.append("")
    lines.append("## Large Expenses to Review")
    if large_expenses:
        for item in large_expenses[:5]:
            lines.append(
                f"- {item.get('transaction_date')}: {item.get('merchant')} — "
                f"{item.get('amount')} NIS ({item.get('normalized_category_en')})"
            )
    else:
        lines.append("- No large expenses detected.")

    lines.append("")
    lines.append("## Savings Advice")
    lines.append(advice_text)

    return "\n".join(lines)


@mcp.tool()
def find_unusual_expenses(csv_path: str) -> str:
    """
    Find large expenses to review using merchant-specific thresholds first,
    then normalized-category fallback.
    """

    df = load_bank_csv(csv_path)
    df = add_normalized_categories(df)
    df = df.copy()

    min_merchant_transactions = 3
    min_category_transactions = 3

    merchant_thresholds = {}
    category_thresholds = {}

    for merchant in df["merchant"].unique():
        merchant_df = df[df["merchant"] == merchant]
        count = len(merchant_df)

        if count >= min_merchant_transactions:
            threshold = merchant_df["amount"].quantile(0.95)
            merchant_thresholds[merchant] = {
                "threshold": round(float(threshold), 2),
                "transactions_count": int(count),
                "method": "merchant_95th_percentile",
            }

    for category in df["normalized_category"].unique():
        category_df = df[df["normalized_category"] == category]
        count = len(category_df)

        if count >= min_category_transactions:
            threshold = category_df["amount"].quantile(0.95)
            method = "normalized_category_95th_percentile"
        else:
            threshold = df["amount"].quantile(0.90)
            method = "global_90th_percentile_fallback"

        category_thresholds[category] = {
            "threshold": round(float(threshold), 2),
            "transactions_count": int(count),
            "method": method,
        }

    def get_threshold(row):
        merchant = row["merchant"]
        category = row["normalized_category"]

        if merchant in merchant_thresholds:
            return merchant_thresholds[merchant]["threshold"]

        return category_thresholds[category]["threshold"]

    def get_method(row):
        merchant = row["merchant"]
        category = row["normalized_category"]

        if merchant in merchant_thresholds:
            return merchant_thresholds[merchant]["method"]

        return category_thresholds[category]["method"]

    df["threshold"] = df.apply(get_threshold, axis=1)
    df["threshold_method"] = df.apply(get_method, axis=1)

    unusual = df[df["amount"] > df["threshold"]].copy()

    unusual = unusual[
        [
            "transaction_date",
            "merchant",
            "category",
            "normalized_category",
            "normalized_category_en",
            "amount",
            "threshold",
            "threshold_method",
        ]
    ].sort_values("amount", ascending=False)

    result = {
        "method": "merchant threshold first, normalized-category fallback",
        "rule": "if merchant has 3+ transactions use merchant 95th percentile, otherwise use normalized category threshold",
        "merchant_thresholds": merchant_thresholds,
        "category_thresholds": category_thresholds,
        "large_expenses_to_review": unusual.to_dict(orient="records"),
    }

    return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.tool()
def generate_savings_advice(analysis_json: str, unusual_json: str) -> str:
    """
    Generate savings advice based only on previous tool outputs.
    This keeps orchestration in the LLM/client, not inside this tool.
    """

    if not analysis_json or not analysis_json.strip():
        return "Could not generate advice: analysis_json is empty. Please call finance.analyze_statement first."

    if not unusual_json or not unusual_json.strip():
        return "Could not generate advice: unusual_json is empty. Please call finance.find_unusual_expenses first."

    analysis = json.loads(analysis_json)
    unusual = json.loads(unusual_json)

    total_spent = analysis.get("total_spent", 0)
    top_categories = analysis.get("top_categories", [])
    top_merchants = analysis.get("top_merchants", [])

    advice = []

    advice.append(f"Total monthly spending: {total_spent:.2f} NIS.")

    if top_categories:
        top_category = top_categories[0]

        category_code = top_category.get("category", "")
        category_en = top_category.get("category_en", "Unknown")
        top_amount = top_category.get("total_amount", 0)

        percent = (top_amount / total_spent * 100) if total_spent else 0

        advice.append(
            f"Biggest category: {category_en} — {top_amount:.2f} NIS, {percent:.1f}% of total spending."
        )

        controllable_categories = [
            "food_and_groceries",
            "restaurants_cafes_bars",
            "home_and_furniture",
            "other",
            "fuel",
        ]

        if category_code in controllable_categories:
            advice.append(
                f"Potential saving: try reducing spending in {category_en} by 10-15% next month."
            )
        else:
            advice.append(
                f"{category_en} may include necessary expenses, so review it before cutting it."
            )

    if top_merchants:
        advice.append("")
        advice.append("Top merchants to review:")
        for merchant in top_merchants[:5]:
            advice.append(
                f"- {merchant.get('merchant')}: {merchant.get('total_amount', 0):.2f} NIS "
                f"across {merchant.get('transactions_count', 0)} transactions"
            )

    large_expenses = unusual.get("large_expenses_to_review", [])

    if large_expenses:
        advice.append("")
        advice.append("Large expenses to review:")
        for item in large_expenses[:5]:
            advice.append(
                f"- {item.get('transaction_date')}: {item.get('merchant')} — "
                f"{item.get('amount')} NIS ({item.get('normalized_category_en')})"
            )

    advice.append("")
    advice.append("Suggested next step: set monthly budget limits for the top 3 normalized categories.")

    return "\n".join(advice)


if __name__ == "__main__":
    mcp.run()