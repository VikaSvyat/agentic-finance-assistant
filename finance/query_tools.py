import os
import re
import sqlite3
import calendar
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_ANALYTICS_CURRENCY = "NIS"
ACTIVE_ANALYTICS_WHERE = "COALESCE(analytics_excluded, 0) = 0"
CURRENCY_SYMBOLS = {
    "NIS": "₪",
    "USD": "$",
}
SPENDING_TRANSACTION_TYPES = {
    "real_expense",
    "standing_order",
    "housing_utility_expense",
    "unknown",
}

CATEGORY_LABELS = {
    "rent_and_utilities": "Rent and utilities",
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
    "flights_and_travel": "Flights and travel",
    "leisure_entertainment_sports": "Leisure, entertainment and sports",
    "communication_services": "Communication services",
    "education_and_courses": "Education and courses",
    "other": "Other",
}

HEBREW_CATEGORY_KEYS = {
    "מזון וצריכה": "food_and_groceries",
    "רפואה ובתי מרקחת": "health_and_pharmacies",
    "מסעדות, קפה וברים": "restaurants_cafes_bars",
    "תחבורה ורכבים": "transport_and_vehicles",
    "עיצוב הבית": "home_and_furniture",
    "עירייה וממשלה": "government_and_municipality",
    "טיסות ותיירות": "flights_and_travel",
    "פנאי, בידור וספורט": "leisure_entertainment_sports",
    "שירותי תקשורת": "communication_services",
    "שונות": "other",
}

ENGLISH_CATEGORY_KEYS = {
    label.casefold(): key
    for key, label in CATEGORY_LABELS.items()
}

TRANSACTION_TYPE_MERCHANT_LABELS = {
    "העברה דיגיטל",
    "הוראת קבע",
    "דמי כרטיס",
}

RECURRING_PAYMENT_CATEGORIES = {
    "communication_services",
    "rent_and_utilities",
    "electricity",
    "gas",
    "utilities",
    "government_and_municipality",
    "education_and_courses",
}

MANDATORY_SPENDING_CATEGORIES = {
    "rent_and_utilities",
    "electricity",
    "gas",
    "utilities",
    "government_and_municipality",
    "education_and_courses",
    "communication_services",
}

DISCRETIONARY_SPENDING_CATEGORIES = {
    "restaurants_cafes_bars",
    "leisure_entertainment_sports",
    "home_and_furniture",
    "flights_and_travel",
}

FOOD_DELIVERY_KEYWORDS = {
    "wolt",
    "תן ביס",
    "10bis",
    "deliver",
    "delivery",
}

SUBSCRIPTION_KEYWORDS = {
    "subscription",
    "מנוי",
    "apple.com/bill",
    "itunes",
}

RECURRING_PAYMENT_KEYWORDS = {
    "communication_services": [
        "hot",
        "hot mobile",
        "בזק",
        "תקשורת",
    ],
    "insurance": [
        "insurance",
        "ביטוח",
        "מגדל",
        "הראל",
        "כלל",
        "הפניקס",
    ],
    "pension": [
        "pension",
        "פנס",
        "פנסיה",
        "גמל",
        "מור גמל",
    ],
    "education_and_courses": [
        "school",
        "tuition",
        "שכ\"ל",
        "שכר לימוד",
        "מרכז חינוך",
        "course",
    ],
    "utilities": [
        "חשמל",
        "חברת החשמל",
        "גז",
        "מים",
        "ארנונה",
        "עיריית",
        "עירית",
    ],
    "standing_order": [
        "הוראת קבע",
        "הו\"ק",
    ],
}

MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def default_db_path() -> str:
    return os.getenv("DB_PATH", "./database/finance.db")


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def db_connect(db_path: str | None = None) -> sqlite3.Connection:
    path = Path(db_path or default_db_path())
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def spending_type_placeholders() -> str:
    return ", ".join("?" for _ in SPENDING_TRANSACTION_TYPES)


def latest_transaction_month(conn: sqlite3.Connection) -> str | None:
    query = f"""
        SELECT MAX(transaction_month)
        FROM transactions
        WHERE {ACTIVE_ANALYTICS_WHERE}
          AND direction = 'debit'
          AND amount_abs > 0
          AND currency = ?
          AND transaction_type IN ({spending_type_placeholders()})
          AND transaction_month IS NOT NULL
    """
    return conn.execute(
        query,
        (BASE_ANALYTICS_CURRENCY, *tuple(SPENDING_TRANSACTION_TYPES)),
    ).fetchone()[0]


def shift_month(month: str, offset: int) -> str:
    year, month_num = [int(part) for part in month.split("-")]
    month_num += offset
    while month_num < 1:
        year -= 1
        month_num += 12
    while month_num > 12:
        year += 1
        month_num -= 12
    return f"{year:04d}-{month_num:02d}"


def month_coverage_metadata(month: str) -> dict[str, Any]:
    year, month_num = [int(part) for part in month.split("-")]
    expected_days = calendar.monthrange(year, month_num)[1]
    today = datetime.now().date()
    month_index = year * 12 + month_num
    current_index = today.year * 12 + today.month

    if month_index < current_index:
        coverage_days = expected_days
    elif month_index == current_index:
        coverage_days = min(today.day, expected_days)
    else:
        coverage_days = 0

    return {
        "month": month,
        "is_complete_month": coverage_days >= expected_days,
        "coverage_days": coverage_days,
        "expected_days": expected_days,
        "coverage_note": (
            "Complete historical month."
            if coverage_days >= expected_days
            else "Partial current/future month; comparisons may reflect incomplete data."
        ),
    }


def month_range(start_month: str, end_month: str) -> list[str]:
    start_year, start_num = [int(part) for part in start_month.split("-")]
    end_year, end_num = [int(part) for part in end_month.split("-")]
    start_index = start_year * 12 + start_num
    end_index = end_year * 12 + end_num
    if start_index > end_index:
        start_month, end_month = end_month, start_month

    months = [start_month]
    while months[-1] != end_month:
        months.append(shift_month(months[-1], 1))
    return months


def resolve_month_range(month_text: str, conn: sqlite3.Connection) -> list[str]:
    text = normalize_text(month_text).casefold()
    match = re.fullmatch(r"(\d{4}-\d{2})\s*(?::|\.{2}|to|-)\s*(\d{4}-\d{2})", text)
    if not match:
        return []

    start_month = resolve_month(match.group(1), conn)
    end_month = resolve_month(match.group(2), conn)
    return month_range(start_month, end_month)


def resolve_month(month: str | None, conn: sqlite3.Connection) -> str:
    text = normalize_text(month).casefold()

    if re.fullmatch(r"\d{4}-\d{2}", text):
        return text

    if text in {"", "latest", "last available month"}:
        resolved = latest_transaction_month(conn)
        if resolved:
            return resolved

    if text in {"current month", "this month", "current"}:
        return datetime.now().strftime("%Y-%m")

    if text in {"last month", "previous month"}:
        return shift_month(datetime.now().strftime("%Y-%m"), -1)

    if text in MONTH_NAMES:
        current_year = datetime.now().year
        return f"{current_year:04d}-{MONTH_NAMES[text]:02d}"

    raise ValueError(f"Could not resolve month: {month!r}. Use YYYY-MM or a supported month name.")


def is_rent_or_utility_payment(*values: object) -> bool:
    text = " ".join(normalize_text(value) for value in values).casefold()
    keywords = [
        "rent",
        "שכירות",
        "ארנונה",
        "עירית חיפה",
        "עיריית חיפה",
        "iriyat haifa",
        "haifa municipality",
        "מור גמל ופנס-י",
        "מור גמל ופנס",
    ]
    return any(keyword.casefold() in text for keyword in keywords)


def is_education_or_course_payment(*values: object) -> bool:
    text = " ".join(normalize_text(value) for value in values).casefold()
    keywords = ["riseguide", "english", "course", "לימוד", "למידה", "אנגלית"]
    return any(keyword.casefold() in text for keyword in keywords)


def normalize_category_key(
    category: object,
    category_en: object,
    merchant: object,
    description: object = "",
) -> str:
    category_text = normalize_text(category)
    category_en_text = normalize_text(category_en).casefold()
    merchant_text = normalize_text(merchant)

    if is_rent_or_utility_payment(category, category_en, merchant, description):
        return "rent_and_utilities"

    if is_education_or_course_payment(category, category_en, merchant, description):
        return "education_and_courses"

    if category_text == "דלק, חשמל וגז":
        fuel_keywords = ["פז", "YELLOW", "yellow", "סונול", "דור אלון", "דלק"]
        electricity_keywords = ["חשמל", "חברת חשמל"]
        gas_keywords = ["גז", "אמישראגז", "פזגז"]

        if any(keyword in merchant_text for keyword in fuel_keywords):
            return "fuel"
        if any(keyword in merchant_text for keyword in electricity_keywords):
            return "electricity"
        if any(keyword in merchant_text for keyword in gas_keywords):
            return "gas"
        return "utilities"

    if category_text in HEBREW_CATEGORY_KEYS:
        return HEBREW_CATEGORY_KEYS[category_text]

    if category_en_text in ENGLISH_CATEGORY_KEYS:
        return ENGLISH_CATEGORY_KEYS[category_en_text]

    return "other"


def category_label(key: str) -> str:
    return CATEGORY_LABELS.get(key, "Other")


def transaction_type_usage_hint(category_key: str, category_en: object = "") -> str:
    if category_key == "other":
        return ""
    label = category_label(category_key)
    if category_key == "pension":
        return "used for pension transfers"
    if category_key == "insurance":
        return "used for insurance payments"
    if category_key == "education_and_courses":
        return "used for education payments"
    if category_key == "rent_and_utilities":
        return "used for paying rent and utilities"
    if category_key in {"electricity", "gas", "utilities"}:
        return f"used for paying {label.lower()}"
    if category_key == "communication_services":
        return "used for communication service payments"
    if category_key == "government_and_municipality":
        return "used for government or municipality payments"
    if category_en:
        return f"used for {str(category_en).strip().lower()}"
    return f"used for {label.lower()}"


def normalize_category_query(category: str) -> str:
    text = normalize_text(category)
    lowered = text.casefold()

    if lowered in CATEGORY_LABELS:
        return lowered
    if lowered in ENGLISH_CATEGORY_KEYS:
        return ENGLISH_CATEGORY_KEYS[lowered]
    if text in HEBREW_CATEGORY_KEYS:
        return HEBREW_CATEGORY_KEYS[text]

    return lowered.replace(" ", "_")


def load_spending_rows(month: str, db_path: str | None = None) -> pd.DataFrame:
    with db_connect(db_path) as conn:
        resolved_month = resolve_month(month, conn)
        params = [BASE_ANALYTICS_CURRENCY, *list(SPENDING_TRANSACTION_TYPES), resolved_month]
        query = f"""
            SELECT
                id,
                transaction_date,
                posting_date,
                charge_date,
                transaction_month,
                cashflow_month,
                merchant,
                description,
                category,
                category_en,
                amount AS signed_amount,
                amount_abs AS amount,
                currency,
                direction,
                source_type,
                transaction_type,
                card_last4,
                account_last4,
                account_id,
                counterparty,
                bank_reference,
                original_amount,
                original_currency
            FROM transactions
            WHERE {ACTIVE_ANALYTICS_WHERE}
              AND direction = 'debit'
              AND amount_abs > 0
              AND currency = ?
              AND transaction_type IN ({spending_type_placeholders()})
              AND transaction_month = ?
        """
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        df.attrs["resolved_month"] = resolved_month
        return df

    df["merchant"] = df["merchant"].fillna(df["description"]).fillna("Unknown")
    df["category"] = df["category"].fillna("")
    df["category_en"] = df["category_en"].fillna("")
    df["normalized_category"] = df.apply(
        lambda row: normalize_category_key(
            row.get("category"),
            row.get("category_en"),
            row.get("merchant"),
            row.get("description"),
        ),
        axis=1,
    )
    df["normalized_category_en"] = df["normalized_category"].apply(category_label)
    df["resolved_month"] = resolved_month
    df.attrs["resolved_month"] = resolved_month
    return df


def net_spending_amount(direction: object, amount_abs: object) -> float:
    amount = float(amount_abs or 0)
    if normalize_text(direction).casefold() == "credit":
        return -amount
    return amount


def exclude_mirrored_review_refunds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    credit_df = df[df["direction"].fillna("").str.casefold() == "credit"]
    card_credit_keys = {
        (
            normalize_text(row.get("transaction_date")),
            normalize_text(row.get("merchant")).casefold(),
            round(float(row.get("amount") or 0), 2),
            normalize_text(row.get("card_last4")),
        )
        for _, row in credit_df.iterrows()
        if normalize_text(row.get("source_type")).casefold() != "bank_account"
    }

    if not card_credit_keys:
        return df

    mirrored_mask = []
    for _, row in df.iterrows():
        key = (
            normalize_text(row.get("transaction_date")),
            normalize_text(row.get("merchant")).casefold(),
            round(float(row.get("amount") or 0), 2),
            normalize_text(row.get("card_last4")),
        )
        mirrored_mask.append(
            normalize_text(row.get("direction")).casefold() == "credit"
            and normalize_text(row.get("source_type")).casefold() == "bank_account"
            and key in card_credit_keys
        )

    return df.loc[[not value for value in mirrored_mask]].copy()


def load_review_spending_rows(month: str, db_path: str | None = None) -> pd.DataFrame:
    with db_connect(db_path) as conn:
        resolved_month = resolve_month(month, conn)
        params = [BASE_ANALYTICS_CURRENCY, *list(SPENDING_TRANSACTION_TYPES), resolved_month]
        query = f"""
            SELECT
                id,
                transaction_date,
                posting_date,
                charge_date,
                transaction_month,
                cashflow_month,
                merchant,
                description,
                category,
                category_en,
                amount AS signed_amount,
                amount_abs AS amount,
                currency,
                direction,
                source_type,
                transaction_type,
                card_last4,
                account_last4,
                account_id,
                counterparty,
                bank_reference,
                original_amount,
                original_currency
            FROM transactions
            WHERE {ACTIVE_ANALYTICS_WHERE}
              AND direction IN ('debit', 'credit')
              AND amount_abs > 0
              AND currency = ?
              AND transaction_type IN ({spending_type_placeholders()})
              AND transaction_month = ?
              AND (
                    direction = 'debit'
                    OR (
                        direction = 'credit'
                        AND source_type != 'bank_account'
                        AND COALESCE(category_en, category, '') != ''
                    )
                  )
        """
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        df.attrs["resolved_month"] = resolved_month
        return df

    df["merchant"] = df["merchant"].fillna(df["description"]).fillna("Unknown")
    df["category"] = df["category"].fillna("")
    df["category_en"] = df["category_en"].fillna("")
    df = exclude_mirrored_review_refunds(df)
    df["normalized_category"] = df.apply(
        lambda row: normalize_category_key(
            row.get("category"),
            row.get("category_en"),
            row.get("merchant"),
            row.get("description"),
        ),
        axis=1,
    )
    df["normalized_category_en"] = df["normalized_category"].apply(category_label)
    df["review_amount"] = df.apply(
        lambda row: net_spending_amount(row.get("direction"), row.get("amount")),
        axis=1,
    )
    df["resolved_month"] = resolved_month
    df.attrs["resolved_month"] = resolved_month
    return df


def get_review_spending_summary(month: str, db_path: str | None = None) -> dict[str, Any]:
    df = load_review_spending_rows(month, db_path)
    resolved_month = resolved_month_from_df(df, month)

    if df.empty:
        return {
            "month": resolved_month,
            "currency": BASE_ANALYTICS_CURRENCY,
            "currency_symbol": CURRENCY_SYMBOLS.get(BASE_ANALYTICS_CURRENCY, BASE_ANALYTICS_CURRENCY),
            **month_coverage_metadata(resolved_month),
            "total_spent": 0,
            "transactions": 0,
            "debit_transactions": 0,
            "credit_transactions": 0,
            "calculation_basis": "net_debits_minus_credits",
        }

    total = round(float(df["review_amount"].sum()), 2)
    return {
        "month": resolved_month,
        "currency": BASE_ANALYTICS_CURRENCY,
        "currency_symbol": CURRENCY_SYMBOLS.get(BASE_ANALYTICS_CURRENCY, BASE_ANALYTICS_CURRENCY),
        **month_coverage_metadata(resolved_month),
        "total_spent": total,
        "transactions": int(len(df)),
        "debit_transactions": int((df["direction"].fillna("").str.casefold() == "debit").sum()),
        "credit_transactions": int((df["direction"].fillna("").str.casefold() == "credit").sum()),
        "calculation_basis": "net_debits_minus_credits",
    }


def resolved_month_from_df(df: pd.DataFrame, fallback: str) -> str:
    if df.attrs.get("resolved_month"):
        return df.attrs["resolved_month"]
    if "resolved_month" in df and len(df):
        return df["resolved_month"].iloc[0]
    return fallback


def get_spending_summary(month: str = "", db_path: str | None = None) -> dict[str, Any]:
    df = load_spending_rows(month, db_path)
    resolved_month = resolved_month_from_df(df, month)
    count = int(len(df))
    total = round(float(df["amount"].sum()), 2) if count else 0
    average = round(float(df["amount"].mean()), 2) if count else 0
    return {
        "month": resolved_month,
        "currency": BASE_ANALYTICS_CURRENCY,
        "total_spent": total,
        "transaction_count": count,
        "average_transaction": average,
    }


def get_category_breakdown(month: str = "", db_path: str | None = None) -> dict[str, Any]:
    df = load_spending_rows(month, db_path)
    resolved_month = resolved_month_from_df(df, month)
    categories = []

    if not df.empty:
        grouped = (
            df.groupby(["normalized_category", "normalized_category_en"])["amount"]
            .agg(["sum", "count"])
            .sort_values("sum", ascending=False)
            .reset_index()
        )
        categories = [
            {
                "category": row["normalized_category"],
                "category_en": row["normalized_category_en"],
                "amount": round(float(row["sum"]), 2),
                "transactions": int(row["count"]),
            }
            for _, row in grouped.iterrows()
        ]

    return {
        "month": resolved_month,
        "currency": BASE_ANALYTICS_CURRENCY,
        "categories": categories,
    }


def get_top_merchants(month: str = "", limit: int = 10, db_path: str | None = None) -> dict[str, Any]:
    df = load_spending_rows(month, db_path)
    resolved_month = resolved_month_from_df(df, month)
    merchants = []

    if not df.empty:
        grouped = (
            df.groupby("merchant")["amount"]
            .agg(["sum", "count", "mean"])
            .sort_values("sum", ascending=False)
            .head(int(limit or 10))
            .reset_index()
        )
        merchants = [
            {
                "merchant": row["merchant"],
                "amount": round(float(row["sum"]), 2),
                "transactions": int(row["count"]),
                "average_transaction": round(float(row["mean"]), 2),
            }
            for _, row in grouped.iterrows()
        ]

    return {
        "month": resolved_month,
        "currency": BASE_ANALYTICS_CURRENCY,
        "merchants": merchants,
    }


def get_largest_transactions(month: str = "", limit: int = 10, db_path: str | None = None) -> dict[str, Any]:
    df = load_spending_rows(month, db_path)
    resolved_month = resolved_month_from_df(df, month)
    transactions = []

    if not df.empty:
        largest = df.sort_values("amount", ascending=False).head(int(limit or 10))
        transactions = [
            {
                "transaction_date": row.get("transaction_date"),
                "merchant": row.get("merchant"),
                "description": row.get("description"),
                "category": row.get("normalized_category"),
                "category_en": row.get("normalized_category_en"),
                "amount": round(float(row.get("amount", 0) or 0), 2),
            }
            for _, row in largest.iterrows()
        ]

    return {
        "month": resolved_month,
        "currency": BASE_ANALYTICS_CURRENCY,
        "transactions": transactions,
    }


def stable_recurring_obligation_labels(
    months: list[str],
    db_path: str | None = None,
) -> set[str]:
    frames = []
    for month in months:
        df = load_spending_rows(month, db_path)
        if not df.empty:
            frames.append(df)

    if not frames:
        return set()

    spending_df = pd.concat(frames, ignore_index=True)
    labels = set()

    for merchant, merchant_df in spending_df.groupby("merchant"):
        months_present = merchant_df["transaction_month"].nunique()
        if months_present < 3:
            continue

        amounts = [float(value) for value in merchant_df["amount"].tolist()]
        stability = amount_stability(amounts)
        category_counts = (
            merchant_df.groupby(["normalized_category", "normalized_category_en"])
            .size()
            .sort_values(ascending=False)
        )
        category_key, category_en = category_counts.index[0]
        recurring_type = recurring_payment_type(merchant, category_key, category_en)

        if is_transaction_type_merchant_label(merchant):
            labels.add(normalize_text(merchant))
            continue

        if recurring_type and stability["is_stable"]:
            labels.add(normalize_text(merchant))

    return labels


def unusual_transactions_for_df(
    df: pd.DataFrame,
    stable_obligation_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    if df.empty:
        return []

    min_large_expense = 200
    merchant_thresholds = {}
    category_thresholds = {}
    stable_obligation_labels = stable_obligation_labels or set()
    df = df[
        ~df["merchant"].apply(
            lambda merchant: normalize_text(merchant) in stable_obligation_labels
        )
    ].copy()

    if df.empty:
        return []

    for merchant in df["merchant"].unique():
        merchant_df = df[df["merchant"] == merchant]
        if len(merchant_df) >= 3:
            merchant_thresholds[merchant] = {
                "threshold": round(float(merchant_df["amount"].quantile(0.95)), 2),
                "transactions_count": int(len(merchant_df)),
                "method": "merchant_95th_percentile",
            }

    for category in df["normalized_category"].unique():
        category_df = df[df["normalized_category"] == category]
        if len(category_df) >= 3:
            threshold = category_df["amount"].quantile(0.95)
            method = "normalized_category_95th_percentile"
        else:
            threshold = df["amount"].quantile(0.90)
            method = "global_90th_percentile_fallback"
        category_thresholds[category] = {
            "threshold": round(float(threshold), 2),
            "transactions_count": int(len(category_df)),
            "method": method,
        }

    def threshold_for(row: pd.Series) -> float:
        merchant = row["merchant"]
        category = row["normalized_category"]
        if merchant in merchant_thresholds:
            return merchant_thresholds[merchant]["threshold"]
        return category_thresholds[category]["threshold"]

    def method_for(row: pd.Series) -> str:
        merchant = row["merchant"]
        category = row["normalized_category"]
        if merchant in merchant_thresholds:
            return merchant_thresholds[merchant]["method"]
        return category_thresholds[category]["method"]

    df = df.copy()
    df["threshold"] = df.apply(threshold_for, axis=1)
    df["threshold_method"] = df.apply(method_for, axis=1)
    unusual = df[(df["amount"] > df["threshold"]) & (df["amount"] >= min_large_expense)]
    unusual = unusual.sort_values("amount", ascending=False)

    return unusual[
        [
            "transaction_date",
            "merchant",
            "normalized_category",
            "normalized_category_en",
            "amount",
            "threshold",
            "threshold_method",
        ]
    ].to_dict(orient="records")


def get_unusual_transactions(month: str = "", db_path: str | None = None) -> dict[str, Any]:
    month_text = normalize_text(month).casefold()
    with db_connect(db_path) as conn:
        range_months = resolve_month_range(month_text, conn)

    if month_text in {"all", "all months", "for all months", "all period", "entire period"} or range_months:
        if range_months:
            selected_months = range_months
            result_month = month_text
        else:
            with db_connect(db_path) as conn:
                selected_months = recent_months(conn, 60)
            result_month = "all"

        stable_labels = stable_recurring_obligation_labels(selected_months, db_path)

        grouped = []
        all_transactions = []
        for selected_month in selected_months:
            df = load_spending_rows(selected_month, db_path)
            transactions = unusual_transactions_for_df(df, stable_labels)
            grouped.append(
                {
                    "month": selected_month,
                    "large_expenses_to_review": transactions,
                }
            )
            for item in transactions:
                all_transactions.append({"month": selected_month, **item})

        return {
            "month": result_month,
            "months": selected_months,
            "currency": BASE_ANALYTICS_CURRENCY,
            "method": "merchant threshold first, normalized-category fallback",
            "excluded_stable_recurring_obligations": sorted(stable_labels),
            "unusual_by_month": grouped,
            "large_expenses_to_review": all_transactions,
        }

    df = load_spending_rows(month, db_path)
    resolved_month = resolved_month_from_df(df, month)
    with db_connect(db_path) as conn:
        context_months = recent_months(conn, 6)
    stable_labels = stable_recurring_obligation_labels(context_months, db_path)

    return {
        "month": resolved_month,
        "currency": BASE_ANALYTICS_CURRENCY,
        "method": "merchant threshold first, normalized-category fallback",
        "excluded_stable_recurring_obligations": sorted(stable_labels),
        "large_expenses_to_review": unusual_transactions_for_df(df, stable_labels),
    }


def compare_months(month_a: str, month_b: str, db_path: str | None = None) -> dict[str, Any]:
    summary_a = get_spending_summary(month_a, db_path)
    summary_b = get_spending_summary(month_b, db_path)
    amount_a = float(summary_a["total_spent"])
    amount_b = float(summary_b["total_spent"])
    difference = round(amount_b - amount_a, 2)
    percentage_change = round((difference / amount_a * 100), 2) if amount_a else None

    cats_a = {
        item["category"]: item
        for item in get_category_breakdown(summary_a["month"], db_path)["categories"]
    }
    cats_b = {
        item["category"]: item
        for item in get_category_breakdown(summary_b["month"], db_path)["categories"]
    }

    changes = []
    for category in sorted(set(cats_a) | set(cats_b)):
        before = float(cats_a.get(category, {}).get("amount", 0))
        after = float(cats_b.get(category, {}).get("amount", 0))
        changes.append(
            {
                "category": category,
                "category_en": category_label(category),
                "month_a_amount": round(before, 2),
                "month_b_amount": round(after, 2),
                "difference": round(after - before, 2),
            }
        )
    changes.sort(key=lambda item: abs(item["difference"]), reverse=True)

    return {
        "month_a": summary_a["month"],
        "month_b": summary_b["month"],
        "currency": BASE_ANALYTICS_CURRENCY,
        "month_a_coverage": month_coverage_metadata(summary_a["month"]),
        "month_b_coverage": month_coverage_metadata(summary_b["month"]),
        "month_a_total": amount_a,
        "month_b_total": amount_b,
        "spending_difference": difference,
        "percentage_change": percentage_change,
        "largest_category_changes": changes[:10],
    }


def get_category_comparison(
    category: str,
    month_a: str,
    month_b: str,
    db_path: str | None = None,
) -> dict[str, Any]:
    category_key = normalize_category_query(category)
    month_a_breakdown = get_category_breakdown(month_a, db_path)
    month_b_breakdown = get_category_breakdown(month_b, db_path)

    def category_entry(breakdown: dict[str, Any]) -> dict[str, Any]:
        return next(
            (
                item
                for item in breakdown["categories"]
                if item["category"] == category_key
            ),
            {
                "category": category_key,
                "category_en": category_label(category_key),
                "amount": 0,
                "transactions": 0,
            },
        )

    first = category_entry(month_a_breakdown)
    second = category_entry(month_b_breakdown)
    first_amount = float(first.get("amount", 0) or 0)
    second_amount = float(second.get("amount", 0) or 0)

    return {
        "category": category_key,
        "category_en": category_label(category_key),
        "currency": BASE_ANALYTICS_CURRENCY,
        "months": [
            {
                "month": month_a_breakdown["month"],
                "amount": round(first_amount, 2),
                "transactions": int(first.get("transactions", 0) or 0),
            },
            {
                "month": month_b_breakdown["month"],
                "amount": round(second_amount, 2),
                "transactions": int(second.get("transactions", 0) or 0),
            },
        ],
        "difference": round(second_amount - first_amount, 2),
    }


def recent_months(conn: sqlite3.Connection, months: int) -> list[str]:
    query = f"""
        SELECT DISTINCT transaction_month
        FROM transactions
        WHERE {ACTIVE_ANALYTICS_WHERE}
          AND direction = 'debit'
          AND amount_abs > 0
          AND currency = ?
          AND transaction_type IN ({spending_type_placeholders()})
          AND transaction_month IS NOT NULL
        ORDER BY transaction_month DESC
        LIMIT ?
    """
    rows = conn.execute(
        query,
        (BASE_ANALYTICS_CURRENCY, *tuple(SPENDING_TRANSACTION_TYPES), int(months)),
    ).fetchall()
    return sorted(row[0] for row in rows)


def get_category_trend(category: str, months: int = 6, db_path: str | None = None) -> dict[str, Any]:
    category_key = normalize_category_query(category)
    with db_connect(db_path) as conn:
        selected_months = recent_months(conn, months)

    monthly_totals = []
    for month in selected_months:
        breakdown = get_category_breakdown(month, db_path)
        match = next(
            (item for item in breakdown["categories"] if item["category"] == category_key),
            None,
        )
        monthly_totals.append(
            {
                "month": month,
                "amount": match["amount"] if match else 0,
                "transactions": match["transactions"] if match else 0,
            }
        )

    return {
        "category": category_key,
        "category_en": category_label(category_key),
        "currency": BASE_ANALYTICS_CURRENCY,
        "monthly_totals": monthly_totals,
    }


def get_recurring_merchants(months: int = 6, db_path: str | None = None) -> dict[str, Any]:
    with db_connect(db_path) as conn:
        selected_months = recent_months(conn, months)

    records = []
    for month in selected_months:
        df = load_spending_rows(month, db_path)
        if df.empty:
            continue
        for _, row in df.iterrows():
            merchant = row.get("merchant")
            if is_transaction_type_merchant_label(merchant):
                continue
            records.append(
                {
                    "month": month,
                    "merchant": merchant,
                    "amount": float(row.get("amount", 0) or 0),
                }
            )

    if not records:
        return {
            "months": selected_months,
            "currency": BASE_ANALYTICS_CURRENCY,
            "merchants_every_month": [],
            "merchants": [],
        }

    df = pd.DataFrame(records)
    grouped = (
        df.groupby("merchant")
        .agg(
            month=("month", lambda values: sorted(set(values))),
            total_amount=("amount", "sum"),
            transactions=("amount", "count"),
            average_amount=("amount", "mean"),
        )
        .reset_index()
    )
    merchants = [
        {
            "merchant": row["merchant"],
            "months_present": len(row["month"]),
            "months": row["month"],
            "total_amount": round(float(row["total_amount"]), 2),
            "transactions": int(row["transactions"]),
            "average_amount": round(float(row["average_amount"]), 2),
        }
        for _, row in grouped.iterrows()
        if len(row["month"]) >= 2
    ]
    merchants.sort(key=lambda item: (-item["months_present"], item["merchant"]))
    merchants_every_month = [
        item
        for item in merchants
        if item["months_present"] == len(selected_months)
    ]

    return {
        "months": selected_months,
        "currency": BASE_ANALYTICS_CURRENCY,
        "merchants_every_month": merchants_every_month,
        "merchants": merchants,
    }


def is_transaction_type_merchant_label(merchant: object) -> bool:
    return normalize_text(merchant) in TRANSACTION_TYPE_MERCHANT_LABELS


def amount_stability(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "min_amount": 0,
            "max_amount": 0,
            "average_amount": 0,
            "relative_variation_percent": 0,
            "is_stable": False,
        }

    minimum = min(values)
    maximum = max(values)
    average = sum(values) / len(values)
    relative_variation = ((maximum - minimum) / average * 100) if average else 0

    return {
        "min_amount": round(float(minimum), 2),
        "max_amount": round(float(maximum), 2),
        "average_amount": round(float(average), 2),
        "relative_variation_percent": round(float(relative_variation), 2),
        "is_stable": relative_variation <= 5,
    }


def build_transaction_type_labels(spending_df: pd.DataFrame) -> list[dict[str, Any]]:
    if spending_df.empty:
        return []

    labels = []
    labeled_df = spending_df[
        spending_df["merchant"].apply(is_transaction_type_merchant_label)
    ]
    if labeled_df.empty:
        return labels

    for label, group in labeled_df.groupby("merchant"):
        amounts = [float(value) for value in group["amount"].tolist()]
        category_counts = (
            group.groupby(["normalized_category", "normalized_category_en"])
            .size()
            .sort_values(ascending=False)
        )
        category_key, category_en = category_counts.index[0]
        dominant_category_transactions = int(category_counts.iloc[0])
        dominant_category_share = dominant_category_transactions / int(len(group))
        months = sorted(set(group["transaction_month"].dropna().tolist()))
        labels.append(
            {
                "label": label,
                "merchant_is_transaction_type": True,
                "months_present": len(months),
                "months": months,
                "transactions": int(len(group)),
                "total_amount": round(float(group["amount"].sum()), 2),
                "category": category_key,
                "category_en": category_en,
                "dominant_category_transactions": dominant_category_transactions,
                "dominant_category_share": round(float(dominant_category_share), 3),
                "usage_hint": (
                    transaction_type_usage_hint(category_key, category_en)
                    if dominant_category_share >= 0.8
                    else ""
                ),
                "amount_stability": amount_stability(amounts),
                "description_note": (
                    "Original descriptions are unchanged; counterparty extraction is not performed."
                ),
            }
        )

    labels.sort(
        key=lambda item: (
            -int(item["months_present"]),
            -float(item["total_amount"]),
            item["label"],
        )
    )
    return labels


def recurring_payment_type(merchant: object, category: object, category_en: object = "") -> str:
    category_key = normalize_text(category)
    text = " ".join(
        [
            normalize_text(merchant),
            normalize_text(category),
            normalize_text(category_en),
        ]
    ).casefold()

    for payment_type, keywords in RECURRING_PAYMENT_KEYWORDS.items():
        if any(keyword.casefold() in text for keyword in keywords):
            return payment_type

    if category_key in RECURRING_PAYMENT_CATEGORIES:
        return category_key

    return ""


def build_recurring_payments(
    recurring_merchants: list[dict[str, Any]],
    spending_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    if spending_df.empty or not recurring_merchants:
        return []

    merchant_months = {
        item["merchant"]: item
        for item in recurring_merchants
        if not is_transaction_type_merchant_label(item.get("merchant"))
    }
    payments = []

    for merchant, item in merchant_months.items():
        merchant_df = spending_df[spending_df["merchant"] == merchant]
        if merchant_df.empty:
            continue

        category_counts = (
            merchant_df.groupby(["normalized_category", "normalized_category_en"])
            .size()
            .sort_values(ascending=False)
        )
        category_key, category_en = category_counts.index[0]
        payment_type = recurring_payment_type(merchant, category_key, category_en)

        if not payment_type:
            continue

        amounts = [float(value) for value in merchant_df["amount"].tolist()]
        stability = amount_stability(amounts)

        payments.append(
            {
                "merchant": merchant,
                "months_present": item.get("months_present", 0),
                "months": item.get("months", []),
                "payment_type": payment_type,
                "category": category_key,
                "category_en": category_en,
                "total_amount": round(float(merchant_df["amount"].sum()), 2),
                "transactions": int(len(merchant_df)),
                "amount_stability": stability,
                "is_stable_recurring_obligation": bool(
                    item.get("months_present", 0) >= 3 and stability["is_stable"]
                ),
                "evidence": (
                    "Appears in multiple months and category or merchant text "
                    "matches deterministic recurring-payment rules."
                ),
            }
        )

    payments.sort(
        key=lambda payment: (
            -int(payment["months_present"]),
            -float(payment["total_amount"]),
            payment["merchant"],
        )
    )
    return payments


def spending_area_type(category: object, merchant: object = "") -> str:
    category_key = normalize_text(category)
    merchant_text = normalize_text(merchant).casefold()

    if any(keyword.casefold() in merchant_text for keyword in FOOD_DELIVERY_KEYWORDS):
        return "food_delivery"
    if any(keyword.casefold() in merchant_text for keyword in SUBSCRIPTION_KEYWORDS):
        return "subscriptions"
    if category_key == "restaurants_cafes_bars":
        return "restaurants"
    if category_key == "leisure_entertainment_sports":
        return "leisure_entertainment_sports"
    if category_key == "home_and_furniture":
        return "shopping"
    if category_key == "flights_and_travel":
        return "flights_and_travel"
    if category_key in MANDATORY_SPENDING_CATEGORIES:
        return "mandatory_fixed"
    if category_key in DISCRETIONARY_SPENDING_CATEGORIES:
        return category_key

    return "other"


def spending_review_classification(category: object, merchant: object = "") -> str:
    area_type = spending_area_type(category, merchant)
    if area_type in {
        "food_delivery",
        "restaurants",
        "leisure_entertainment_sports",
        "shopping",
        "subscriptions",
        "flights_and_travel",
    }:
        return "reviewable_discretionary"
    if area_type == "mandatory_fixed":
        return "mandatory_fixed"
    return "neutral"


def build_spending_area_summaries(spending_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if spending_df.empty:
        return [], []

    rows = []
    for _, row in spending_df.iterrows():
        merchant = row.get("merchant")
        if is_transaction_type_merchant_label(merchant):
            continue
        area_type = spending_area_type(row.get("normalized_category"), merchant)
        classification = spending_review_classification(row.get("normalized_category"), merchant)
        rows.append(
            {
                "type": area_type,
                "classification": classification,
                "merchant": normalize_text(merchant) or "Unknown",
                "amount": float(row.get("review_amount", row.get("amount", 0)) or 0),
                "transactions": 1,
                "debit_transactions": int(normalize_text(row.get("direction")).casefold() == "debit"),
                "credit_transactions": int(normalize_text(row.get("direction")).casefold() == "credit"),
            }
        )

    if not rows:
        return [], []

    area_df = pd.DataFrame(rows)
    grouped = (
        area_df.groupby(["type", "classification"])
        .agg(
            total_amount=("amount", "sum"),
            transactions=("transactions", "sum"),
            debit_transactions=("debit_transactions", "sum"),
            credit_transactions=("credit_transactions", "sum"),
        )
        .reset_index()
    )

    reviewable = []
    mandatory = []
    for _, row in grouped.iterrows():
        area_rows = area_df[
            (area_df["type"] == row["type"])
            & (area_df["classification"] == row["classification"])
        ]
        merchant_grouped = (
            area_rows.groupby("merchant")
            .agg(
                amount=("amount", "sum"),
                transactions=("transactions", "sum"),
                debit_transactions=("debit_transactions", "sum"),
                credit_transactions=("credit_transactions", "sum"),
            )
            .reset_index()
        )
        merchant_grouped = merchant_grouped[merchant_grouped["amount"] > 0]
        merchant_grouped = merchant_grouped.sort_values("amount", ascending=False).head(5)
        item = {
            "type": row["type"],
            "amount": round(float(row["total_amount"]), 2),
            "transactions": int(row["transactions"]),
            "debit_transactions": int(row["debit_transactions"]),
            "credit_transactions": int(row["credit_transactions"]),
            "calculation_basis": "net_debits_minus_credits",
            "top_merchants": [
                {
                    "merchant": merchant_row["merchant"],
                    "amount": round(float(merchant_row["amount"]), 2),
                    "transactions": int(merchant_row["transactions"]),
                    "debit_transactions": int(merchant_row["debit_transactions"]),
                    "credit_transactions": int(merchant_row["credit_transactions"]),
                }
                for _, merchant_row in merchant_grouped.iterrows()
            ],
            "reason": (
                "Discretionary or non-essential spending category"
                if row["classification"] == "reviewable_discretionary"
                else "Mandatory or fixed spending category"
            ),
        }
        if row["classification"] == "reviewable_discretionary":
            reviewable.append(item)
        elif row["classification"] == "mandatory_fixed":
            mandatory.append(item)

    reviewable.sort(key=lambda item: -float(item["amount"]))
    mandatory.sort(key=lambda item: -float(item["amount"]))
    return reviewable, mandatory


def prepare_financial_review_context(months: int = 6, db_path: str | None = None) -> dict[str, Any]:
    with db_connect(db_path) as conn:
        selected_months = recent_months(conn, months)

    monthly_spending_totals = [
        get_review_spending_summary(month, db_path)
        for month in selected_months
    ]

    monthly_frames = []
    for month in selected_months:
        df = load_review_spending_rows(month, db_path)
        if not df.empty:
            monthly_frames.append(df)

    spending_df = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    total_spent = round(
        float(sum(item.get("total_spent", 0) or 0 for item in monthly_spending_totals)),
        2,
    )

    top_categories = []
    category_concentration = {
        "top_category": None,
        "top_category_key": None,
        "top_category_amount": 0,
        "top_category_percent": 0,
    }

    if not spending_df.empty:
        category_grouped = (
            spending_df.groupby(["normalized_category", "normalized_category_en"])
            .agg(
                total_amount=("review_amount", "sum"),
                transactions=("review_amount", "count"),
                debit_transactions=("direction", lambda values: int((values.fillna("").str.casefold() == "debit").sum())),
                credit_transactions=("direction", lambda values: int((values.fillna("").str.casefold() == "credit").sum())),
                months_present=("transaction_month", "nunique"),
            )
            .sort_values("total_amount", ascending=False)
            .reset_index()
        )
        category_grouped = category_grouped[category_grouped["total_amount"] > 0]
        top_categories = [
            {
                "category": row["normalized_category"],
                "category_en": row["normalized_category_en"],
                "total_amount": round(float(row["total_amount"]), 2),
                "transactions": int(row["transactions"]),
                "debit_transactions": int(row["debit_transactions"]),
                "credit_transactions": int(row["credit_transactions"]),
                "months_present": int(row["months_present"]),
                "percent_of_total": round(float(row["total_amount"]) / total_spent * 100, 2)
                if total_spent
                else 0,
                "spending_classification": spending_review_classification(row["normalized_category"]),
                "calculation_basis": "net_debits_minus_credits",
            }
            for _, row in category_grouped.head(10).iterrows()
        ]

        if top_categories:
            top_category = top_categories[0]
            category_concentration = {
                "top_category": top_category["category_en"],
                "top_category_key": top_category["category"],
                "top_category_amount": top_category["total_amount"],
                "top_category_percent": top_category["percent_of_total"],
            }

    recurring = get_recurring_merchants(months, db_path)
    transaction_type_labels = build_transaction_type_labels(spending_df)
    recurring_merchants = [
        item
        for item in recurring.get("merchants", [])
        if not is_transaction_type_merchant_label(item.get("merchant"))
    ]
    recurring_payments = build_recurring_payments(recurring_merchants, spending_df)
    stable_recurring_obligations = [
        {
            **item,
            "classification": "stable_recurring_obligation",
        }
        for item in recurring_payments
        if item.get("is_stable_recurring_obligation")
    ]
    stable_transaction_type_obligations = [
        {
            "label": item.get("label"),
            "months_present": item.get("months_present"),
            "months": item.get("months"),
            "transactions": item.get("transactions"),
            "total_amount": item.get("total_amount"),
            "category": item.get("category"),
            "category_en": item.get("category_en"),
            "dominant_category_transactions": item.get("dominant_category_transactions"),
            "dominant_category_share": item.get("dominant_category_share"),
            "usage_hint": item.get("usage_hint"),
            "amount_stability": item.get("amount_stability"),
            "classification": "stable_recurring_obligation",
            "source": "transaction_type_label",
            "description_note": item.get("description_note"),
        }
        for item in transaction_type_labels
        if item.get("months_present", 0) >= 3 and item.get("amount_stability", {}).get("is_stable")
    ]
    stable_recurring_obligations = stable_recurring_obligations + stable_transaction_type_obligations

    stable_obligation_merchants = {
        item["merchant"]
        for item in stable_recurring_obligations
        if item.get("merchant")
    }
    stable_transaction_type_labels = {
        item["label"]
        for item in stable_transaction_type_obligations
        if item.get("label")
    }
    stable_labels = stable_obligation_merchants | stable_transaction_type_labels

    unusual_transactions = []
    largest_expenses = []

    for month in selected_months:
        unusual = get_unusual_transactions(month, db_path)
        for item in unusual.get("large_expenses_to_review", []):
            unusual_transactions.append(
                {
                    "month": month,
                    "transaction_date": item.get("transaction_date"),
                    "merchant": item.get("merchant"),
                    "amount": item.get("amount"),
                    "category": item.get("normalized_category"),
                    "category_en": item.get("normalized_category_en"),
                    "threshold": item.get("threshold"),
                    "threshold_method": item.get("threshold_method"),
                    "review_classification": (
                        "stable_recurring_obligation_not_suspicious"
                        if item.get("merchant") in stable_labels
                        else "unusual_transaction"
                    ),
                }
            )

        largest = get_largest_transactions(month, limit=3, db_path=db_path)
        for item in largest.get("transactions", []):
            largest_expenses.append({"month": month, **item})

    unusual_transactions.sort(key=lambda item: float(item.get("amount", 0) or 0), reverse=True)
    largest_expenses.sort(key=lambda item: float(item.get("amount", 0) or 0), reverse=True)

    reviewable_spending_areas, mandatory_spending_areas = build_spending_area_summaries(spending_df)
    possible_review_areas = []

    for item in reviewable_spending_areas[:8]:
        possible_review_areas.append(
            {
                "type": item.get("type"),
                "amount": item.get("amount"),
                "transactions": item.get("transactions"),
                "evidence": item.get("reason"),
            }
        )

    for item in unusual_transactions[:5]:
        if item.get("review_classification") == "stable_recurring_obligation_not_suspicious":
            continue
        if spending_review_classification(item.get("category"), item.get("merchant")) == "mandatory_fixed":
            continue
        possible_review_areas.append(
            {
                "type": "unusual_transaction",
                "label": item.get("merchant"),
                "month": item.get("month"),
                "amount": item.get("amount"),
                "category_en": item.get("category_en"),
                "evidence": "Returned by deterministic unusual-transaction detection.",
            }
        )

    for item in largest_expenses[:5]:
        if spending_review_classification(item.get("category"), item.get("merchant")) == "mandatory_fixed":
            continue
        if item.get("merchant") in stable_labels:
            continue
        possible_review_areas.append(
            {
                "type": "large_expense",
                "label": item.get("merchant"),
                "month": item.get("month"),
                "amount": item.get("amount"),
                "category_en": item.get("category_en"),
                "evidence": "One of the largest expenses in the covered period.",
            }
        )

    return {
        "currency": BASE_ANALYTICS_CURRENCY,
        "currency_symbol": CURRENCY_SYMBOLS.get(BASE_ANALYTICS_CURRENCY, BASE_ANALYTICS_CURRENCY),
        "amount_calculation_basis": "net_debits_minus_credits_for_financial_review",
        "months_covered": selected_months,
        "month_coverage": [
            month_coverage_metadata(month)
            for month in selected_months
        ],
        "total_spent": total_spent,
        "monthly_spending_totals": monthly_spending_totals,
        "top_categories": top_categories,
        "category_concentration": category_concentration,
        "reviewable_spending_areas": reviewable_spending_areas[:25],
        "mandatory_spending_areas": mandatory_spending_areas[:25],
        "transaction_type_labels": transaction_type_labels[:25],
        "recurring_merchants": recurring_merchants[:25],
        "recurring_payments": recurring_payments[:25],
        "stable_recurring_obligations": stable_recurring_obligations[:25],
        "unusual_transactions": unusual_transactions[:25],
        "largest_expenses": largest_expenses[:25],
        "possible_review_areas": possible_review_areas[:25],
        "data_quality_log": {
            "transaction_type_labels_count": len(transaction_type_labels),
            "recurring_merchants_count": len(recurring_merchants),
            "stable_recurring_obligations_count": len(stable_recurring_obligations),
        },
    }


def search_transactions(keyword: str, limit: int = 50, db_path: str | None = None) -> dict[str, Any]:
    text = normalize_text(keyword)
    if not text:
        return {"keyword": keyword, "matches": []}

    like = f"%{text}%"
    query = f"""
        SELECT
            transaction_date,
            posting_date,
            transaction_month,
            cashflow_month,
            merchant,
            description,
            category,
            category_en,
            amount AS signed_amount,
            amount_abs,
            currency,
            direction,
            source_type,
            transaction_type,
            card_last4,
            account_last4,
            account_id,
            counterparty,
            bank_reference,
            analytics_excluded
        FROM transactions
        WHERE (merchant LIKE ? OR description LIKE ? OR category LIKE ? OR category_en LIKE ? OR counterparty LIKE ?)
        ORDER BY transaction_date DESC, id DESC
        LIMIT ?
    """
    with db_connect(db_path) as conn:
        rows = conn.execute(query, (like, like, like, like, like, int(limit or 50))).fetchall()

    return {
        "keyword": text,
        "matches": [dict(row) for row in rows],
    }
