from mcp.server.fastmcp import FastMCP
import pandas as pd
import json
import os
import hashlib
import sqlite3
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.runtime_config import load_runtime_config
from finance import query_tools
from finance.display import format_merchant_display

load_runtime_config()

# This file is the deterministic finance layer of the app.
# The LLM can choose these MCP tools, but the calculations themselves happen here.
mcp = FastMCP("finance-mcp")

# Parsed CSV data is reused across tools during one app run.
# This keeps iterative development faster without changing financial results.
FINANCE_CACHE_ENABLED = os.getenv("FINANCE_CACHE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FINANCE_CACHE_MAX_ENTRIES = int(os.getenv("FINANCE_CACHE_MAX_ENTRIES", "16"))

_RAW_STATEMENT_CACHE: dict[tuple[str, int, int], pd.DataFrame] = {}
_NORMALIZED_STATEMENT_CACHE: dict[tuple[str, int, int], pd.DataFrame] = {}

TRANSACTION_COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "source_file": "TEXT NOT NULL",
    "source_type": "TEXT NOT NULL",
    "transaction_type": "TEXT NOT NULL",
    "transaction_date": "TEXT",
    "posting_date": "TEXT",
    "charge_date": "TEXT",
    "transaction_month": "TEXT",
    "cashflow_month": "TEXT",
    "statement_month": "TEXT",
    "merchant": "TEXT",
    "description": "TEXT",
    "category": "TEXT",
    "category_en": "TEXT",
    "amount": "REAL NOT NULL",
    "amount_abs": "REAL NOT NULL",
    "currency": "TEXT DEFAULT 'NIS'",
    "direction": "TEXT NOT NULL",
    "card_last4": "TEXT",
    "account_last4": "TEXT",
    "account_id": "TEXT",
    "counterparty": "TEXT",
    "bank_reference": "TEXT",
    "original_amount": "REAL",
    "original_currency": "TEXT",
    "raw_row_json": "TEXT",
    "normalized_key": "TEXT",
    "duplicate_group_id": "TEXT",
    "duplicate_status": "TEXT DEFAULT 'unique'",
    "duplicate_of_transaction_id": "INTEGER",
    "duplicate_reason": "TEXT",
    "analytics_excluded": "INTEGER DEFAULT 0",
    "analytics_exclusion_reason": "TEXT",
    "created_at": "TEXT DEFAULT CURRENT_TIMESTAMP",
}

TRANSACTION_INDEXES = {
    "idx_transactions_statement_month": "statement_month",
    "idx_transactions_transaction_month": "transaction_month",
    "idx_transactions_cashflow_month": "cashflow_month",
    "idx_transactions_card_last4": "card_last4",
    "idx_transactions_transaction_date": "transaction_date",
    "idx_transactions_source_type": "source_type",
    "idx_transactions_transaction_type": "transaction_type",
    "idx_transactions_analytics_excluded": "analytics_excluded",
    "idx_transactions_duplicate_status": "duplicate_status",
}

HEBREW_CATEGORY_LABELS = {
    "מזון וצריכה": "Food and groceries",
    "דלק, חשמל וגז": "Fuel, electricity and gas",
    "רפואה ובתי מרקחת": "Health and pharmacies",
    "מסעדות, קפה וברים": "Restaurants, cafes and bars",
    "תחבורה ורכבים": "Transport and vehicles",
    "עיצוב הבית": "Home and furniture",
    "עירייה וממשלה": "Government and municipality",
    "שונות": "Other",
    "טיסות ותיירות": "Flights and travel",
    "פנאי, בידור וספורט": "Leisure, entertainment and sports",
    "שירותי תקשורת": "Communication services",
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

XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

DEBIT_CARD_LAST4 = {"1043", "0946"}
CREDIT_CARD_LAST4 = {"9542", "7544"}
ACTIVE_ANALYTICS_WHERE = "COALESCE(analytics_excluded, 0) = 0"
SPENDING_TRANSACTION_TYPES = {"real_expense", "standing_order", "housing_utility_expense", "unknown"}
BASE_ANALYTICS_CURRENCY = "NIS"


def default_db_path() -> str:
    return os.getenv("DB_PATH", "./database/finance.db")


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_date(value: object) -> str | None:
    text = normalize_text(value)

    if not text:
        return None

    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass

    return text


def statement_month_from(*dates: str | None) -> str | None:
    for value in dates:
        normalized = normalize_date(value)
        if normalized and len(normalized) >= 7:
            return normalized[:7]
    return None


def month_from_date(value: object) -> str | None:
    # Month fields are stored once during import so analytics can group rows
    # without re-deciding which date is relevant for each source type.
    return statement_month_from(normalize_date(value))


def derive_transaction_month(transaction: dict) -> str | None:
    # transaction_month answers "when did the purchase/activity happen?"
    # For bank rows we prefer a purchase date extracted from the description;
    # if there is none, the posting date is the best deterministic fallback.
    source_type = normalize_text(transaction.get("source_type"))

    if source_type == "bank_account":
        description_date = extract_transaction_date_from_description(
            transaction.get("description")
        )
        return month_from_date(description_date) or month_from_date(transaction.get("posting_date"))

    return month_from_date(transaction.get("transaction_date")) or month_from_date(
        transaction.get("posting_date")
    ) or month_from_date(transaction.get("charge_date"))


def derive_cashflow_month(transaction: dict) -> str | None:
    # cashflow_month answers "when did the money move in the bank/cashflow view?"
    # Credit-card purchases affect cash flow on the charge date, while bank rows
    # and debit-card rows affect cash flow when posted to the account.
    source_type = normalize_text(transaction.get("source_type"))

    if source_type == "credit_card":
        return month_from_date(transaction.get("charge_date")) or month_from_date(
            transaction.get("posting_date")
        ) or month_from_date(transaction.get("transaction_date"))

    if source_type in {"bank_account", "debit_card"}:
        return month_from_date(transaction.get("posting_date")) or month_from_date(
            transaction.get("charge_date")
        ) or month_from_date(transaction.get("transaction_date"))

    return month_from_date(transaction.get("posting_date")) or month_from_date(
        transaction.get("charge_date")
    ) or month_from_date(transaction.get("transaction_date"))


def apply_transaction_months(transaction: dict) -> dict:
    # statement_month is kept as a compatibility alias for existing reports.
    # New analytics should choose transaction_month for spending analysis or
    # cashflow_month for bank cash-flow analysis.
    item = dict(transaction)
    item["transaction_month"] = item.get("transaction_month") or derive_transaction_month(item)
    item["cashflow_month"] = item.get("cashflow_month") or derive_cashflow_month(item)
    item["statement_month"] = item.get("statement_month") or item.get("transaction_month")
    return item


def apply_transaction_category_overrides(transaction: dict) -> dict:
    # Store explicit categories for known housing/utility patterns so future
    # analytics do not depend only on runtime normalization.
    item = dict(transaction)
    if is_rent_or_utility_payment(
        item.get("category"),
        item.get("category_en"),
        item.get("merchant"),
        item.get("description"),
    ):
        item["category"] = "ארנונה, שכירות וחשבונות"
        item["category_en"] = CATEGORY_LABELS["rent_and_utilities"]
        if item.get("direction") == "debit" and item.get("transaction_type") in {
            "bank_transfer",
            "standing_order",
            "unknown",
        }:
            item["transaction_type"] = "housing_utility_expense"

    if is_education_or_course_payment(
        item.get("category"),
        item.get("category_en"),
        item.get("merchant"),
        item.get("description"),
    ):
        item["category"] = "לימודים וקורסים"
        item["category_en"] = CATEGORY_LABELS["education_and_courses"]

    item["counterparty"] = item.get("counterparty") or extract_counterparty(
        item.get("description"),
        item.get("raw_row_json"),
    )
    return item


def parse_amount(value: object) -> float:
    text = normalize_text(value)

    if not text:
        return 0.0

    text = (
        text.replace("₪", "")
        .replace(" ", "")
        .replace("\u200e", "")
        .replace("\u200f", "")
    )

    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    return float(text)


def normalize_currency(value: object) -> str:
    text = normalize_text(value)

    if text in {"₪", "ILS", "NIS", "שח", 'ש"ח'}:
        return "NIS"

    return text or "NIS"


def normalize_card_last4(value: object) -> str:
    text = normalize_text(value)

    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    return text


def detect_card_source_type(card_last4: object) -> str:
    card = normalize_card_last4(card_last4)

    if card in DEBIT_CARD_LAST4:
        return "debit_card"

    if card in CREDIT_CARD_LAST4:
        return "credit_card"

    return "unknown"


def extract_card_last4(text: object) -> str | None:
    normalized = normalize_text(text)
    patterns = [
        r"מסתיים ב-?(\d{4})",
        r"ending(?: in)?\s*(\d{4})",
        r"card\s*(?:no\.?)?\s*(\d{4})",
        r"כרטיס.*?(\d{4})",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def extract_transaction_date_from_description(text: object) -> str | None:
    normalized = normalize_text(text)
    match = re.search(r"מתאריך\s+(\d{1,2}/\d{1,2}/\d{2,4})", normalized)

    if match:
        value = match.group(1)
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                pass

    return None


def extract_merchant_from_bank_description(description: object, expanded: object) -> str | None:
    text = normalize_text(expanded) or normalize_text(description)

    if " ב-" in text:
        return text.rsplit(" ב-", 1)[-1].strip()

    return normalize_text(description) or None


def is_rent_or_utility_payment(*values: object) -> bool:
    # Housing costs arrive through several channels: manual bank transfers for
    # rent, standing-order building payments, and municipality/card payments for
    # Arnona. Keep this deterministic so analytics can include them as expenses.
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
    # Some card exports contain misleading bank categories. Merchant rules keep
    # known education/subscription services out of government/municipality totals.
    text = " ".join(normalize_text(value) for value in values).casefold()
    keywords = [
        "riseguide",
        "english",
        "course",
        "לימוד",
        "למידה",
        "אנגלית",
    ]
    return any(keyword.casefold() in text for keyword in keywords)


def extract_counterparty(*values: object) -> str | None:
    # Card statements can include BIT transfer details in the notes field, e.g.
    # "למי: <name>". Bank-account mirror rows usually do not contain that name.
    text = " ".join(normalize_text(value) for value in values)
    patterns = [
        r"למי:\s*([^;|,]+)",
        r"ממי:\s*([^;|,]+)",
        r"to:\s*([^;|,]+)",
        r"from:\s*([^;|,]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(1)).strip(" \"'")

    return None


def normalize_category_key(
    category: object,
    category_en: object,
    merchant: object,
    description: object = "",
) -> str:
    # Transaction-table analytics need the same category keys as the old CSV flow
    # so the UI, reports, and AI insight context can keep their existing contract.
    category_text = normalize_text(category)
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

    category_en_text = normalize_text(category_en).casefold()
    if category_en_text in ENGLISH_CATEGORY_KEYS:
        return ENGLISH_CATEGORY_KEYS[category_en_text]

    return "other"


def classify_bank_transaction(description: object, expanded: object, direction: str) -> str:
    text = f"{normalize_text(description)} {normalize_text(expanded)}".casefold()

    if direction == "debit" and is_rent_or_utility_payment(description, expanded):
        return "housing_utility_expense"

    if any(keyword.casefold() in text for keyword in ["כרטיס דביט", "דביט", "debit card"]):
        return "real_expense"

    if any(keyword.casefold() in text for keyword in ["national visa", "כרטיס אשראי", "ויזה", "visa", "max", "מקס", "לאומי קארד"]):
        return "card_settlement"

    if any(keyword.casefold() in text for keyword in ["הוראת קבע", "standing order"]):
        return "standing_order"

    if any(keyword.casefold() in text for keyword in ["salary", "משכורת"]):
        return "salary_income"

    if any(keyword.casefold() in text for keyword in ["ביטוח לאומי", "national insurance"]):
        return "government_income"

    if any(keyword.casefold() in text for keyword in ["העברה", "transfer", "הע. אינטרנט"]):
        return "bank_transfer" if direction == "debit" else "internal_transfer"

    return "unknown"


def direction_from_amount(amount: float) -> str:
    if amount > 0:
        return "debit"
    if amount < 0:
        return "credit"
    return "unknown"


class SimpleHtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self.current_table: list[list[str]] | None = None
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None
        self.in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()

        if tag == "table":
            self.current_table = []
        elif tag == "tr" and self.current_table is not None:
            self.current_row = []
        elif tag in {"td", "th"} and self.current_row is not None:
            self.current_cell = []
            self.in_cell = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in {"td", "th"} and self.in_cell:
            text = normalize_text("".join(self.current_cell or []))
            self.current_row.append(text)
            self.current_cell = None
            self.in_cell = False
        elif tag == "tr" and self.current_row is not None:
            if any(self.current_row):
                self.current_table.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.current_table is not None:
            if self.current_table:
                self.tables.append(self.current_table)
            self.current_table = None

    def handle_data(self, data: str) -> None:
        if self.in_cell and self.current_cell is not None:
            self.current_cell.append(data)


def generate_normalized_key(transaction: dict) -> str:
    source_type = normalize_text(transaction.get("source_type"))
    is_card_row = (
        source_type in {"credit_card", "debit_card"}
        or bool(transaction.get("charge_date"))
        or (
            source_type == "unknown"
            and bool(transaction.get("card_last4"))
            and bool(transaction.get("merchant"))
            and not bool(transaction.get("posting_date"))
        )
    )

    if is_card_row:
        parts = [
            source_type,
            normalize_card_last4(transaction.get("card_last4")),
            normalize_text(transaction.get("transaction_date")),
            normalize_text(transaction.get("charge_date")),
            normalize_text(transaction.get("merchant")).casefold(),
            f"{float(transaction.get('amount', 0) or 0):.2f}",
            normalize_text(transaction.get("currency")),
        ]
    else:
        parts = [
            source_type,
            normalize_text(transaction.get("posting_date")),
            normalize_text(transaction.get("description")).casefold(),
            f"{float(transaction.get('amount', 0) or 0):.2f}",
            normalize_text(transaction.get("bank_reference")),
            normalize_text(transaction.get("currency")),
        ]

    raw_key = "|".join(parts)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def ensure_transactions_table(db_path: str) -> None:
    # Safe migration: create the transaction layer without touching old reports.
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        column_sql = ",\n            ".join(
            f"{name} {definition}"
            for name, definition in TRANSACTION_COLUMNS.items()
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS transactions (
                {column_sql}
            )
            """
        )

        existing_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(transactions)").fetchall()
        }

        for name, definition in TRANSACTION_COLUMNS.items():
            if name in existing_columns:
                continue
            if name == "id":
                continue
            conn.execute(f"ALTER TABLE transactions ADD COLUMN {name} {definition}")

        for index_name, column_name in TRANSACTION_INDEXES.items():
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON transactions({column_name})"
            )

        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_transactions_normalized_key_unique
            ON transactions(normalized_key)
            WHERE normalized_key IS NOT NULL
            """
        )

        # Backfill/recompute derived month fields for existing databases.
        # These values are deterministic projections of existing dates, so it is
        # safe to refresh them without touching amounts or normalized keys.
        rows = conn.execute(
            """
            SELECT id, source_type, transaction_date, posting_date, charge_date, description,
                   transaction_month, cashflow_month, statement_month
            FROM transactions
            """
        ).fetchall()

        for row in rows:
            item = {
                "source_type": row[1],
                "transaction_date": row[2],
                "posting_date": row[3],
                "charge_date": row[4],
                "description": row[5],
                "transaction_month": None,
                "cashflow_month": None,
                "statement_month": row[8],
            }
            item = apply_transaction_months(item)
            conn.execute(
                """
                UPDATE transactions
                SET transaction_month = ?, cashflow_month = ?, statement_month = ?
                WHERE id = ?
                """,
                (
                    item.get("transaction_month"),
                    item.get("cashflow_month"),
                    item.get("statement_month"),
                    row[0],
                ),
            )


def normalize_transaction_for_insert(transaction: dict) -> dict:
    item = dict(transaction)

    amount = parse_amount(item.get("amount"))
    item["amount"] = amount
    item["amount_abs"] = abs(amount)
    item["currency"] = normalize_currency(item.get("currency"))
    item["direction"] = item.get("direction") or direction_from_amount(amount)
    item["transaction_date"] = normalize_date(item.get("transaction_date"))
    item["posting_date"] = normalize_date(item.get("posting_date"))
    item["charge_date"] = normalize_date(item.get("charge_date"))
    item = apply_transaction_months(item)
    item = apply_transaction_category_overrides(item)
    item["normalized_key"] = item.get("normalized_key") or generate_normalized_key(item)
    item["raw_row_json"] = item.get("raw_row_json") or json.dumps(
        item.get("raw_row", {}),
        ensure_ascii=False,
    )

    if "original_amount" in item and item["original_amount"] not in (None, ""):
        item["original_amount"] = parse_amount(item["original_amount"])
    else:
        item["original_amount"] = None

    item["original_currency"] = (
        normalize_currency(item.get("original_currency"))
        if item.get("original_currency")
        else None
    )

    for key in TRANSACTION_COLUMNS:
        if key not in item and key not in {"id", "created_at"}:
            item[key] = None

    return item


def save_transactions_to_db(transactions: list[dict], db_path: str | None = None) -> dict:
    db_path = db_path or default_db_path()
    ensure_transactions_table(db_path)

    total_received_count = len(transactions)
    inserted_count = 0
    skipped_duplicates_count = 0

    insert_columns = [
        column
        for column in TRANSACTION_COLUMNS
        if column not in {"id", "created_at"}
    ]
    placeholders = ", ".join("?" for _ in insert_columns)
    columns_sql = ", ".join(insert_columns)

    with sqlite3.connect(db_path) as conn:
        for transaction in transactions:
            item = normalize_transaction_for_insert(transaction)
            values = [item.get(column) for column in insert_columns]
            cursor = conn.execute(
                f"""
                INSERT OR IGNORE INTO transactions ({columns_sql})
                VALUES ({placeholders})
                """,
                values,
            )

            if cursor.rowcount:
                inserted_count += 1
            else:
                skipped_duplicates_count += 1

    return {
        "total_received_count": total_received_count,
        "inserted_count": inserted_count,
        "skipped_duplicates_count": skipped_duplicates_count,
    }


def spending_type_placeholders() -> str:
    return ", ".join("?" for _ in SPENDING_TRANSACTION_TYPES)


def latest_spending_transaction_month(conn: sqlite3.Connection) -> str | None:
    # Monthly reports default to the latest purchase/activity month, not the
    # latest cashflow month, because the dashboard is currently spending-focused.
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


def load_spending_transactions_from_db(
    analysis_month: str = "",
    db_path: str | None = None,
) -> pd.DataFrame:
    # This is the new analytics boundary. It converts persisted transaction rows
    # into the same shape the old CSV analytics functions expected.
    db_path = db_path or default_db_path()
    ensure_transactions_table(db_path)

    with sqlite3.connect(db_path) as conn:
        selected_month = normalize_text(analysis_month) or latest_spending_transaction_month(conn)

        if not selected_month:
            return empty_analytics_spending_df()

        params = [BASE_ANALYTICS_CURRENCY, *list(SPENDING_TRANSACTION_TYPES), selected_month]
        query = f"""
            SELECT
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
        return empty_analytics_spending_df()

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
    df["normalized_category_en"] = df.apply(
        lambda row: CATEGORY_LABELS.get(row.get("normalized_category"))
        or normalize_text(row.get("category_en"))
        or "Other",
        axis=1,
    )
    df["amount"] = df["amount"].astype(float)
    df["analysis_source"] = "transactions_table"
    df["analysis_month"] = selected_month

    return df


def empty_analytics_spending_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "transaction_date",
            "posting_date",
            "charge_date",
            "transaction_month",
            "cashflow_month",
            "merchant",
            "description",
            "category",
            "category_en",
            "signed_amount",
            "amount",
            "currency",
            "direction",
            "source_type",
            "transaction_type",
            "card_last4",
            "account_last4",
            "account_id",
            "counterparty",
            "bank_reference",
            "original_amount",
            "original_currency",
            "normalized_category",
            "normalized_category_en",
            "analysis_source",
            "analysis_month",
        ]
    )


def load_analytics_spending_df(csv_path: str = "", analysis_month: str = "") -> pd.DataFrame:
    # Prefer persisted transactions. The CSV fallback keeps old tests and flows
    # usable when no transaction import has been performed yet.
    db_df = load_spending_transactions_from_db(analysis_month=analysis_month)
    if not db_df.empty:
        return db_df

    df = load_normalized_bank_csv(csv_path).copy()

    # Legacy CSV rows do not have explicit direction. Positive amounts are kept
    # as expenses; negative rows are excluded so credits/refunds do not appear as
    # Large Expenses to Review.
    df = df[df["amount"] > 0].copy()
    if "currency" in df.columns:
        df = df[df["currency"].apply(normalize_currency) == BASE_ANALYTICS_CURRENCY].copy()
    if "transaction_month" not in df.columns:
        df["transaction_month"] = df.apply(
            lambda row: month_from_date(row.get("transaction_date")),
            axis=1,
        )
    if "cashflow_month" not in df.columns:
        df["cashflow_month"] = df.apply(
            lambda row: month_from_date(row.get("charge_date"))
            or month_from_date(row.get("transaction_date")),
            axis=1,
        )
    df["signed_amount"] = df["amount"]
    df["direction"] = "debit"
    df["source_type"] = "csv_fallback"
    df["transaction_type"] = "real_expense"
    df["analysis_source"] = "csv_fallback"
    df["analysis_month"] = None

    return df


def xlsx_cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("a:v", XLSX_NS)
    inline = cell.find("a:is/a:t", XLSX_NS)

    if inline is not None:
        return inline.text or ""

    if value is None:
        return ""

    raw = value.text or ""

    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except Exception:
            return raw

    return raw


def read_xlsx_rows(path: str) -> list[dict]:
    # Lightweight XLSX reader for the card-export format.
    # Avoids requiring openpyxl just to parse rows from a simple workbook.
    rows: list[dict] = []

    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", XLSX_NS):
                parts = [text.text or "" for text in item.findall(".//a:t", XLSX_NS)]
                shared_strings.append("".join(parts))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("rel:Relationship", XLSX_NS)
        }

        for sheet in workbook.findall("a:sheets/a:sheet", XLSX_NS):
            sheet_name = sheet.attrib.get("name", "")
            rel_id = sheet.attrib.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            target = rel_map[rel_id]
            sheet_path = "xl/" + target.lstrip("/")
            root = ET.fromstring(archive.read(sheet_path))
            header: list[str] | None = None

            for row in root.findall("a:sheetData/a:row", XLSX_NS):
                values = [
                    xlsx_cell_text(cell, shared_strings)
                    for cell in row.findall("a:c", XLSX_NS)
                ]

                if not any(values):
                    continue

                if "תאריך עסקה" in values and "שם בית העסק" in values:
                    header = values
                    continue

                if not header:
                    continue

                row_data = {
                    header[index]: values[index] if index < len(values) else ""
                    for index in range(len(header))
                }
                row_data["_sheet_name"] = sheet_name
                rows.append(row_data)

    return rows


def parse_credit_card_transactions(path: str) -> list[dict]:
    source_file = str(Path(path).resolve())
    transactions: list[dict] = []

    for row in read_xlsx_rows(path):
        raw_transaction_date = normalize_text(row.get("תאריך עסקה"))
        merchant = normalize_text(row.get("שם בית העסק"))
        raw_amount = normalize_text(row.get("סכום חיוב"))

        if not raw_transaction_date or raw_transaction_date == "סך הכל":
            continue

        if not merchant or not raw_amount:
            continue

        amount = parse_amount(raw_amount)
        currency = normalize_currency(row.get("מטבע חיוב"))
        original_amount = row.get("סכום עסקה מקורי") or None
        original_currency = row.get("מטבע עסקה מקורי") or None
        card_last4 = normalize_card_last4(row.get("4 ספרות אחרונות של כרטיס האשראי"))
        source_type = detect_card_source_type(card_last4)
        counterparty = extract_counterparty(row.get("הערות"))

        transaction = {
            "source_file": source_file,
            "source_type": source_type,
            "transaction_type": "real_expense" if source_type in {"credit_card", "debit_card"} else "unknown",
            "transaction_date": normalize_date(raw_transaction_date),
            "posting_date": None,
            "charge_date": normalize_date(row.get("תאריך חיוב")),
            "merchant": merchant,
            "description": merchant,
            "category": normalize_text(row.get("קטגוריה")),
            "category_en": HEBREW_CATEGORY_LABELS.get(normalize_text(row.get("קטגוריה")), "Other"),
            "amount": amount,
            "amount_abs": abs(amount),
            "currency": currency,
            "direction": direction_from_amount(amount),
            "card_last4": card_last4,
            "counterparty": counterparty,
            "bank_reference": None,
            "original_amount": original_amount,
            "original_currency": original_currency,
            "raw_row_json": json.dumps(row, ensure_ascii=False),
        }
        # Card rows keep both analytical views: purchase month for spending and
        # charge/posting month for cash-flow analysis.
        transaction = apply_transaction_months(transaction)
        transaction["normalized_key"] = generate_normalized_key(transaction)
        transactions.append(transaction)

    return transactions


def infer_csv_source_type(df: pd.DataFrame) -> str:
    if "card_last_4" in df.columns:
        cards = {
            detect_card_source_type(value)
            for value in df["card_last_4"].dropna().tolist()
        }

        if len(cards) == 1:
            return next(iter(cards))

        if cards - {"unknown"}:
            return "unknown"

    if "charge_date" in df.columns:
        return "unknown"

    return "unknown"


def parse_csv_transactions(path: str) -> list[dict]:
    # CSV transaction import reuses the same parser/normalizer as monthly analysis.
    # This keeps transaction storage consistent with existing reports.
    source_file = str(Path(path).resolve())
    df = load_normalized_bank_csv(path)
    source_type = infer_csv_source_type(df)
    transactions: list[dict] = []

    for _, row in df.iterrows():
        amount = float(row.get("amount", 0) or 0)
        transaction_date = normalize_date(row.get("transaction_date"))
        charge_date = normalize_date(row.get("charge_date")) if "charge_date" in df.columns else None
        card_last4 = None

        if "card_last_4" in df.columns:
            card_last4 = normalize_card_last4(row.get("card_last_4"))
        elif "source_id" in df.columns:
            card_last4 = normalize_card_last4(row.get("source_id"))

        row_source_type = detect_card_source_type(card_last4) if card_last4 else source_type

        transaction = {
            "source_file": source_file,
            "source_type": row_source_type,
            "transaction_type": "real_expense" if row_source_type in {"credit_card", "debit_card"} else "unknown",
            "transaction_date": transaction_date,
            "posting_date": None,
            "charge_date": charge_date,
            "merchant": normalize_text(row.get("merchant")),
            "description": normalize_text(row.get("merchant")),
            "category": normalize_text(row.get("category")),
            "category_en": normalize_text(row.get("normalized_category_en")),
            "amount": amount,
            "amount_abs": abs(amount),
            "currency": normalize_currency(row.get("currency")) if "currency" in df.columns else "NIS",
            "direction": direction_from_amount(amount),
            "card_last4": card_last4,
            "bank_reference": None,
            "original_amount": None,
            "original_currency": None,
            "raw_row_json": json.dumps(row.to_dict(), ensure_ascii=False, default=str),
        }
        # CSV rows may contain either card-like or bank-like exports. The source
        # type plus available dates decides the two month fields consistently.
        transaction = apply_transaction_months(transaction)
        transaction["normalized_key"] = generate_normalized_key(transaction)
        transactions.append(transaction)

    return transactions


def read_html_tables_stdlib(path: str) -> list[pd.DataFrame]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    parser = SimpleHtmlTableParser()
    parser.feed(text)
    dataframes: list[pd.DataFrame] = []

    for table in parser.tables:
        if not table:
            continue

        max_columns = max(len(row) for row in table)
        normalized_rows = [
            row + [""] * (max_columns - len(row))
            for row in table
        ]
        dataframes.append(pd.DataFrame(normalized_rows))

    return dataframes


def read_bank_account_tables(path: str) -> list[pd.DataFrame]:
    try:
        return pd.read_excel(path, sheet_name=None, header=None).values()
    except Exception:
        pass

    try:
        return pd.read_html(path)
    except Exception:
        return read_html_tables_stdlib(path)


def rows_look_like_card_statement(rows: list[dict]) -> bool:
    if not rows:
        return False

    columns = set(rows[0].keys())
    return {
        "תאריך עסקה",
        "שם בית העסק",
        "4 ספרות אחרונות של כרטיס האשראי",
        "סכום חיוב",
    }.issubset(columns)


def tables_look_like_bank_account(tables: list[pd.DataFrame]) -> bool:
    try:
        find_bank_transaction_table(tables)
        return True
    except Exception:
        return False


def find_bank_transaction_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    required = {"תאריך", "תיאור"}

    for table in tables:
        table = table.fillna("")

        for index, row in table.iterrows():
            values = [normalize_text(value) for value in row.tolist()]

            if required.issubset(set(values)) and ("בחובה" in values or "בזכות" in values):
                header = values
                data = table.iloc[index + 1:].copy()
                data.columns = header[: len(data.columns)]
                data = data.loc[:, [column for column in data.columns if normalize_text(column)]]
                return data

    raise ValueError("Could not find bank account transaction table")


def get_row_value(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row.index:
            return normalize_text(row.get(name))
    return ""


def parse_bank_account_transactions(path: str) -> list[dict]:
    source_file = str(Path(path).resolve())
    table = find_bank_transaction_table(list(read_bank_account_tables(path)))
    transactions: list[dict] = []

    for _, row in table.iterrows():
        posting_date = normalize_date(get_row_value(row, "תאריך"))
        value_date = normalize_date(get_row_value(row, "תאריך ערך"))
        description = get_row_value(row, "תיאור")
        expanded_description = get_row_value(row, "תאור מורחב", "פירוט נוסף")
        bank_reference = get_row_value(row, "אסמכתא")
        debit_text = get_row_value(row, "בחובה")
        credit_text = get_row_value(row, "בזכות")

        if not posting_date or not description:
            continue

        debit = parse_amount(debit_text) if debit_text else 0.0
        credit = parse_amount(credit_text) if credit_text else 0.0

        if debit > 0:
            amount = -debit
            direction = "debit"
        elif credit > 0:
            amount = credit
            direction = "credit"
        else:
            continue

        full_description = normalize_text(f"{description} {expanded_description}")
        # Bank spending month follows the purchase date embedded in the text when
        # present; otherwise it deliberately falls back to posting_date.
        transaction_date = extract_transaction_date_from_description(full_description) or posting_date
        card_last4 = extract_card_last4(full_description)
        transaction_type = classify_bank_transaction(description, expanded_description, direction)

        transaction = {
            "source_file": source_file,
            "source_type": "bank_account",
            "transaction_type": transaction_type,
            "transaction_date": transaction_date,
            "posting_date": posting_date,
            "charge_date": None,
            "merchant": extract_merchant_from_bank_description(description, expanded_description),
            "description": full_description,
            "category": None,
            "category_en": None,
            "amount": amount,
            "amount_abs": abs(amount),
            "currency": "NIS",
            "direction": direction,
            "card_last4": card_last4,
            "account_last4": None,
            "account_id": None,
            "bank_reference": bank_reference,
            "original_amount": None,
            "original_currency": None,
            "raw_row_json": json.dumps(row.to_dict(), ensure_ascii=False, default=str),
        }
        # For bank rows, transaction_date can come from the description; posting
        # date remains the authoritative cash-flow month.
        transaction = apply_transaction_months(transaction)
        transaction["normalized_key"] = generate_normalized_key(transaction)
        transactions.append(transaction)

    return transactions


def parse_transactions_from_file(path: str) -> tuple[list[dict], str]:
    # Extension chooses the reader; file content chooses the financial source/parser.
    suffix = Path(path).suffix.lower()

    if suffix in {".xlsx", ".xlsm"}:
        rows = read_xlsx_rows(path)
        if rows_look_like_card_statement(rows):
            return parse_credit_card_transactions(path), "card_statement"
        raise ValueError("Unsupported .xlsx content: no recognized card statement table")

    if suffix == ".xls":
        tables = list(read_bank_account_tables(path))
        if tables_look_like_bank_account(tables):
            return parse_bank_account_transactions(path), "bank_account"

        try:
            rows = read_xlsx_rows(path)
            if rows_look_like_card_statement(rows):
                return parse_credit_card_transactions(path), "card_statement"
        except Exception:
            pass

        raise ValueError("Unsupported .xls content: no recognized bank or card table")

    if suffix == ".csv":
        return parse_csv_transactions(path), "csv_statement"

    raise ValueError(f"Unsupported transaction file type: {suffix}")


def summarize_transaction_import(transactions: list[dict], save_result: dict) -> dict:
    dates = [
        value
        for transaction in transactions
        for value in [transaction.get("transaction_date"), transaction.get("posting_date"), transaction.get("charge_date")]
        if value
    ]
    transaction_months = [
        transaction.get("transaction_month")
        for transaction in transactions
        if transaction.get("transaction_month")
    ]
    cashflow_months = [
        transaction.get("cashflow_month")
        for transaction in transactions
        if transaction.get("cashflow_month")
    ]
    card_last4_values = sorted(
        {
            normalize_text(transaction.get("card_last4"))
            for transaction in transactions
            if normalize_text(transaction.get("card_last4"))
        }
    )

    return {
        **save_result,
        "min_date": min(dates) if dates else None,
        "max_date": max(dates) if dates else None,
        "min_transaction_month": min(transaction_months) if transaction_months else None,
        "max_transaction_month": max(transaction_months) if transaction_months else None,
        "min_cashflow_month": min(cashflow_months) if cashflow_months else None,
        "max_cashflow_month": max(cashflow_months) if cashflow_months else None,
        "missing_transaction_month_count": len(transactions) - len(transaction_months),
        "missing_cashflow_month_count": len(transactions) - len(cashflow_months),
        "distinct_card_last4": card_last4_values,
        "distinct_card_last4_count": len(card_last4_values),
    }


def date_distance_days(left: str | None, right: str | None) -> int | None:
    if not left or not right:
        return None

    try:
        left_date = datetime.fromisoformat(left).date()
        right_date = datetime.fromisoformat(right).date()
    except ValueError:
        return None

    return (left_date - right_date).days


def text_similarity_enough(left: str | None, right: str | None) -> bool:
    left_text = normalize_text(left).casefold()
    right_text = normalize_text(right).casefold()

    if not left_text or not right_text:
        return False

    if left_text in right_text or right_text in left_text:
        return True

    left_tokens = set(left_text.split())
    right_tokens = set(right_text.split())

    if not left_tokens or not right_tokens:
        return False

    overlap = len(left_tokens & right_tokens)
    return overlap / min(len(left_tokens), len(right_tokens)) >= 0.5


def mark_duplicate_transactions_in_db(db_path: str | None = None) -> dict:
    db_path = db_path or default_db_path()
    ensure_transactions_table(db_path)

    debit_card_duplicate_groups_found = 0
    debit_card_duplicates_marked = 0
    credit_card_settlements_excluded = 0
    ambiguous_matches_count = 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        total_transactions_checked = conn.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0]

        conn.execute(
            """
            UPDATE transactions
            SET duplicate_group_id = NULL,
                duplicate_status = 'unique',
                duplicate_of_transaction_id = NULL,
                duplicate_reason = NULL,
                analytics_excluded = 0,
                analytics_exclusion_reason = NULL
            """
        )

        settlement_cursor = conn.execute(
            """
            UPDATE transactions
            SET analytics_excluded = 1,
                analytics_exclusion_reason = 'credit_card_monthly_settlement'
            WHERE source_type = 'bank_account'
              AND transaction_type = 'card_settlement'
            """
        )
        credit_card_settlements_excluded = settlement_cursor.rowcount

        bank_rows = conn.execute(
            """
            SELECT *
            FROM transactions
            WHERE source_type = 'bank_account'
              AND card_last4 IN (?, ?)
              AND direction = 'debit'
              AND analytics_excluded = 0
            """,
            tuple(DEBIT_CARD_LAST4),
        ).fetchall()

        for bank_row in bank_rows:
            card_candidates = conn.execute(
                """
                SELECT *
                FROM transactions
                WHERE source_type != 'bank_account'
                  AND card_last4 = ?
                  AND direction = 'debit'
                  AND ABS(amount_abs - ?) <= 0.01
                """,
                (bank_row["card_last4"], bank_row["amount_abs"]),
            ).fetchall()

            matches = []

            for card_row in card_candidates:
                same_transaction_date = (
                    bank_row["transaction_date"]
                    and card_row["transaction_date"]
                    and bank_row["transaction_date"] == card_row["transaction_date"]
                )
                posting_gap = date_distance_days(
                    bank_row["posting_date"],
                    card_row["transaction_date"],
                )
                date_close = posting_gap is not None and 0 <= posting_gap <= 3
                text_close = text_similarity_enough(
                    bank_row["description"],
                    card_row["merchant"],
                )

                if (same_transaction_date or date_close) and text_close:
                    matches.append(card_row)
                elif same_transaction_date or date_close:
                    ambiguous_matches_count += 1

            if len(matches) != 1:
                continue

            primary = matches[0]
            duplicate_group_id = hashlib.sha256(
                f"debit-card-duplicate|{primary['id']}|{bank_row['id']}".encode("utf-8")
            ).hexdigest()

            conn.execute(
                """
                UPDATE transactions
                SET duplicate_group_id = ?,
                    duplicate_status = 'primary'
                WHERE id = ?
                """,
                (duplicate_group_id, primary["id"]),
            )
            conn.execute(
                """
                UPDATE transactions
                SET duplicate_group_id = ?,
                    duplicate_status = 'duplicate',
                    duplicate_of_transaction_id = ?,
                    duplicate_reason = 'duplicate_debit_card_transaction',
                    analytics_excluded = 1,
                    analytics_exclusion_reason = 'duplicate_debit_card_transaction'
                WHERE id = ?
                """,
                (duplicate_group_id, primary["id"], bank_row["id"]),
            )

            debit_card_duplicate_groups_found += 1
            debit_card_duplicates_marked += 1

        analytics_excluded_total = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE analytics_excluded = 1"
        ).fetchone()[0]

    return {
        "total_transactions_checked": total_transactions_checked,
        "debit_card_duplicate_groups_found": debit_card_duplicate_groups_found,
        "debit_card_duplicates_marked": debit_card_duplicates_marked,
        "credit_card_settlements_excluded": credit_card_settlements_excluded,
        "ambiguous_matches_count": ambiguous_matches_count,
        "analytics_excluded_total": analytics_excluded_total,
    }


def file_cache_key(csv_path: str) -> tuple[str, int, int]:
    # The cache key includes file metadata, so edited statements are reloaded.
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    stat = path.stat()

    return (
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def remember_cached_frame(
    cache: dict[tuple[str, int, int], pd.DataFrame],
    key: tuple[str, int, int],
    df: pd.DataFrame,
) -> None:
    # Store a copy so later transformations cannot mutate cached source data.
    cache[key] = df.copy(deep=True)

    while len(cache) > FINANCE_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


def load_bank_csv_uncached(csv_path: str) -> pd.DataFrame:
    # Input boundary: every finance workflow starts by loading a CSV into
    # a predictable DataFrame schema.
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    preview = pd.read_csv(path, sep=None, engine="python", nrows=5)

    normalized_required = {"transaction_date", "merchant", "category", "amount"}

    # Merged statements are already normalized, so they can skip Hebrew header detection.
    if normalized_required.issubset(set(preview.columns)):
        df = pd.read_csv(path, sep=None, engine="python")

        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["amount"])
        df = df[df["merchant"].notna()]

        return df
    
    # Raw Israeli bank/card exports can contain metadata lines before the real header.
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

    # Convert bank-specific Hebrew column names into app-wide English field names.
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

    # Amounts arrive as localized strings, so normalize them before calculations.
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


def load_bank_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a statement CSV with a small in-process cache.

    Cache keys include path, file size, and mtime, so changed files are
    automatically parsed again. Callers receive a copy to protect cached data.
    """

    if not FINANCE_CACHE_ENABLED:
        return load_bank_csv_uncached(csv_path)

    key = file_cache_key(csv_path)

    if key not in _RAW_STATEMENT_CACHE:
        remember_cached_frame(
            _RAW_STATEMENT_CACHE,
            key,
            load_bank_csv_uncached(csv_path),
        )

    return _RAW_STATEMENT_CACHE[key].copy(deep=True)


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
        # Category normalization gives all later tools stable category names.
        # Merchant keywords refine broad bank categories into more useful groups.
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


def load_normalized_bank_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a statement and add normalized categories with caching.

    This avoids repeating both CSV parsing and category normalization across
    tools such as analyze_statement and find_unusual_expenses in the same run.
    """

    if not FINANCE_CACHE_ENABLED:
        return add_normalized_categories(load_bank_csv_uncached(csv_path))

    key = file_cache_key(csv_path)

    if key not in _NORMALIZED_STATEMENT_CACHE:
        remember_cached_frame(
            _NORMALIZED_STATEMENT_CACHE,
            key,
            add_normalized_categories(load_bank_csv(csv_path)),
        )

    return _NORMALIZED_STATEMENT_CACHE[key].copy(deep=True)


# MCP tool: exposes row-level categorization for debugging and detailed inspection.
@mcp.tool()
def categorize_transactions(csv_path: str = "", analysis_month: str = "") -> str:
    """
    Return transactions with original bank category and normalized category.
    """

    df = load_analytics_spending_df(csv_path, analysis_month)

    result = df[
        [
            "transaction_date",
            "transaction_month",
            "cashflow_month",
            "merchant",
            "category",
            "normalized_category",
            "normalized_category_en",
            "amount",
        ]
    ].to_dict(orient="records")

    return json.dumps(result, ensure_ascii=False, indent=2)


# MCP tool: main deterministic summary used by the dashboard and report.
@mcp.tool()
def analyze_statement(csv_path: str = "", analysis_month: str = "") -> str:
    """
    Analyze monthly spending and return a clean structured summary.

    The preferred source is the persisted transactions table. csv_path is kept
    for compatibility and only used as a fallback when the table is empty.
    Merchant names stay in original Hebrew.
    Categories are shown in Hebrew and English.
    """

    df = load_analytics_spending_df(csv_path, analysis_month)

    transactions_count = int(len(df))
    total_spent = round(float(df["amount"].sum()), 2) if transactions_count else 0
    average_transaction = round(float(df["amount"].mean()), 2) if transactions_count else 0
    selected_month = normalize_text(df["analysis_month"].iloc[0]) if transactions_count else None
    analysis_source = normalize_text(df["analysis_source"].iloc[0]) if transactions_count else "empty"

    # Category and merchant rankings are the main dashboard inputs.
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
        "analysis_source": analysis_source,
        "analysis_month": selected_month,
        "month_field": "transaction_month",
        "total_spent": total_spent,
        "transactions_count": transactions_count,
        "average_transaction": average_transaction,
        "top_categories": top_categories,
        "top_merchants": top_merchants,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# MCP tool: focused category aggregation, useful if the agent needs only category data.
@mcp.tool()
def get_spending_summary(month: str = "") -> str:
    """
    Return read-only spending summary for one transaction_month.
    """

    result = query_tools.get_spending_summary(month=month)
    return json.dumps(result, ensure_ascii=False, indent=2)


# MCP tool: focused category aggregation for transaction-history questions.
@mcp.tool()
def get_category_breakdown(month: str = "", csv_path: str = "", analysis_month: str = "") -> str:
    """
    Return spending grouped by normalized category.
    """

    selected_month = month or analysis_month
    result = query_tools.get_category_breakdown(month=selected_month)

    return json.dumps(result, ensure_ascii=False, indent=2)


# MCP tool: focused merchant aggregation for transaction-history questions.
@mcp.tool()
def get_top_merchants(month: str = "", limit: int = 10, csv_path: str = "", analysis_month: str = "") -> str:
    """
    Return top merchants by total spending.
    """
    selected_month = month or analysis_month
    result = query_tools.get_top_merchants(month=selected_month, limit=limit)

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_largest_transactions(month: str = "", limit: int = 10) -> str:
    """
    Return largest transactions for one transaction_month.
    """

    result = query_tools.get_largest_transactions(month=month, limit=limit)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_unusual_transactions(month: str = "") -> str:
    """
    Return unusual transactions for one transaction_month using deterministic thresholds.
    """

    result = query_tools.get_unusual_transactions(month=month)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def compare_months(month_a: str, month_b: str) -> str:
    """
    Compare spending between two transaction_month values.
    """

    result = query_tools.compare_months(month_a=month_a, month_b=month_b)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_category_comparison(category: str, month_a: str, month_b: str) -> str:
    """
    Compare one normalized category between two transaction_month values.
    """

    result = query_tools.get_category_comparison(
        category=category,
        month_a=month_a,
        month_b=month_b,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_category_trend(category: str, months: int = 6) -> str:
    """
    Return monthly totals for a category across recent months.
    """

    result = query_tools.get_category_trend(category=category, months=months)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_recurring_merchants(months: int = 6) -> str:
    """
    Return merchants that appear in multiple recent months.
    """

    result = query_tools.get_recurring_merchants(months=months)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def prepare_financial_review_context(months: int = 6) -> str:
    """
    Return deterministic multi-month context for financial review questions.
    """

    result = query_tools.prepare_financial_review_context(months=months)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def search_transactions(keyword: str, limit: int = 50) -> str:
    """
    Search transaction history by merchant, description, category, or counterparty.
    """

    result = query_tools.search_transactions(keyword=keyword, limit=limit)
    return json.dumps(result, ensure_ascii=False, indent=2)


# MCP tool: builds the final markdown report from already-computed tool outputs.
@mcp.tool()
def generate_monthly_report(analysis_json: str, unusual_json: str, advice_text: str) -> str:
    """
    Generate a clean monthly finance report from tool outputs.
    """

    # The report is assembled from previous tool outputs.
    # It should not introduce new financial facts.
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
            f"- {format_merchant_display(item.get('merchant'))}: {item.get('total_amount', 0):.2f} NIS "
            f"({item.get('transactions_count', 0)} transactions)"
        )

    large_expenses = unusual.get("large_expenses_to_review", [])

    lines.append("")
    lines.append("## Large or Unusial Expenses to Review")
    if large_expenses:
        for item in large_expenses[:5]:
            lines.append(
                f"- {item.get('transaction_date')}: {format_merchant_display(item.get('merchant'))} — "
                f"{item.get('amount')} NIS ({item.get('normalized_category_en')})"
            )
    else:
        lines.append("- No large expenses detected.")

    lines.append("")
    lines.append("## Savings Advice")
    lines.append(advice_text)

    return "\n".join(lines)


# MCP tool: prepares structured facts that the LLM can interpret safely.
@mcp.tool()
def prepare_financial_insight_context(analysis_json: str, unusual_json: str) -> str:
    """
    Prepare structured financial context for LLM interpretation.

    This tool does not generate natural language advice. It only transforms
    deterministic analysis outputs into a compact JSON context.
    """

    if not analysis_json or not analysis_json.strip():
        return json.dumps(
            {"error": "analysis_json is empty. Call finance.analyze_statement first."},
            ensure_ascii=False,
            indent=2,
        )

    if not unusual_json or not unusual_json.strip():
        return json.dumps(
            {"error": "unusual_json is empty. Call finance.find_unusual_expenses first."},
            ensure_ascii=False,
            indent=2,
        )

    # This is the bridge between deterministic finance tools and the LLM.
    # It prepares facts for interpretation, but does not write advice itself.
    analysis = json.loads(analysis_json)
    unusual = json.loads(unusual_json)

    total_spent = float(analysis.get("total_spent", 0) or 0)
    top_categories = analysis.get("top_categories", []) or []
    top_merchants = analysis.get("top_merchants", []) or []
    large_expenses = unusual.get("large_expenses_to_review", []) or []

    # Add shares so the LLM can discuss concentration without recalculating.
    largest_spending_categories = []
    for item in top_categories[:5]:
        amount = float(item.get("total_amount", 0) or 0)
        largest_spending_categories.append(
            {
                "category": item.get("category"),
                "category_en": item.get("category_en"),
                "total_amount": round(amount, 2),
                "share_of_total_percent": round((amount / total_spent * 100), 2) if total_spent else 0,
                "transactions_count": item.get("transactions_count", 0),
                "average_transaction": item.get("average_transaction", 0),
            }
        )

    # Repeated merchants often reveal habits/subscriptions worth reviewing.
    recurring_merchants = []
    for item in top_merchants:
        count = int(item.get("transactions_count", 0) or 0)
        if count < 2:
            continue

        amount = float(item.get("total_amount", 0) or 0)
        recurring_merchants.append(
            {
                "merchant": item.get("merchant"),
                "total_amount": round(amount, 2),
                "share_of_total_percent": round((amount / total_spent * 100), 2) if total_spent else 0,
                "transactions_count": count,
                "average_transaction": item.get("average_transaction", 0),
            }
        )

    top_category_amount = sum(
        float(item.get("total_amount", 0) or 0)
        for item in top_categories[:1]
    )
    top_3_category_amount = sum(
        float(item.get("total_amount", 0) or 0)
        for item in top_categories[:3]
    )

    # Concentration tells whether spending is spread out or dominated by a few categories.
    category_concentration = {
        "top_category_share_percent": round((top_category_amount / total_spent * 100), 2) if total_spent else 0,
        "top_3_categories_share_percent": round((top_3_category_amount / total_spent * 100), 2) if total_spent else 0,
        "top_category": top_categories[0].get("category_en") if top_categories else None,
    }

    spending_patterns = []

    if largest_spending_categories:
        spending_patterns.append(
            {
                "type": "largest_category",
                "description": "Largest category by total amount",
                "data": largest_spending_categories[0],
            }
        )

    if category_concentration["top_3_categories_share_percent"] >= 60:
        spending_patterns.append(
            {
                "type": "category_concentration",
                "description": "Most spending is concentrated in the top 3 categories",
                "data": category_concentration,
            }
        )

    if recurring_merchants:
        spending_patterns.append(
            {
                "type": "recurring_merchants",
                "description": "Merchants with repeated transactions",
                "data": recurring_merchants[:5],
            }
        )

    if large_expenses:
        spending_patterns.append(
            {
                "type": "unusual_expenses",
                "description": "Transactions above merchant/category thresholds",
                "data": large_expenses[:5],
            }
        )

    # These categories are not automatically "bad"; they are just safer candidates
    # for review because they often include discretionary transactions.
    controllable_categories = {
        "food_and_groceries",
        "restaurants_cafes_bars",
        "home_and_furniture",
        "other",
        "fuel",
    }

    possible_saving_opportunities = [
        {
            "type": "category_review",
            "category": item.get("category"),
            "category_en": item.get("category_en"),
            "total_amount": item.get("total_amount"),
            "share_of_total_percent": item.get("share_of_total_percent"),
            "reason": "High-spend category that may include discretionary transactions",
        }
        for item in largest_spending_categories
        if item.get("category") in controllable_categories
    ]

    possible_saving_opportunities.extend(
        {
            "type": "merchant_review",
            "merchant": item.get("merchant"),
            "total_amount": item.get("total_amount"),
            "transactions_count": item.get("transactions_count"),
            "reason": "Recurring merchant with meaningful monthly spend",
        }
        for item in recurring_merchants[:5]
    )

    possible_saving_opportunities.extend(
        {
            "type": "unusual_expense_review",
            "merchant": item.get("merchant"),
            "transaction_date": item.get("transaction_date"),
            "amount": item.get("amount"),
            "category_en": item.get("normalized_category_en"),
            "reason": "Transaction exceeded the deterministic unusual-expense threshold",
        }
        for item in large_expenses[:5]
    )

    result = {
        "currency": analysis.get("currency", "NIS"),
        "spending_summary": {
            "total_spent": analysis.get("total_spent", 0),
            "transactions_count": analysis.get("transactions_count", 0),
            "average_transaction": analysis.get("average_transaction", 0),
        },
        "largest_spending_categories": largest_spending_categories,
        "recurring_merchants": recurring_merchants[:10],
        "unusual_expenses": large_expenses[:10],
        "category_concentration": category_concentration,
        "spending_patterns": spending_patterns,
        "possible_saving_opportunities": possible_saving_opportunities[:12],
        "llm_guidance": [
            "Use these deterministic facts to explain spending behavior.",
            "Do not invent amounts, percentages, merchants, or categories.",
            "Avoid generic 10-15% reduction advice unless a specific fact supports it.",
            "Focus on observations, review areas, and practical next actions grounded in the data.",
        ],
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# MCP tool: flags transactions that are unusually high for this user's own data.
@mcp.tool()
def find_unusual_expenses(csv_path: str = "", analysis_month: str = "") -> str:
    """
    Find large or unusial expenses to review using merchant-specific thresholds first,
    then normalized-category fallback.
    """
    MIN_LARGE_EXPENSE_NIS = 200

    df = load_analytics_spending_df(csv_path, analysis_month)
    df = df.copy()

    if df.empty:
        return json.dumps(
            {
                "method": "merchant threshold first, normalized-category fallback",
                "rule": "only active debit expense rows from transactions are considered",
                "analysis_source": "empty",
                "analysis_month": None,
                "merchant_thresholds": {},
                "category_thresholds": {},
                "large_expenses_to_review": [],
            },
            ensure_ascii=False,
            indent=2,
        )

    selected_month = normalize_text(df["analysis_month"].iloc[0])
    analysis_source = normalize_text(df["analysis_source"].iloc[0])

    # Thresholds adapt to the user's own history instead of using one fixed limit.
    min_merchant_transactions = 3
    min_category_transactions = 3

    merchant_thresholds = {}
    category_thresholds = {}

    # Prefer merchant-specific thresholds when there are enough transactions.
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

    # Fall back to category/global thresholds for sparse merchants.
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

    # A transaction is flagged only when it is both statistically high and meaningful in NIS.
    unusual = df[
        (df["amount"] > df["threshold"]) &
        (df["amount"] >= MIN_LARGE_EXPENSE_NIS)
    ].copy()

    unusual = unusual[
        [
            "transaction_date",
            "transaction_month",
            "cashflow_month",
            "merchant",
            "category",
            "normalized_category",
            "normalized_category_en",
            "amount",
            "signed_amount",
            "direction",
            "source_type",
            "transaction_type",
            "threshold",
            "threshold_method",
        ]
    ].sort_values("amount", ascending=False)

    result = {
        "method": "merchant threshold first, normalized-category fallback",
        "rule": "only active debit expense rows are considered; if merchant has 3+ transactions use merchant 95th percentile, otherwise use normalized category threshold",
        "analysis_source": analysis_source,
        "analysis_month": selected_month,
        "month_field": "transaction_month",
        "merchant_thresholds": merchant_thresholds,
        "category_thresholds": category_thresholds,
        "large_expenses_to_review": unusual.to_dict(orient="records"),
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# MCP tool: legacy deterministic advice path kept for compatibility with older flows.
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

    # Legacy deterministic advice tool kept for compatibility.
    # The current agent workflow prefers AI insights generated from structured context.
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
                f"- {format_merchant_display(merchant.get('merchant'))}: {merchant.get('total_amount', 0):.2f} NIS "
                f"across {merchant.get('transactions_count', 0)} transactions"
            )

    large_expenses = unusual.get("large_expenses_to_review", [])

    if large_expenses:
        advice.append("")
        advice.append("Large or unusial expenses to review:")
        for item in large_expenses[:5]:
            advice.append(
                f"- {item.get('transaction_date')}: {format_merchant_display(item.get('merchant'))} — "
                f"{item.get('amount')} NIS ({item.get('normalized_category_en')})"
            )

    advice.append("")
    advice.append("Suggested next step: set monthly budget limits for the top 3 normalized categories.")

    return "\n".join(advice)



# MCP tool: prepares a SQLite record; the separate SQLite MCP server does the write.
@mcp.tool()
def prepare_monthly_report_record(
    analysis_json: str,
    unusual_json: str,
    advice_text: str,
    monthly_report: str,
) -> str:
    """
    Prepare a clean monthly report record for saving to SQLite.

    This tool does not save anything by itself.
    It only prepares the table name and data payload.
    The LLM should call sqlite.create_record after this tool.
    """

    from datetime import datetime

    if not analysis_json or not analysis_json.strip():
        return json.dumps(
            {"error": "analysis_json is empty. Call finance.analyze_statement first."},
            ensure_ascii=False,
            indent=2,
        )

    if not unusual_json or not unusual_json.strip():
        return json.dumps(
            {"error": "unusual_json is empty. Call finance.find_unusual_expenses first."},
            ensure_ascii=False,
            indent=2,
        )

    if not advice_text or not advice_text.strip():
        return json.dumps(
            {"error": "advice_text is empty. Call finance.generate_ai_financial_insights first."},
            ensure_ascii=False,
            indent=2,
        )

    if not monthly_report or not monthly_report.strip():
        return json.dumps(
            {"error": "monthly_report is empty. Call finance.generate_monthly_report first."},
            ensure_ascii=False,
            indent=2,
        )

    # SQLite is handled by a separate MCP server. This tool only prepares the payload.
    analysis = json.loads(analysis_json)

    result = {
        "table": "monthly_reports",
        "data": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "total_spent": analysis.get("total_spent"),
            "transactions_count": analysis.get("transactions_count"),
            "average_transaction": analysis.get("average_transaction"),
            "analysis_json": analysis_json,
            "unusual_json": unusual_json,
            "advice_text": advice_text,
            "monthly_report": monthly_report,
        },
    }

    return json.dumps(result, ensure_ascii=False, indent=2)

def load_multiple_bank_csvs(csv_files: list[str]) -> pd.DataFrame:
    """
    Load and combine several bank/card CSV statements into one DataFrame.

    Reuses the existing load_bank_csv() function, so each input file is normalized
    exactly like a single-file statement. Adds source_id from the last 4 card digits
    when available, otherwise infers it from the file name.
    """

    frames = []

    for csv_path in csv_files:
        df = load_bank_csv(csv_path)
        df = df.copy()

        # source_id lets the merged file preserve which card/file each row came from.
        source_name = Path(csv_path).stem

        if "card_last_4" in df.columns:
            df["source_id"] = (
                df["card_last_4"]
                .astype(str)
                .str.extract(r"(\d{4})", expand=False)
            )
        else:
            df["source_id"] = None

        inferred = pd.Series([source_name]).str.extract(r"(\d{4})", expand=False).iloc[0]
        if not inferred:
            inferred = source_name

        df["source_id"] = df["source_id"].fillna(str(inferred))
        df["source_file"] = source_name

        frames.append(df)

    if not frames:
        raise ValueError("No CSV files were provided")

    return pd.concat(frames, ignore_index=True)


# MCP tool: combines a directory of statements into one normalized CSV for analysis.
@mcp.tool()
def merge_statements(csv_files: list[str], output_csv_path: str = "data/merged_statement.csv") -> str:
    """
    Merge several bank/card CSV statements into one normalized CSV file.

    This tool does not analyze the data. It only prepares a combined CSV that can be
    passed to the existing finance.analyze_statement and finance.find_unusual_expenses tools.
    """

    if not csv_files:
        return json.dumps(
            {"error": "csv_files is empty. Discover files with filesystem tools first."},
            ensure_ascii=False,
            indent=2,
        )

    # After merging, downstream tools can use the exact same single-file workflow.
    df = load_multiple_bank_csvs(csv_files)

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    result = {
        "merged_csv_path": str(output_path),
        "input_files": csv_files,
        "files_count": len(csv_files),
        "rows_count": int(len(df)),
        "sources_count": int(df["source_id"].nunique()) if "source_id" in df.columns else None,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# MCP tool: parses a machine-readable Israeli credit-card XLSX export.
@mcp.tool()
def parse_credit_card_statement(file_path: str) -> str:
    """
    Parse credit-card Excel transactions into normalized transaction dictionaries.

    This does not write to SQLite. Use finance.save_transactions or
    finance.import_transactions_from_file to persist the parsed rows.
    """

    transactions = parse_credit_card_transactions(file_path)

    return json.dumps(
        {
            "source_type": transactions[0].get("source_type") if transactions else "unknown",
            "source_types": sorted({item.get("source_type") for item in transactions}),
            "source_file": str(Path(file_path).resolve()),
            "total_parsed_count": len(transactions),
            "transactions": transactions,
        },
        ensure_ascii=False,
        indent=2,
    )


# MCP tool: placeholder for a future bank-account parser once a stable input format exists.
@mcp.tool()
def parse_bank_account_statement(file_path: str) -> str:
    """
    Parse a machine-readable bank-account export into normalized transactions.

    Supports HTML-based .xls exports from Leumi. PDF parsing remains deferred.
    """

    transactions = parse_bank_account_transactions(file_path)

    return json.dumps(
        {
            "source_file": str(Path(file_path).resolve()),
            "source_type": "bank_account",
            "total_parsed_count": len(transactions),
            "transactions": transactions,
        },
        ensure_ascii=False,
        indent=2,
    )


# MCP tool: saves normalized transaction dictionaries with row-level deduplication.
@mcp.tool()
def save_transactions(transactions: list[dict], db_path: str = "") -> str:
    """
    Save transactions into SQLite using normalized_key deduplication.

    Re-uploading the same file does not delete existing rows. Only rows with new
    normalized_key values are inserted.
    """

    result = save_transactions_to_db(
        transactions=transactions,
        db_path=db_path or default_db_path(),
    )

    return json.dumps(result, ensure_ascii=False, indent=2)


def bank_import_detection_summary(transactions: list[dict], save_result: dict) -> dict:
    summary = summarize_transaction_import(transactions, save_result)
    transaction_types = [transaction.get("transaction_type") for transaction in transactions]

    return {
        "parsed_count": len(transactions),
        **summary,
        "detected_debit_card_count": transaction_types.count("real_expense"),
        "detected_card_settlement_count": transaction_types.count("card_settlement"),
        "detected_income_count": (
            transaction_types.count("salary_income")
            + transaction_types.count("government_income")
        ),
        "detected_transfer_count": (
            transaction_types.count("bank_transfer")
            + transaction_types.count("internal_transfer")
        ),
        "unknown_count": transaction_types.count("unknown"),
    }


# MCP tool: imports bank-account rows into transactions without duplicate interpretation.
@mcp.tool()
def import_bank_account_statement(file_path: str, db_path: str = "") -> str:
    """
    Parse and save bank-account transactions using normalized_key upload dedup.
    """

    transactions = parse_bank_account_transactions(file_path)
    save_result = save_transactions_to_db(
        transactions=transactions,
        db_path=db_path or default_db_path(),
    )

    return json.dumps(
        {
            "source_type": "bank_account",
            "source_file": str(Path(file_path).resolve()),
            **bank_import_detection_summary(transactions, save_result),
        },
        ensure_ascii=False,
        indent=2,
    )


# MCP tool: marks duplicate/settlement rows for analytics exclusion after import.
@mcp.tool()
def mark_duplicate_transactions(db_path: str = "") -> str:
    """
    Mark duplicate debit-card rows and credit-card settlement rows.

    Rows are not deleted. Excluded rows are flagged with analytics_excluded = 1.
    Future analytics should use WHERE analytics_excluded = 0.
    """

    result = mark_duplicate_transactions_in_db(db_path or default_db_path())
    return json.dumps(result, ensure_ascii=False, indent=2)


# MCP tool: detects supported file type, parses it, and saves only new transactions.
@mcp.tool()
def import_transactions_from_file(file_path: str, db_path: str = "") -> str:
    """
    Import supported transaction files into SQLite.

    Currently supports Israeli card XLSX/CSV exports and HTML-based bank .xls exports.
    """

    try:
        transactions, detected_file_kind = parse_transactions_from_file(file_path)
    except Exception as e:
        return json.dumps(
            {
                "source_file": str(Path(file_path).resolve()),
                "total_received_count": 0,
                "inserted_count": 0,
                "skipped_duplicates_count": 0,
                "error": str(e),
            },
            ensure_ascii=False,
            indent=2,
        )

    save_result = save_transactions_to_db(
        transactions=transactions,
        db_path=db_path or default_db_path(),
    )
    summary = (
        bank_import_detection_summary(transactions, save_result)
        if detected_file_kind == "bank_account"
        else summarize_transaction_import(transactions, save_result)
    )

    result = {
        "detected_file_kind": detected_file_kind,
        "source_types": sorted({item.get("source_type") for item in transactions}),
        "source_file": str(Path(file_path).resolve()),
        **summary,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
