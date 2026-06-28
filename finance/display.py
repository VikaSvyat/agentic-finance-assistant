import re


HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


QUOTE_TRANSLATION = str.maketrans({
    "”": '"',
    "“": '"',
    "״": '"',
    "׳": "'",
})


KNOWN_MERCHANT_ALIASES = {
    "העברה דיגיטל": "Digital Transfer",
    "הוראת קבע": "Standing Order",
    "דמי כרטיס": "Card Fee",
    "קצבת ילדים": "Child Benefit",
    "עיריית חיפה": "Haifa Municipality",
    "עירית חיפה": "Haifa Municipality",
    "המוסד לביטוח לאומי": "National Insurance Institute",
    "חברת החשמל לישראל": "Israel Electric Company",
    "סופר פארם": "Super-Pharm",
    "רמי לוי": "Rami Levy",
    "איקאה": "IKEA",
    "איקאה IKEA": "IKEA",
    "WOLT": "WOLT",
    "מיגס בולדר בע\"מ": "Migs Boulder Ltd",
    "מעדנית אברהמי": "Avrahami Delicatessen",
    "מרכז חינוך ליאו בק": "Leo Baeck Education Center",
    "מרכז חינוך ליאו בק שכ\"ל": "Leo Baeck Education Center",
    "מאפיית אריאל": "Ariel Bakery",
    "מועדון כדורגל כרמל חיפה": "Carmel Haifa Football Club",
    "אושר עד חיפה": "Osher Ad Haifa Discount Supermarket",
    "מור גמל ופנס-י": "Mor Pension Fund",
    "בזק הוראות קבע": "Bezeq Standing Order",
    "בזק": "Bezeq",
    "דר לוריא ילנה": "Dr. Yelena Luria",
    "אברהמי יוסי": "Yossi Avrahami",
    "אמי יצור ביגוד מגן": "Protective Clothing Manufacturing",
    "אמסלם אספקה טכנית": "Amsalem Technical Supply",
    "הפנינג": "Happening",
    "עמותת מטפסי תל אביב-סילבר": "Tel Aviv Climbers Association",
    "קפה בר BIRDY": "BIRDY Cafe Bar",
    "אייר חיפה": "Air Haifa",
    "מנוי קפה הו\"ק": "Coffee Subscription Standing Order",
    "אפליקציית YELLOW מנוי קפה": "YELLOW Coffee Subscription",
    "פז אפליקציית יילו": "Paz YELLOW App",
    "פז / YELLOWIסטלה מאריס": "Paz YELLOW Stella Maris",
    "פז/ YELLOW הדר": "Paz YELLOW Hadar",
    "חנות לא מוכרת": "Unknown Store",
    "אוטודיפו חוצות": "AutoDepo Hutsot",
    "קייט ונופש": "Kite and Vacation",
    "אי מוטורס בע\"מ": "E Motors Ltd",
    "אי מוטורס בעמ": "E Motors Ltd",
    "מגדל חיים": "Migdal Life",
    "לגו - צפון-צמרת": "Lego - North-Top",
    "אפקה-המכללה האקדמית להנדס": "Afeka-Academic College of Engineering",
    "וואי-פיי הנהלת חשבונות": "Wi-Fi Accounting",
    "המוסד לביטוח לאומי -": "National Insurance Institute",
    "רולדין חורב": "Rolladin Horev",
}


KNOWN_BRANDS = {
    "פנגו": "Pango",
    "מוביט": "Moovit",
    "בזק": "Bezeq",
    "סופר": "Super",
    "פארם": "Pharm",
    "סופר פארם": "Super-Pharm",
    "רמי לוי": "Rami Levy",
    "איקאה": "IKEA",
    "WOLT": "WOLT",
    "ילו": "YELLOW",
    "יילו": "YELLOW",
    "YELLOW": "YELLOW",
    "פז": "Paz",
    "אוטודיפו": "AutoDepo",
    "לגו": "Lego",
    "מגדל": "Migdal",
    "רולדין": "Rolladin",
}


COMMON_WORD_TRANSLATIONS = {
    "מ.תחבורה": "Transport",
    "תחבורה": "Transport",
    "מעדנית": "Delicatessen",
    "מאפיית": "Bakery",
    "מאפה": "Bakery",
    "חינוך": "Education",
    "מרכז": "Center",
    "מועדון": "Club",
    "כדורגל": "Football",
    "עיריית": "Municipality",
    "עיריה": "Municipality",
    "עירייה": "Municipality",
    "חשמל": "Electricity",
    "בריאות": "Health",
    "ביטוח": "Insurance",
    "גמל": "Pension Fund",
    "פנס": "Pension",
    "פנסיה": "Pension",
    "העברה": "Transfer",
    "דיגיטל": "Digital",
    "הוראת": "Standing",
    "קבע": "Order",
    "דמי": "Fee",
    "כרטיס": "Card",
    "חנות": "Store",
    "קייט": "Kite",
    "נופש": "Vacation",
    "ספורט": "Sport",
    "אספקה": "Supply",
    "טכנית": "Technical",
    "הכרמל": "Carmel",
    "לישראל": "Israel",
    "חיים": "Life",
    "מוטורס": "Motors",
    "המכללה": "College",
    "האקדמית": "Academic",
    "להנדס": "Engineering",
    "וואי": "Wi",
    "פיי": "Fi",
    "הנהלת": "Management",
    "חשבונות": "Accounting",
    "תיאטרון": "Theater",
    "לא": "",
    "מוכר": "Unknown",
    "מוכרת": "Unknown",
    "בע\"מ": "Ltd",
    "בעמ": "Ltd",
}


KNOWN_PHRASE_TRANSLATIONS = {
    "הוראת קבע": "Standing Order",
    "קבע הוראת": "Standing Order",
    "דמי כרטיס": "Card Fee",
    "העברה דיגיטל": "Digital Transfer",
    "מ.תחבורה": "Transport",
    "ביטוח לאומי": "National Insurance",
    "חברת החשמל": "Electric Company",
    "אספקה טכנית": "Technical Supplies",
    "לא מוכרת": "Unknown",
    "לא מוכר": "Unknown",
    "בע\"מ": "Ltd",
    "בעמ": "Ltd",
}


MEANINGFUL_PATTERNS = [
    (re.compile(r"^מעדנית\s+(?P<name>.+)$"), "{name} Delicatessen"),
    (re.compile(r"^מאפיית\s+(?P<name>.+)$"), "{name} Bakery"),
    (re.compile(r"^חנות\s+(?P<name>.+)$"), "{name} Store"),
    (re.compile(r"^מועדון כדורגל\s+(?P<name>.+)$"), "{name} Football Club"),
    (re.compile(r"^מרכז חינוך\s+(?P<name>.+?)(?:\s+שכ\"?ל)?$"), "{name} Education Center"),
]


HEBREW_TRANSLITERATION = {
    "א": "",
    "ב": "b",
    "ג": "g",
    "ד": "d",
    "ה": "h",
    "ו": "v",
    "ז": "z",
    "ח": "h",
    "ט": "t",
    "י": "y",
    "כ": "k",
    "ך": "k",
    "ל": "l",
    "מ": "m",
    "ם": "m",
    "נ": "n",
    "ן": "n",
    "ס": "s",
    "ע": "",
    "פ": "p",
    "ף": "p",
    "צ": "tz",
    "ץ": "tz",
    "ק": "k",
    "ר": "r",
    "ש": "sh",
    "ת": "t",
}


NAME_TRANSLITERATIONS = {
    "אברהמי": "Avrahami",
    "אושר": "Osher",
    "עד": "Ad",
    "חיפה": "Haifa",
    "אריאל": "Ariel",
    "כרמל": "Carmel",
    "ליאו": "Leo",
    "בק": "Baeck",
    "ילנה": "Yelena",
    "לוריא": "Luria",
    "יוסי": "Yossi",
    "חוצות": "Hutsot",
    "צפון": "North",
    "צמרת": "Top",
    "אפקה": "Afeka",
    "אי": "E",
}


def has_hebrew(text: object) -> bool:
    return bool(HEBREW_RE.search(str(text or "")))


def normalize_display_text(text: object) -> str:
    return " ".join(str(text or "").strip().split())


def canonical_display_text(text: object) -> str:
    return normalize_display_text(text).translate(QUOTE_TRANSLATION)


def transliterate_hebrew_word(word: str) -> str:
    if word in NAME_TRANSLITERATIONS:
        return NAME_TRANSLITERATIONS[word]

    result = []
    for char in word:
        if char in HEBREW_TRANSLITERATION:
            result.append(HEBREW_TRANSLITERATION[char])
        elif char.isascii():
            result.append(char)
        elif char in {"'", "\"", "-", "/", ".", "&"}:
            result.append(char)

    transliterated = "".join(result).strip("-'\" ")
    return transliterated.capitalize() if transliterated else word


def transliterate_hebrew(text: str) -> str:
    words = re.split(r"(\s+)", text)
    return "".join(
        part if part.isspace() else transliterate_hebrew_word(part)
        for part in words
    ).strip()


def meaningful_hebrew_display(text: str) -> str:
    for pattern, template in MEANINGFUL_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = transliterate_hebrew(match.group("name"))
        return template.format(name=name)
    return ""


def alias_lookup(text: str) -> str:
    canonical = canonical_display_text(text)
    aliases = {
        canonical_display_text(key): value
        for key, value in KNOWN_MERCHANT_ALIASES.items()
    }
    return aliases.get(canonical, "")


def translate_known_phrases(text: str) -> str:
    translated = canonical_display_text(text)
    for phrase, replacement in sorted(
        KNOWN_PHRASE_TRANSLATIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        translated = translated.replace(phrase, replacement)
    for phrase, replacement in sorted(
        KNOWN_BRANDS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        translated = translated.replace(phrase, replacement)
    return translated


def display_token(token: str) -> str:
    if not token:
        return ""

    canonical = canonical_display_text(token)
    if canonical in KNOWN_BRANDS:
        return KNOWN_BRANDS[canonical]
    if canonical in COMMON_WORD_TRANSLATIONS:
        return COMMON_WORD_TRANSLATIONS[canonical]
    if has_hebrew(canonical):
        return transliterate_hebrew(canonical)
    return canonical


def collapse_display_parts(parts: list[str]) -> str:
    text = " ".join(part for part in parts if part).strip()
    text = re.sub(r"\s+([,./])\s+", r"\1 ", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenized_display(text: str) -> str:
    phrase_translated = translate_known_phrases(text)
    parts = []
    for token in re.split(r"(\s+|-|/|,)", phrase_translated):
        if not token or token.isspace():
            continue
        if token in {"-", "/", ","}:
            parts.append(token)
            continue
        parts.append(display_token(token))
    return collapse_display_parts(parts)


def merchant_display_label(merchant: object) -> str:
    text = normalize_display_text(merchant)
    if not text:
        return ""

    alias = alias_lookup(text)
    if alias:
        return alias

    meaningful = meaningful_hebrew_display(text)
    if meaningful:
        return meaningful

    if has_hebrew(text):
        return tokenized_display(text)

    return text


def format_merchant_display(merchant: object) -> str:
    text = normalize_display_text(merchant)
    if not text:
        return ""

    label = merchant_display_label(text)
    if not label or label == text:
        return text

    return f"{text} [{label}]"
