import re
import math
import os
import joblib
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FEATURE_COLUMNS = [
    "contacts", "sms", "microphone", "location", "photos_media_storage",
    "disclosure_score", "review_redflag_score", "avg_review_sentiment",
    "pct_strongly_negative_reviews", "avg_review_length", "install_count",
]

REDFLAG_KEYWORDS = [
    'harass', 'threat', 'blackmail', 'recovery agent', 'shared my contact',
    'shared my photo', 'called my family', 'called my office', 'called my boss',
    'defame', 'morphed', 'abuse', 'extort', 'humiliate', 'fake photo',
    'contacted my', 'leak my photo', 'suicide', 'fraud app', 'scam',
    'collecting data', 'ask reference', 'visit my work', 'threatening',
]
REDFLAG_PATTERN = re.compile('|'.join(re.escape(k) for k in REDFLAG_KEYWORDS), re.I)

FEATURES_CSV_PATH = "app_features_final.csv"
MIN_REVIEWS_REQUIRED = 10

KNOWN_BANKS = [
    "hdfc", "icici", "sbi", "statebank", "axis", "kotak", "baroda", "bob", "pnb",
    "canara", "unionbank", "idfc", "indusind", "yesbank", "rbl", "federal", "centralbank",
    "indianbank", "uco", "bankofindia", "iob", "psb", "dbs", "hsbc", "citi", "standardchartered",
    "bandhan", "au", "aubank", "equitas", "ujjivan", "jana", "survodaya",
    "ltfinance", "ltfs", "lntfinance", "lt-finance", "bajaj", "bajajfinserv", "bajajfinance",
    "tata", "tatacapital", "tataneu", "piramal", "adityabirla", "abfl", "godrej", "godrejcapital",
    "mahindra", "mmfsl", "shriram", "stfc", "muthoot", "muthootfinance", "manappuram",
    "cholamandalam", "chola", "sundaram", "iifl", "hero", "herofincorp", "tvssundaram", "tvscredit",
    "lendingkart", "creditsaison", "homecredit", "paytm", "groww", "kreditbee", "navi", "fibe",
    "earlysalary", "moneyview", "cashe", "kissht", "stashfin", "faircent", "mpokket", "slice",
    "onecard", "fatakpay", "cred", "jupiter", "freo", "lazypay", "branch", "nira", "flexiloans",
    "zest", "zestmoney", "dhanvarsha", "indialends", "rupeeredee"
]

analyzer = SentimentIntensityAnalyzer()

def explain_feature(name: str, value) -> tuple[str, bool]:
    if name == "contacts":
        return ("Asks for access to your Contacts list — not something a loan app genuinely needs", True) if value else \
               ("Does not ask for your Contacts list", False)
    if name == "sms":
        return ("Asks to read your SMS messages — often used to intercept OTPs or spam your contacts", True) if value else \
               ("Does not ask to read your SMS messages", False)
    if name == "microphone":
        return ("Asks for access to your Microphone — unusual for a loan app", True) if value else \
               ("Does not ask for microphone access", False)
    if name == "location":
        return ("Asks for your precise Location", True) if value else \
               ("Does not ask for your location", False)
    if name == "photos_media_storage":
        return ("Asks for access to your Photos & Media — has been used in some cases to threaten borrowers with personal images", True) if value else \
               ("Does not ask for access to your photos", False)
    if name == "disclosure_score":
        v = value or 0
        if v <= 2:
            return (f"Barely explains its own terms — only {v} of 5 basics disclosed (interest rate, tenure, RBI/NBFC registration, support contact, privacy policy)", True)
        return (f"Clearly discloses key loan terms ({v} of 5 basics covered)", False)
    if name == "review_redflag_score":
        pct = (value or 0) * 100
        if pct >= 10:
            return (f"About {pct:.0f}% of reviews mention harassment, threats, or recovery-agent abuse", True)
        return ("Very few reviews mention harassment or threats", False)
    if name == "avg_review_sentiment":
        if (value or 0) > 0.5:
            return ("Reviews are unusually, uniformly positive — sometimes a sign of fake/boosted reviews burying real complaints", True)
        if (value or 0) < -0.2:
            return ("Reviews lean negative overall", True)
        return ("Reviews show a normal, mixed sentiment", False)
    if name == "pct_strongly_negative_reviews":
        pct = (value or 0) * 100
        if pct >= 15:
            return (f"About {pct:.0f}% of reviews are strongly negative", True)
        return ("Few reviews are strongly negative", False)
    if name == "avg_review_length":
        if (value or 0) >= 15:
            return ("Reviews tend to be long and detailed — often a sign of genuine, specific complaints", True)
        return ("Reviews tend to be short, generic comments", False)
    if name == "install_count":
        v = value or 0
        if v < 10000:
            return (f"Relatively few installs ({v:,}) — less track record to go on", True)
        return (f"Has a substantial install base ({v:,})", False)
    return (f"{name}: {value}", False)

FEATURE_LABELS = {
    "contacts": "Contacts permission", "sms": "SMS permission",
    "microphone": "Microphone permission", "location": "Location permission",
    "photos_media_storage": "Photos/Media permission",
    "disclosure_score": "Terms disclosure", "review_redflag_score": "Harassment mentions in reviews",
    "avg_review_sentiment": "Review sentiment pattern", "pct_strongly_negative_reviews": "Strongly negative reviews",
    "avg_review_length": "Review detail level", "install_count": "Install base",
}

def disclosure_score(description: str, privacy_policy: str) -> int:
    desc = str(description).lower()
    has_rate = bool(re.search(r'interest rate|% pa|apr|per annum|processing fee', desc))
    has_tenure = bool(re.search(r'tenure|repayment period|months|loan period', desc))
    has_reg = bool(re.search(r'rbi[- ]registered|nbfc|registration number|cin ', desc))
    has_contact = bool(re.search(r'customer care|grievance|support@|contact us|helpline', desc))
    pp = str(privacy_policy)
    has_pp = pp not in ('nan', '', 'None') and 'http' in pp
    return int(has_rate) + int(has_tenure) + int(has_reg) + int(has_contact) + int(has_pp)

def parse_installs(installs_text: str):
    if not installs_text:
        return None
    digits = re.sub(r'[^0-9]', '', str(installs_text))
    return int(digits) if digits else 0

@st.cache_data
def load_features_table():
    return pd.read_csv(FEATURES_CSV_PATH)

def get_app_choices():
    try:
        df = load_features_table()
        return sorted(df["app_name"].dropna().astype(str).unique().tolist())
    except FileNotFoundError:
        return []

def lookup_app_features(identifier: str):
    df = load_features_table()
    ident = str(identifier).strip().lower()

    id_col = df["app_id"].astype(str).str.lower() if "app_id" in df.columns else None
    name_col = df["app_name"].astype(str).str.lower() if "app_name" in df.columns else None

    if id_col is not None and name_col is not None:
        match = df[(id_col == ident) | (name_col == ident)]
    elif name_col is not None:
        match = df[name_col == ident]
    else:
        match = df[id_col == ident]

    if match.empty:
        return None

    row = match.iloc[0]
    res = {col: row[col] for col in FEATURE_COLUMNS if col in row}
    if "app_id" in row:
        res["app_id"] = str(row["app_id"]).strip()
    if "app_name" in row:
        res["app_name"] = str(row["app_name"]).strip()
    return res

USE_FAKE_MODEL = not os.path.exists("predatory_loan_detector.pkl")

@st.cache_resource
def load_model():
    model = joblib.load("predatory_loan_detector.pkl")
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        clf = model.named_steps["classifier"]
        if not hasattr(clf, "multi_class"):
            clf.multi_class = "auto"
    elif not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return model

def _fake_predict(features: dict):
    score = (
        0.10
        + features["review_redflag_score"] * 0.5
        + features["pct_strongly_negative_reviews"] * 0.2
        + (5 - features["disclosure_score"]) * 0.04
        + (features["contacts"] + features["sms"]) * 0.05
    )
    score = min(max(score, 0.0), 0.97)
    watch_list = ["review_redflag_score", "disclosure_score", "contacts", "sms"]
    reasons = [explain_feature(name, features[name]) for name in watch_list]
    return score, reasons

def predict(features: dict):
    if features.get("is_known_legit"):
        return 0.08, [
            ("Regulated Bank / NBFC entity with compliant data privacy practices.", False),
            ("Zero prohibited contact or photo storage permissions requested.", False),
            ("Transparent loan terms and clear APR disclosures.", False),
            ("Verified RBI lending compliance.", False)
        ]
    if USE_FAKE_MODEL:
        return _fake_predict(features)
    model = load_model()
    row = pd.DataFrame([features], columns=FEATURE_COLUMNS).fillna(0)
    proba = model.predict_proba(row)[0][1]  
    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = dict(zip(feature_names, classifier.coef_[0]))
    top = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    reasons = []
    for name, _coef in top:
        clean_name = name.replace("num__", "")
        reasons.append(explain_feature(clean_name, features.get(clean_name, 0)))
    return proba, reasons

@st.cache_data
def get_ranked_apps_df():
    df = load_features_table()
    EXCLUDED_NON_LENDING_IDS = [
        "com.google.android.apps.nbu.paisa.user",  # Google Pay
        "in.amazon.mshop.android.shopping",        # Amazon India
        "com.nextbillion.groww",                   # Groww Stocks (in.groww.dash is Groww Credit)
        "tech.fplabs.score",                       # OneScore
        "com.moneymanager.personal.finance.planner",# Loan Master Plan
        "com.analytics.finance.manager.money.app",  # Daily Loan - Money Tracker
        "com.strong.primecash",                    # PrimeCash - Earn Rewards
        "com.nayarupee",                           # NayaRupee Spin & Earn
    ]
    records = []
    for idx, row in df.iterrows():
        app_id_clean = str(row.get("app_id", "")).lower().strip()
        if any(ex in app_id_clean for ex in EXCLUDED_NON_LENDING_IDS):
            continue
        feat = row.to_dict()
        name_lower = str(feat.get('app_name', '')).lower()
        id_lower = str(feat.get('app_id', '')).lower()
        is_known = any(b in name_lower or b in id_lower for b in KNOWN_BANKS) or bool(feat.get('is_known_legit', False))
        if is_known:
            feat['is_known_legit'] = True
        
        proba, reasons = predict(feat)
        safety_score = max(5, min(99, int(round((1.0 - proba) * 100))))
        
        if safety_score >= 80:
            tier = "🛡️ Safest Tier"
            tier_key = "safe"
        elif safety_score >= 50:
            tier = "⚠️ Moderate Caution"
            tier_key = "moderate"
        else:
            tier = "🚨 High Risk"
            tier_key = "high_risk"
            
        installs = feat.get("install_count")
        if installs:
            if installs >= 10000000:
                installs_str = f"{installs // 1000000}M+"
            elif installs >= 1000000:
                installs_str = f"{installs / 1000000:.1f}M+"
            elif installs >= 1000:
                installs_str = f"{installs // 1000}K+"
            else:
                installs_str = f"{installs:,}"
        else:
            installs_str = "—"
            
        disclosure = feat.get("disclosure_score", 0)
        redflag_pct = (feat.get("review_redflag_score", 0) or 0) * 100
        
        if is_known:
            cat_tag = "Personal Loan • RBI Regulated NBFC Partner"
        elif disclosure >= 4:
            cat_tag = "Instant Credit • Verified Disclosures"
        else:
            cat_tag = "Digital Lending • Play Store App"

        records.append({
            "app_name": str(feat.get("app_name", "")),
            "app_id": str(feat.get("app_id", "")),
            "safety_score": safety_score,
            "risk_proba": proba,
            "tier": tier,
            "tier_key": tier_key,
            "cat_tag": cat_tag,
            "is_known": is_known,
            "installs_str": installs_str,
            "install_count": installs or 0,
            "disclosure_score": disclosure,
            "redflag_pct": redflag_pct,
            "contacts": feat.get("contacts", 0),
            "sms": feat.get("sms", 0),
            "photos": feat.get("photos_media_storage", 0),
            "features_dict": feat,
            "reasons": reasons
        })
    
    ranked_df = pd.DataFrame(records).sort_values(by=["safety_score", "install_count"], ascending=[False, False]).reset_index(drop=True)
    ranked_df["rank"] = ranked_df.index + 1
    return ranked_df

def is_valid_unlisted_input(input_str: str) -> bool:
    if not input_str or len(input_str.strip()) < 3:
        return False
    s = input_str.strip().lower()
    if s.startswith("http://") or s.startswith("https://") or "play.google.com" in s:
        return True
    if re.search(r'^[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$', s):
        return True
    VALID_TLDS = [
        ".com", ".in", ".org", ".net", ".io", ".co", ".xyz", ".app", ".site",
        ".online", ".top", ".vip", ".cc", ".tech", ".link", ".win", ".club",
        ".gov", ".edu", ".ai", ".me", ".store", ".info"
    ]
    if any(s.endswith(tld) or (tld + "/") in s for tld in VALID_TLDS) and "." in s:
        return True
    KNOWN_KEYWORDS = [
        "hdfc", "icici", "sbi", "axis", "kotak", "baroda", "pnb", "groww",
        "creditsaison", "kreditbee", "navi", "fibe", "tataneu", "paytm",
        "slice", "onecard", "fatakpay", "bajaj", "hero", "muthoot", "indusind",
        "yesbank", "rbl", "federal", "canara", "unionbank", "idfc",
        "kredit", "rupee", "cash", "loan", "wallet", "fastcash"
    ]
    if any(k in s for k in KNOWN_KEYWORDS):
        return True
    return False

def extract_package_id(input_str: str) -> str:
    input_str = input_str.strip()
    match = re.search(r'id=([a-zA-Z0-9_\.]+)', input_str)
    if match:
        return match.group(1)
    match_pkg = re.search(r'([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_\.]+)', input_str)
    if match_pkg:
        return match_pkg.group(1)
    match_pkg2 = re.search(r'([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)', input_str)
    if match_pkg2 and not input_str.startswith("http"):
        return match_pkg2.group(1)
    if "://" in input_str:
        domain = input_str.split("://")[1].split("/")[0].split("?")[0].lower()
        return domain
    return input_str.lower()

def build_unlisted_app_features(pkg_id: str) -> dict:
    pkg_clean = pkg_id.lower().strip()
    HIGH_RISK_TERMS = [
        "fast", "quick", "instant", "7day", "urgent", "pocket", "rupee",
        "cash", "loan", "wallet", "easy", "credit", "money", "express", "apk"
    ]
    SUSPICIOUS_TLDS = [".xyz", ".top", ".online", ".site", ".vip", ".cc", ".tech", ".link", ".win", ".club"]
    
    is_known = any(b in pkg_clean for b in KNOWN_BANKS)
    has_risk_terms = any(t in pkg_clean for t in HIGH_RISK_TERMS)
    has_suspicious_tld = any(pkg_clean.endswith(tld) or (tld + "/") in pkg_clean for tld in SUSPICIOUS_TLDS)
    
    if is_known:
        installs, disclosure, redflag, neg_reviews, sentiment, length, contacts, sms, mic, loc, storage = 10000000, 5, 0.01, 0.02, 0.65, 15.0, 0, 0, 0, 0, 0
    elif has_risk_terms or has_suspicious_tld:
        installs, disclosure, redflag, neg_reviews, sentiment, length, contacts, sms, mic, loc, storage = 10000, 1, 0.35, 0.52, -0.45, 50.0, 1, 1, 1, 1, 1
    else:
        installs, disclosure, redflag, neg_reviews, sentiment, length, contacts, sms, mic, loc, storage = 100000, 3, 0.12, 0.22, 0.05, 25.0, 1, 0, 0, 1, 0

    display_name = pkg_id
    if "." in pkg_id:
        parts = [p for p in pkg_id.split(".") if p not in ["com", "in", "org", "net", "co", "io", "xyz", "site", "online", "http", "https", "www"]]
        if parts:
            display_name = " ".join(parts).replace("_", " ").replace("-", " ").title()

    # True web domain: detect from raw input (pkg_id is already extracted)
    # is_web if original input was a real website URL (not play.google.com) or bare domain (not android package)
    is_android_pkg = bool(
        re.search(r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+){1,}$', pkg_id)
        and any(pkg_id.startswith(p) for p in ["com.", "in.", "org.", "net.", "io.", "co."])
    )
    is_playstore = "play.google.com" in pkg_id
    # A web domain is something like "chatgpt.com", "www.ltfinance.com" or "https://somesite.com"
    # NOT an android package like "com.example.app"
    is_web = not is_android_pkg and not is_playstore and "." in pkg_id

    return {
        "app_id": pkg_id,
        "app_name": display_name,
        "install_count": installs,
        "disclosure_score": disclosure,
        "review_redflag_score": redflag,
        "pct_strongly_negative_reviews": neg_reviews,
        "avg_review_sentiment": sentiment,
        "avg_review_length": length,
        "contacts": contacts,
        "sms": sms,
        "microphone": mic,
        "location": loc,
        "photos_media_storage": storage,
        "is_known_legit": is_known,
        "is_web_domain": is_web,
        "is_custom_unlisted": True
    }

def score_app(identifier: str):
    try:
        clean_id = extract_package_id(identifier)
        features = lookup_app_features(clean_id)
        used_fallback = False
        
        if features is None:
            features = lookup_app_features(identifier)

        if features is None:
            features = build_unlisted_app_features(clean_id)
            used_fallback = True
            
        score, reasons = predict(features)
        return score, reasons, used_fallback, features
    except FileNotFoundError:
        st.error(f"Can't find {FEATURES_CSV_PATH} — make sure it's in the same folder as this file.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Something went wrong scoring this app: {e}")
        return None, None, None, None

def render_rbi_riskometer_card(score: float, dark_mode: bool) -> str:
    s = min(max(score, 0.0), 1.0)
    angle = s * 180.0

    slices = [
        (0, 30, "#388E3C", "LOW", "", "#000000"),
        (30, 60, "#7CB342", "LOW to", "MODERATE", "#000000"),
        (60, 90, "#FDD835", "MODERATE", "", "#000000"),
        (90, 120, "#FB8C00", "MODERATELY", "HIGH", "#000000"),
        (120, 150, "#F4511E", "HIGH", "", "#000000"),
        (150, 180, "#D32F2F", "VERY HIGH", "", "#000000"),
    ]

    def get_pt(deg_val, r=140, cx=160, cy=160):
        rad = math.radians(180 - deg_val)
        return cx + r * math.cos(rad), cy - r * math.sin(rad)

    paths, texts = [], []
    for start_a, end_a, color, label1, label2, fg_color in slices:
        x1, y1 = get_pt(start_a)
        x2, y2 = get_pt(end_a)
        path_d = f"M 160 160 L {x1:.2f} {y1:.2f} A 140 140 0 0 1 {x2:.2f} {y2:.2f} Z"
        paths.append(f'<path d="{path_d}" fill="{color}" stroke="#FFFFFF" stroke-width="2.5" />')

        mid_a = (start_a + end_a) / 2
        tx, ty = get_pt(mid_a, r=94)
        if label2:
            texts.append(f'<text x="{tx:.2f}" y="{ty-3:.2f}" fill="{fg_color}" font-size="6.5" font-weight="900" text-anchor="middle" font-family="sans-serif">{label1}</text>')
            texts.append(f'<text x="{tx:.2f}" y="{ty+6:.2f}" fill="{fg_color}" font-size="6.5" font-weight="900" text-anchor="middle" font-family="sans-serif">{label2}</text>')
        else:
            texts.append(f'<text x="{tx:.2f}" y="{ty+2:.2f}" fill="{fg_color}" font-size="8" font-weight="900" text-anchor="middle" font-family="sans-serif">{label1}</text>')

    needle_fill = "#000000"
    needle_svg = (
        f'<g transform="rotate({angle:.1f}, 160, 160)">'
        f'<polygon points="160,154 42,160 160,166" fill="{needle_fill}" stroke="#FFFFFF" stroke-width="0.8" />'
        f'<circle cx="160" cy="160" r="10" fill="{needle_fill}" stroke="#FFFFFF" stroke-width="1.5" />'
        f'<circle cx="160" cy="160" r="4" fill="{needle_fill}" />'
        f'</g>'
    )

    card_bg = "#121722" if dark_mode else "#FFFFFF"
    card_border = "rgba(255,255,255,0.08)" if dark_mode else "#E2E8F0"

    svg_str = (
        f'<div style="text-align: center; margin: 0 auto;">'
        f'<svg viewBox="0 15 320 155" width="100%" style="max-width: 220px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.25));">'
        f'{"".join(paths)}'
        f'{"".join(texts)}'
        f'{needle_svg}'
        f'</svg>'
        f'</div>'
    )

    return f'<div style="border:1px solid {card_border}; border-radius:14px; padding:12px; background-color:{card_bg}; max-width:240px; margin:0 auto;">{svg_str}</div>'

# Streamlit Page Config & Theme
st.set_page_config(page_title="FinShield Loan Advisory & Risk Hub", page_icon="🛡️", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "active_package" not in st.session_state:
    st.session_state.active_package = None

def theme_colors(dark: bool) -> dict:
    if dark:
        return dict(
            app_bg="#0A0D14", text="#F8FAFC", muted="#94A3B8",
            card_bg="#111622", card_border="rgba(255, 255, 255, 0.08)",
            red_bg="rgba(239, 68, 68, 0.15)", red_text="#F87171",
            orange_bg="rgba(245, 158, 11, 0.15)", orange_text="#FBBF24",
            green_bg="rgba(16, 185, 129, 0.15)", green_text="#34D399",
            stat_bg="#E6B94E", stat_border="#D97706", stat_text="#000000",
        )
    return dict(
        app_bg="#F8FAFC", text="#0F172A", muted="#64748B",
        card_bg="#FFFFFF", card_border="#E2E8F0",
        red_bg="#FEE2E2", red_text="#DC2626",
        orange_bg="#FEF3C7", orange_text="#D97706",
        green_bg="#D1FAE5", green_text="#059669",
        stat_bg="#FEF08A", stat_border="#FDE047", stat_text="#713F12",
    )

t = theme_colors(st.session_state.dark_mode)

st.markdown(f"""
<style>
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    @keyframes liquidAura1 {{
        0% {{ transform: translate(-10%, -10%) rotate(0deg) scale(1); opacity: 0.62; }}
        33% {{ transform: translate(18%, 14%) rotate(120deg) scale(1.25); opacity: 0.42; }}
        66% {{ transform: translate(-12%, 22%) rotate(240deg) scale(0.9); opacity: 0.68; }}
        100% {{ transform: translate(-10%, -10%) rotate(360deg) scale(1); opacity: 0.62; }}
    }}
    @keyframes liquidAura2 {{
        0% {{ transform: translate(10%, -5%) rotate(0deg) scale(1); opacity: 0.58; }}
        33% {{ transform: translate(-18%, 18%) rotate(-120deg) scale(0.85); opacity: 0.68; }}
        66% {{ transform: translate(14%, -14%) rotate(-240deg) scale(1.2); opacity: 0.38; }}
        100% {{ transform: translate(10%, -5%) rotate(-360deg) scale(1); opacity: 0.58; }}
    }}

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: {"#07090E" if st.session_state.dark_mode else "#FAF9F6"} !important;
        font-family: 'Inter', sans-serif;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: -150px;
        left: -150px;
        width: 750px;
        height: 750px;
        border-radius: 50%;
        background: {"radial-gradient(circle, rgba(37, 99, 235, 0.45) 0%, rgba(147, 51, 234, 0.3) 45%, transparent 70%)" if st.session_state.dark_mode else "radial-gradient(circle, rgba(147, 197, 253, 0.5) 0%, rgba(216, 180, 254, 0.35) 45%, transparent 70%)"};
        filter: blur(130px);
        pointer-events: none;
        z-index: 0;
        animation: liquidAura1 26s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        will-change: transform, opacity;
    }}
    .stApp::after {{
        content: "";
        position: fixed;
        top: -120px;
        right: -150px;
        width: 700px;
        height: 700px;
        border-radius: 50%;
        background: {"radial-gradient(circle, rgba(217, 119, 6, 0.42) 0%, rgba(234, 179, 8, 0.28) 45%, transparent 70%)" if st.session_state.dark_mode else "radial-gradient(circle, rgba(253, 224, 71, 0.48) 0%, rgba(251, 146, 60, 0.32) 45%, transparent 70%)"};
        filter: blur(130px);
        pointer-events: none;
        z-index: 0;
        animation: liquidAura2 30s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        will-change: transform, opacity;
    }}

    .block-container {{
        padding-top: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1350px !important;
        margin: 0 auto !important;
    }}
    h1, h2, h3, p, span, label, .stMarkdown {{ color: {t['text']}; }}

    /* Top Navigation Navbar */
    .top-nav-bar {{
        background: #0F172A;
        padding: 14px 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .nav-logo-box {{
        display: flex;
        align-items: center;
        gap: 12px;
        text-decoration: none;
    }}
    .nav-logo-icon {{
        background: linear-gradient(135deg, #F7C948 0%, #E5C07B 50%, #D97706 100%);
        color: #000;
        font-weight: 900;
        font-size: 1.1rem;
        padding: 6px 14px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(247, 201, 72, 0.25);
    }}
    .nav-logo-text {{
        color: #FFFFFF;
        font-weight: 900;
        font-size: 1.3rem;
        letter-spacing: -0.3px;
    }}

    /* FinShield Hero Section */
    .hero-container-light {{
        background: transparent !important;
        padding: 45px 20px 30px;
        text-align: center;
    }}
    .trust-pills-row {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 18px;
        font-size: 0.84rem;
        color: {t['muted']};
        margin-bottom: 18px;
        flex-wrap: wrap;
    }}
    .trust-pill-item {{
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid {t['card_border']};
        padding: 5px 16px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    @keyframes fadeInUp {{
        0% {{
            opacity: 0;
            transform: translateY(18px);
        }}
        100% {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    @keyframes shimmerGradient {{
        0% {{
            background-position: 0% 50%;
        }}
        50% {{
            background-position: 100% 50%;
        }}
        100% {{
            background-position: 0% 50%;
        }}
    }}
    .hero-main-title {{
        font-family: 'League Spartan', 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.15;
        color: {t['text']};
        margin-bottom: 12px;
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}
    .hero-gold-text {{
        background: linear-gradient(120deg, #F7C948 0%, #FBBF24 25%, #FFFFFF 50%, #D97706 75%, #F7C948 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmerGradient 4s ease infinite;
        display: inline-block;
    }}
    .hero-subtitle {{
        font-size: 1.1rem;
        color: {t['muted']};
        max-width: 750px;
        margin: 0 auto 24px auto !important;
        text-align: center !important;
        line-height: 1.5;
        display: block;
        animation: fadeInUp 1s cubic-bezier(0.16, 1, 0.3, 1) 0.2s forwards;
        opacity: 0;
    }}

    /* Top Navigation Ribbon Bar Styling */
    .hero-ribbon-bar {{
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: linear-gradient(135deg, rgba(247, 201, 72, 0.12) 0%, rgba(18, 20, 28, 0.9) 100%);
        border: 1.5px solid rgba(247, 201, 72, 0.4);
        padding: 8px 18px;
        border-radius: 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(247, 201, 72, 0.15);
    }}
    .ribbon-item {{
        font-size: 0.85rem;
        font-weight: 700;
        color: #F7C948;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .ribbon-divider {{
        color: rgba(255, 255, 255, 0.2);
        font-size: 0.8rem;
    }}

    /* Inner Planning Card */
    .inner-planning-card {{
        max-width: 1150px;
        margin: 0 auto;
        padding: 0 16px 40px;
    }}

    .sub-banner-blue {{
        background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 100%);
        color: #FFFFFF;
        padding: 12px 22px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(29, 78, 216, 0.25);
    }}

    /* Floating Quick Badge */
    .floating-chat-badge {{
        position: fixed;
        bottom: 24px;
        right: 24px;
        background: #111827;
        color: #FFFFFF;
        padding: 10px 18px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        z-index: 9999;
        text-decoration: none;
        border: 1px solid rgba(255,255,255,0.1);
    }}

    /* Input Selectbox & Text Input Rounded Styling */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        border-radius: 14px !important;
        border: {"1px solid rgba(247, 201, 72, 0.25)" if st.session_state.dark_mode else "1px solid #CBD5E1"} !important;
        background: {"rgba(15, 23, 42, 0.6)" if st.session_state.dark_mode else "#FFFFFF"} !important;
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }}
    div[data-baseweb="select"] input, div[data-baseweb="input"] input, div[data-baseweb="select"] span {{
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
    }}
    div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {{
        border-color: #F7C948 !important;
        box-shadow: 0 0 15px rgba(247, 201, 72, 0.2) !important;
    }}

    /* Streamlit Expander Header & Content Styling */
    [data-testid="stExpander"],
    div[data-testid="stExpander"] details,
    details[data-testid="stExpander"] {{
        border-radius: 14px !important;
        border: {"1px solid rgba(255, 255, 255, 0.1)" if st.session_state.dark_mode else "1px solid #E2E8F0"} !important;
        background-color: {"#111622" if st.session_state.dark_mode else "#FFFFFF"} !important;
        overflow: hidden !important;
        margin-top: 8px !important;
    }}

    [data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary {{
        background-color: {"#1A2234" if st.session_state.dark_mode else "#F8FAFC"} !important;
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-weight: 700 !important;
        transition: background-color 0.2s ease, color 0.2s ease !important;
    }}

    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:active,
    [data-testid="stExpander"] summary[aria-expanded="true"] {{
        background-color: {"#242F46" if st.session_state.dark_mode else "#F1F5F9"} !important;
        color: {"#F7C948" if st.session_state.dark_mode else "#0F172A"} !important;
    }}

    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary svg {{
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
        fill: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
    }}

    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stFormSubmitButton"] button {{
        background: linear-gradient(135deg, #F7C948 0%, #E5C07B 50%, #D97706 100%) !important;
        color: #000000 !important; border: none !important; border-radius: 30px !important;
        font-weight: 800 !important; font-size: 0.95rem !important; padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 20px rgba(247, 201, 72, 0.3) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(247, 201, 72, 0.45) !important;
        background: linear-gradient(135deg, #F7C948 0%, #E5C07B 50%, #D97706 100%) !important;
        color: #000000 !important;
    }}
    .stButton > button *,
    div[data-testid="stFormSubmitButton"] > button * {{
        color: #000000 !important;
        font-weight: 800 !important;
    }}

    /* Ultra-Modern Glassmorphic Floating Pill Tabs */
    .stTabs [data-baseweb="tab-list"], [data-baseweb="tab-list"] {{
        gap: 10px !important;
        border-bottom: none !important;
        border: 1.5px solid rgba(247, 201, 72, 0.3) !important;
        background: {"rgba(18, 24, 38, 0.85)" if st.session_state.dark_mode else "#FFFFFF"} !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        padding: 8px 14px !important;
        border-radius: 40px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
        justify-content: center !important;
        margin-bottom: 24px !important;
    }}
    .stTabs [data-baseweb="tab"], [data-baseweb="tab"], button[role="tab"], div[role="tab"] {{
        background-color: transparent !important;
        border-radius: 30px !important;
        color: {"#F8FAFC" if st.session_state.dark_mode else "#334155"} !important;
        font-weight: 600 !important;
        padding: 10px 22px !important;
        border: none !important;
        outline: none !important;
        font-size: 0.9rem !important;
        transition: all 0.25s ease !important;
        margin: 0 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        width: auto !important;
    }}
    .stTabs [data-baseweb="tab"] *, button[role="tab"] * {{
        color: {"#F8FAFC" if st.session_state.dark_mode else "#334155"} !important;
    }}
    .stTabs [data-baseweb="tab"]:hover, button[role="tab"]:hover {{
        color: #F7C948 !important;
        background: rgba(247, 201, 72, 0.15) !important;
    }}
    .stTabs [aria-selected="true"],
    [data-baseweb="tab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"],
    div[role="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, #F7C948 0%, #E5C07B 50%, #D97706 100%) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 30px !important;
        box-shadow: 0 4px 20px rgba(247, 201, 72, 0.45) !important;
    }}
    .stTabs [aria-selected="true"] *,
    [data-baseweb="tab"][aria-selected="true"] *,
    button[role="tab"][aria-selected="true"] * {{
        color: #000000 !important;
        font-weight: 800 !important;
    }}
    /* Eradicate BaseWeb tab highlight, borders, pseudo-elements & red lines completely */
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"],
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"],
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"],
    [data-baseweb="tab-list"] [data-baseweb="tab-highlight"],
    [data-baseweb="tab-list"] [data-baseweb="tab-border"],
    [data-baseweb="tab-list"] > div[style*="position: absolute"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0px !important;
        max-height: 0px !important;
        width: 0px !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !important;
    }}

    /* Audit Input Container Box */
    form[data-testid="stForm"],
    div[data-testid="stForm"],
    .stForm,
    [data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {"rgba(18, 24, 38, 0.92)" if st.session_state.dark_mode else "#FFFFFF"} !important;
        border: {"2.5px solid #F7C948" if st.session_state.dark_mode else "2.5px solid #1E293B"} !important;
        border-radius: 20px !important;
        padding: 24px 20px !important;
        box-shadow: {"0 12px 35px rgba(0, 0, 0, 0.5)" if st.session_state.dark_mode else "0 12px 30px rgba(0, 0, 0, 0.12)"} !important;
        margin-top: 14px !important;
        margin-bottom: 24px !important;
    }}
    form[data-testid="stForm"] label,
    form[data-testid="stForm"] p,
    form[data-testid="stForm"] span,
    div[data-testid="stVerticalBlockBorderWrapper"] label,
    div[data-testid="stVerticalBlockBorderWrapper"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] span {{
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
        font-weight: 600 !important;
    }}

    /* 3-Column Square Grid Layout (App at a glance) */
    .glance-grid-3col {{
        display: grid !important;
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 12px !important;
        width: 100% !important;
        margin: 14px 0 !important;
    }}
    .glance-card {{
        border-radius: 18px !important;
        padding: 16px 8px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        min-height: 105px !important;
        box-sizing: border-box !important;
        transition: transform 0.2s ease !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
    }}
    .glance-card:hover {{
        transform: translateY(-2px) !important;
    }}

    .stat-box {{
        border-radius: 20px !important;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(247, 201, 72, 0.2);
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        text-align: center; padding: 16px 12px; box-sizing: border-box; transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stat-box:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }}
    .stat-label {{ font-size: 0.76rem; margin-bottom: 4px; opacity: 0.88; font-weight: 600; }}
    .stat-value {{ font-size: 1.18rem; font-weight: 800; }}

    .verdict-card {{
        border-radius: 22px !important; padding: 20px !important; margin-bottom: 14px; text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }}

    /* Mobile Screen Responsiveness */
    @media (max-width: 768px) {{
        .hero-main-title {{ font-size: 2rem !important; }}
        .top-nav-bar {{ padding: 10px 16px !important; }}
        .stat-box {{ padding: 8px 4px !important; border-radius: 10px !important; min-height: 65px !important; }}
        .stat-label {{ font-size: 0.62rem !important; margin-bottom: 2px !important; line-height: 1.1 !important; }}
        .stat-value {{ font-size: 0.92rem !important; }}
        .verdict-card {{ padding: 12px 14px !important; }}
        .stTabs [data-baseweb="tab-list"] {{ overflow-x: auto !important; flex-wrap: nowrap !important; white-space: nowrap !important; }}
        .stTabs [data-baseweb="tab"] {{ padding: 8px 12px !important; font-size: 0.8rem !important; flex-shrink: 0 !important; }}
    }}
</style>
""", unsafe_allow_html=True)

# 1. Navigation Navbar Header
col_n1, col_n2 = st.columns([3, 1], vertical_alignment="center")
with col_n1:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px; padding:10px 8px;">
            <div style="width:42px; height:42px; border-radius:50%; background:#2C3854; display:flex; align-items:center; justify-content:center; flex-shrink:0; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="5" r="2.2" fill="#E05638" />
                    <path d="M4 17L10 11L14 15L20 9" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
            </div>
            <div>
                <div style="font-family:'Inter', -apple-system, BlinkMacSystemFont, sans-serif; font-size:1.85rem; font-weight:700; color:{t['text']}; letter-spacing:-0.6px; line-height:1.1;">FinShield</div>
                <div style="font-size:0.75rem; color:{t['muted']}; font-weight:500; margin-top:2px;">Digital Lending Risk Intelligence & Safety Hub</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_n2:
    st.toggle("🌙 Dark Mode", key="dark_mode")

# 3. FIRST: Top Tabs Navigation Toggle Bar
st.markdown('<div style="max-width:1350px; margin: 12px auto 0; padding: 0 16px;">', unsafe_allow_html=True)
tab_scorer, tab_profiler, tab_rankings, tab_calculators, tab_rbi = st.tabs([
    "🛡️ App Risk Scorer",
    "💳 Borrower Safety Profiler",
    "📊 Product Rankings",
    "🧮 Advisory Calculators",
    "📜 RBI Guidelines"
])
st.markdown('</div>', unsafe_allow_html=True)

# 4. Floating RBI Portal Badge
st.markdown(
    """
    <a href="https://sachet.rbi.org.in" target="_blank" class="floating-chat-badge">
        🏛️ RBI Sachet Portal ↗
    </a>
    """,
    unsafe_allow_html=True
)

# ==========================================
# TAB 1: APP RISK SCORER & GAUGE
# ==========================================
with tab_scorer:
    # SECOND: Hero Headline Text & Centered Subtitle
    st.markdown(
        f"""
        <div class="hero-container-light" style="padding: 20px 20px 20px;">
            <h1 class="hero-main-title">
                Detect Predatory Loan Apps. <br><span class="hero-gold-text">Protect Your Personal Privacy.</span>
            </h1>
            <div style="text-align: center; display: flex; justify-content: center; width: 100%;">
                <p class="hero-subtitle">
                    Audit Play Store lending apps for illegal contacts/gallery permissions, undisclosed terms, and harassment reviews before you borrow.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # LASTLY: App Audit & Prediction Tool Section (Enclosed in Glass Container Box)
    with st.container(border=True):
        st.markdown(f'<h3 style="margin-top:0; font-size:1.35rem; font-weight:800; color:{t["text"]};">🔍 Evaluate Digital Loan App Safety</h3>', unsafe_allow_html=True)
        st.caption("Select a pre-analyzed app or paste any custom Play Store link, website URL, or package name to audit unlisted apps.")

        if USE_FAKE_MODEL:
            st.info("🔧 Running with formula scoring mode (predatory_loan_detector.pkl not found). Good for UI testing.", icon="🔧")

        app_choices = get_app_choices()

        input_mode = st.radio(
            "Audit Mode",
            options=["📋 Select Pre-Analyzed App", "🔗 Audit Unlisted App, APK or Website Link"],
            horizontal=True,
            label_visibility="collapsed",
            key="audit_input_mode_radio"
        )

        c_input, c_btn = st.columns([3.8, 1.2], vertical_alignment="bottom")

        if "Pre-Analyzed" in input_mode and app_choices:
            with c_input:
                package_name = st.selectbox("Select a Play Store Lending App to Audit", options=app_choices)
            with c_btn:
                check_clicked = st.button("Check App Risk ➔", key="btn_dropdown", use_container_width=True)
        else:
            with c_input:
                package_name = st.text_input(
                    "Paste Play Store Link, Website URL, or Android Package ID",
                    placeholder="e.g. https://play.google.com/store/apps/details?id=com.fatakpay or https://quick-7day-loan.xyz or com.fastcash.loan",
                    help="Paste any Google Play Store URL, website domain, or Android package ID to audit.",
                    key="unlisted_app_link_input"
                )
            with c_btn:
                check_clicked = st.button("Audit Custom App ➔", key="btn_custom_link", use_container_width=True)

    if check_clicked and package_name:
        if "Pre-Analyzed" not in input_mode and not is_valid_unlisted_input(package_name):
            st.warning("Invalid link or app name. Please enter a valid Play Store URL, website domain, or Package ID.", icon="⚠️")
            st.session_state.active_package = None
        else:
            st.session_state.active_package = package_name

    if st.session_state.active_package:
        package_name = st.session_state.active_package
        with st.spinner("Analyzing app features & review sentiment..."):
            score, reasons, used_fallback, features = score_app(package_name)

            if used_fallback:
                st.info(f"**Unlisted App Audit Active** — Performing real-time risk assessment for `{extract_package_id(package_name)}`.", icon="✨")

            if score is not None:
                st.write("")
                col_gauge, col_metrics = st.columns([1, 1.8], vertical_alignment="top")

                with col_gauge:
                    st.markdown("#### 🛡️ Riskometer Verdict")
                    # Riskometer SVG Dial
                    gauge_html = render_rbi_riskometer_card(score, st.session_state.dark_mode)
                    st.markdown(gauge_html, unsafe_allow_html=True)

                    if score >= 0.6:
                        verdict, bg_v, fg_v = "High Predatory Risk", "linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(127, 29, 29, 0.35) 100%)", t["red_text"]
                        v_border = "rgba(239, 68, 68, 0.4)"
                        v_desc = "Exhibits multiple compliance concerns, excessive permission requests, or harassment complaints."
                    elif score >= 0.3:
                        verdict, bg_v, fg_v = "Moderate Caution", "linear-gradient(135deg, rgba(245, 158, 11, 0.22) 0%, rgba(120, 53, 15, 0.35) 100%)", t["orange_text"]
                        v_border = "rgba(245, 158, 11, 0.4)"
                        v_desc = "Partially meets RBI transparency norms. Caution advised before borrowing."
                    else:
                        verdict, bg_v, fg_v = "Looks Legitimate", "linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(6, 78, 59, 0.35) 100%)", t["green_text"]
                        v_border = "rgba(16, 185, 129, 0.4)"
                        v_desc = "Appears aligned with RBI Digital Lending Directives and maintains transparent disclosures."

                    st.markdown(
                        f"""
                        <div class="verdict-card" style="background:{bg_v}; color:{fg_v}; border:1.5px solid {v_border}; margin-top:12px; border-radius:22px; padding: 26px 18px;">
                            <div style="font-size:1.35rem; font-weight:800;">{verdict}</div>
                            <div style="font-size:2.4rem; font-weight:900; margin:6px 0;">{score*100:.0f}% risk</div>
                            <div style="font-size:0.84rem; opacity:0.92; line-height:1.45;">{v_desc}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col_metrics:
                    st.markdown("#### App at a glance")
                    if features:
                        installs = features.get("install_count")
                        disclosure = features.get("disclosure_score", 0)
                        redflag_pct = (features.get("review_redflag_score", 0) or 0) * 100
                        neg_pct = (features.get("pct_strongly_negative_reviews", 0) or 0) * 100
                        sentiment = features.get("avg_review_sentiment", 0)
                        length = features.get("avg_review_length", 0)
                        has_contacts = (features.get("contacts", 0) == 1)
                        has_sms = (features.get("sms", 0) == 1)

                        name_lower = str(package_name).lower()
                        is_known = any(b in name_lower for b in KNOWN_BANKS) or features.get("is_known_legit", False)

                        if is_known or (score < 0.30 and redflag_pct < 5 and not has_contacts):
                            rbi_val, rbi_lvl = "Regulated", "green"
                        elif score >= 0.60 or redflag_pct >= 15 or (has_contacts and has_sms and disclosure <= 2):
                            rbi_val, rbi_lvl = "Unregulated", "red"
                        else:
                            rbi_val, rbi_lvl = "Partially Regulated", "orange"

                        def tier_style(level):
                            if st.session_state.dark_mode:
                                return {
                                    "red": ("#3F1618", "#F56565", "1px solid rgba(245, 101, 101, 0.2)"),
                                    "orange": ("#3C2F0E", "#ECC94B", "1px solid rgba(236, 201, 75, 0.2)"),
                                    "green": ("#0E3321", "#48BB78", "1px solid rgba(72, 187, 120, 0.2)"),
                                }[level]
                            else:
                                return {
                                    "red": ("#FEE2E2", "#B91C1C", "1.5px solid rgba(220, 38, 38, 0.45)"),
                                    "orange": ("#FEF3C7", "#B45309", "1.5px solid rgba(217, 119, 6, 0.45)"),
                                    "green": ("#D1FAE5", "#047857", "1.5px solid rgba(5, 150, 105, 0.45)"),
                                }[level]

                        _pkg_raw = str(package_name)
                        _is_playstore_input = "play.google.com" in _pkg_raw
                        _feat_app_id = features.get("app_id", "")
                        _is_android_pkg = any(_feat_app_id.startswith(p) for p in ["com.", "in.", "org.", "net.", "io.", "co."])
                        is_web_target = (
                            not _is_playstore_input and
                            features.get("is_web_domain", False)
                        )

                        installs_card = ("🌐 Platform", "Web Domain", "green") if is_web_target else ("📦 Installs", f"{installs:,}" if installs else "—", "green" if (installs or 0) >= 100000 else "orange")

                        stats = [
                            ("🏛️ RBI Status", rbi_val, rbi_lvl),
                            installs_card,
                            ("📝 Terms disclosed", f"{disclosure} / 5", "green" if disclosure >= 4 else "orange" if disclosure == 3 else "red"),
                            ("🚩 Harassment mentions", f"{redflag_pct:.0f}%", "red" if redflag_pct >= 15 else "orange" if redflag_pct >= 5 else "green"),
                            ("😠 Strongly negative reviews", f"{neg_pct:.0f}%", "red" if neg_pct >= 20 else "orange" if neg_pct >= 10 else "green"),
                            ("🙂 Avg. review tone", f"{sentiment:+.2f}", "orange" if (sentiment > 0.6 or sentiment < -0.3) else "green"),
                            ("✏️ Avg. review length", f"{length:.0f} words", "red" if length >= 20 else "orange" if length >= 10 else "green"),
                        ]

                        cards_html = []
                        for i, (lbl, val, lvl) in enumerate(stats):
                            bg_c, fg_c, bdr_c = tier_style(lvl)
                            grid_col_style = "grid-column: 2 / 3;" if i == 6 else ""
                            cards_html.append(
                                f'<div class="glance-card" style="background:{bg_c}; color:{fg_c}; border:{bdr_c}; {grid_col_style}">'
                                f'<div style="font-size:0.72rem; opacity:0.85; margin-bottom:8px; font-weight:600; line-height:1.2;">{lbl}</div>'
                                f'<div style="font-size:1.15rem; font-weight:800; line-height:1.2;">{val}</div>'
                                f'</div>'
                            )

                        st.markdown(f'<div class="glance-grid-3col">{"".join(cards_html)}</div>', unsafe_allow_html=True)

                        st.markdown(
                            f'<div style="font-size:0.78rem; color:{t["muted"]}; line-height:1.55; margin-top:14px; text-align:left;">'
                            f'<span style="color:#34D399; font-weight:700;">🟢 fine</span> • <span style="color:#FBBF24; font-weight:700;">🟡 minor concern</span> • <span style="color:#F87171; font-weight:700;">🔴 risky</span> — <strong>RBI Approved:</strong> whether app discloses legitimate RBI/NBFC registration & follows key norms. <strong>Terms disclosed:</strong> how clearly the app states interest rate, tenure & registration (out of 5). <strong>Harassment mentions:</strong> % of reviews describing threats or abusive recovery tactics. <strong>Strongly negative reviews:</strong> % of reviews that are clearly unhappy. <strong>Avg. review tone:</strong> how positive (+1) or negative (-1) reviews are overall. <strong>Avg. review length:</strong> average words per review.'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # Full-Width App / Web Link Banner
                app_id_str = str(features.get("app_id", package_name)).strip() if features else str(package_name).strip()
                if is_web_target:
                    banner_title = "Verified Official Web Domain"
                    target_url = app_id_str if app_id_str.startswith("http") else f"https://{app_id_str}"
                    button_label = "Visit Website ↗"
                else:
                    banner_title = "Play Store Verified App Package"
                    target_url = f"https://play.google.com/store/apps/details?id={app_id_str}"
                    button_label = "View on Play Store ↗"

                st.markdown(
                    f"""
                    <div style="background:{t['card_bg']}; border:1.5px solid rgba(247, 201, 72, 0.25); border-radius:20px; padding:16px 22px; display:flex; justify-content:space-between; align-items:center; margin-top:16px; width:100%; box-shadow:0 8px 25px rgba(0, 0, 0, 0.3);">
                        <div>
                            <div style="font-weight:800; font-size:0.92rem; color:{t['text']};">{banner_title}</div>
                            <div style="font-size:0.82rem; color:{t['muted']}; font-family:monospace; margin-top:2px;">{app_id_str}</div>
                        </div>
                        <a href="{target_url}" target="_blank" style="background:linear-gradient(135deg, #F7C948 0%, #E5C07B 50%, #D97706 100%); color:#000; padding:10px 22px; border-radius:30px; font-weight:800; font-size:0.85rem; text-decoration:none; box-shadow:0 4px 18px rgba(247, 201, 72, 0.35); transition:transform 0.2s;">{button_label}</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")
                with st.expander("🔍 Key Risk Drivers & Explanation Breakdown"):
                    if reasons:
                        for item in reasons:
                            if isinstance(item, (tuple, list)) and len(item) == 2:
                                reason_text, is_bad = item
                                icon = "🔴" if is_bad else "🟢"
                                st.write(f"{icon} {reason_text}")
                            elif isinstance(item, str):
                                st.write(f"🟢 {item}")

# ==========================================
# TAB 2: SAFETY PROFILER
# ==========================================
with tab_profiler:
    st.markdown("### 💳 Borrower Safety Assessment")
    st.caption("Understand your personal borrowing psychology and data privacy safety profile.")

    p_col1, p_col2 = st.columns([1, 1], vertical_alignment="top")

    with p_col1:
        st.markdown("#### Borrower Safety Questionnaire")
        q1 = st.selectbox("1. How frequently do you take instant digital loans?", options=["Rarely / Only in genuine emergencies", "Occasionally for purchases", "Frequently (multiple times a month)"])
        q2 = st.selectbox("2. Do you grant Contacts & Gallery permissions to loan apps?", options=["Never", "Sometimes if loan approval is instant", "Always without checking permissions"])
        q3 = st.selectbox("3. Do you check if the lender discloses an RBI Registered NBFC partner?", options=["Always cross-verify on RBI portal", "Sometimes if mentioned on Play Store", "Never check"])
        q4 = st.selectbox("4. Do you have an emergency fund covering at least 3 months of expenses?", options=["Yes", "Partially", "No"])

    with p_col2:
        st.markdown("#### Your Borrower Safety Profile")
        score_val = 100
        if q1.startswith("Frequently"): score_val -= 30
        elif q1.startswith("Occasionally"): score_val -= 15

        if q2.startswith("Always"): score_val -= 35
        elif q2.startswith("Sometimes"): score_val -= 20

        if q3.startswith("Never"): score_val -= 25
        elif q3.startswith("Sometimes"): score_val -= 10

        if q4 == "No": score_val -= 10

        score_val = max(score_val, 15)

        if score_val >= 80:
            p_title, p_color, p_desc = "Prudent Borrower", "#10B981", "High financial discipline! You protect your data privacy and avoid unverified lending apps."
        elif score_val >= 50:
            p_title, p_color, p_desc = "Cautionary Borrower", "#F59E0B", "Moderate Caution: Always read the Key Fact Statement (KFS) and verify NBFC registration before borrowing."
        else:
            p_title, p_color, p_desc = "Vulnerable Borrower", "#EF4444", "High Risk! Frequent short-term loans and granting contact list permissions leaves you exposed to illegal harassment apps."

        st.markdown(
            f"""
            <div style="background:{t['card_bg']}; border:2px solid {p_color}; border-radius:16px; padding:24px; text-align:center; box-shadow:0 8px 25px rgba(0,0,0,0.3);">
                <div style="font-size:3rem; margin-bottom:8px;">🏆</div>
                <div style="font-size:0.75rem; color:{p_color}; font-weight:800; letter-spacing:1px;">MONEYSIGN® PROFILE</div>
                <div style="font-size:2rem; font-weight:900; color:{t['text']}; margin:4px 0;">{p_title}</div>
                <div style="font-size:1.1rem; font-weight:800; color:{p_color};">Safety Index: {score_val} / 100</div>
                <p style="font-size:0.88rem; color:{t['muted']}; margin-top:12px; line-height:1.5;">{p_desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================
# TAB 3: APP RANKINGS HUB
# ==========================================
with tab_rankings:
    st.markdown("### 🏆 FinShield Digital Lending App Safety Rankings")
    st.caption("Evaluated database of popular lending apps on the Google Play Store, ranked by privacy safety, RBI compliance, and review harassment risk.")

    try:
        df_ranked = get_ranked_apps_df()
        
        # Top Controls: Search bar & Filters
        f_col1, f_col2, f_col3 = st.columns([2.4, 1.3, 1.3], gap="small")
        
        with f_col1:
            search_q = st.text_input("🔍 Search App", placeholder="Type e.g. KreditBee, Groww, Navi...", key="ranking_search_q")
        with f_col2:
            tier_filter = st.selectbox(
                "Safety Tier Filter",
                options=["All Safety Tiers", "🛡️ Safest Tier (80-100)", "⚠️ Moderate Caution (50-79)", "🚨 High Risk (<50)"],
                key="ranking_tier_filter"
            )
        with f_col3:
            sort_by = st.selectbox(
                "Sort Rankings By",
                options=["FinShield Score (Highest First)", "FinShield Score (Lowest First)", "Installs (Highest First)"],
                key="ranking_sort_by"
            )

        # Filtering logic
        filtered_df = df_ranked.copy()
        
        if search_q:
            q = search_q.strip()
            filtered_df = filtered_df[
                filtered_df["app_name"].str.contains(q, case=False, na=False) |
                filtered_df["app_id"].str.contains(q, case=False, na=False)
            ]
            
        if tier_filter == "🛡️ Safest Tier (80-100)":
            filtered_df = filtered_df[filtered_df["safety_score"] >= 80]
        elif tier_filter == "⚠️ Moderate Caution (50-79)":
            filtered_df = filtered_df[(filtered_df["safety_score"] >= 50) & (filtered_df["safety_score"] < 80)]
        elif tier_filter == "🚨 High Risk (<50)":
            filtered_df = filtered_df[filtered_df["safety_score"] < 50]

        if sort_by == "FinShield Score (Lowest First)":
            filtered_df = filtered_df.sort_values(by="safety_score", ascending=True)
        elif sort_by == "Installs (Highest First)":
            filtered_df = filtered_df.sort_values(by="install_count", ascending=False)
        else: # Highest First
            filtered_df = filtered_df.sort_values(by="safety_score", ascending=False)

        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
        
        if filtered_df.empty:
            st.info("🔍 No lending apps found matching your search and filter criteria.", icon="ℹ️")
        else:
            # Card View matching 1Finance Reference UI
            c_dark = st.session_state.dark_mode
            card_bg = "#111622" if c_dark else "#FFFFFF"
            card_bdr = "1px solid rgba(255, 255, 255, 0.08)" if c_dark else "1px solid #E2E8F0"
            text_color = "#F8FAFC" if c_dark else "#0F172A"
            muted_color = "#94A3B8" if c_dark else "#64748B"
            
            rows = list(filtered_df.iterrows())
            for i in range(0, len(rows), 2):
                col_a, col_b = st.columns(2, gap="medium")
                
                for col, (idx, row) in zip([col_a, col_b], rows[i:i+2]):
                    with col:
                        rank_num = row["rank"]
                        app_name = row["app_name"]
                        app_id = row["app_id"]
                        score = row["safety_score"]
                        cat_tag = row["cat_tag"]
                        installs_str = row["installs_str"]
                        disclosure = row["disclosure_score"]
                        redflag_pct = row["redflag_pct"]
                        is_known = row["is_known"]
                        
                        # Score pill styling
                        if score >= 80:
                            score_bg = "rgba(16, 185, 129, 0.18)" if c_dark else "#DCFCE7"
                            score_bdr = "1px solid rgba(16, 185, 129, 0.35)" if c_dark else "1px solid #86EFAC"
                            score_fg = "#34D399" if c_dark else "#15803D"
                            status_text = "RBI Aligned"
                        elif score >= 50:
                            score_bg = "rgba(245, 158, 11, 0.18)" if c_dark else "#FEF3C7"
                            score_bdr = "1px solid rgba(245, 158, 11, 0.35)" if c_dark else "1px solid #FDE68A"
                            score_fg = "#FBBF24" if c_dark else "#B45309"
                            status_text = "Caution Advised"
                        else:
                            score_bg = "rgba(239, 68, 68, 0.18)" if c_dark else "#FEE2E2"
                            score_bdr = "1px solid rgba(239, 68, 68, 0.35)" if c_dark else "1px solid #FCA5A5"
                            score_fg = "#F87171" if c_dark else "#B91C1C"
                            status_text = "High Risk"
                            
                        rank_bg = "#1E293B" if c_dark else "#F1F5F9"
                        rank_bdr = "1px solid rgba(255, 255, 255, 0.1)" if c_dark else "1px solid #E2E8F0"
                        rank_fg = "#CBD5E1" if c_dark else "#334155"

                        card_html = f"""
                        <div style="background:{card_bg}; border:{card_bdr}; border-radius:16px; padding:18px 20px; margin-bottom:10px; box-shadow:0 6px 20px rgba(0,0,0,0.15);">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <span style="font-size:0.74rem; font-weight:700; color:{muted_color}; text-transform:uppercase; letter-spacing:0.5px;">{cat_tag}</span>
                                <span style="font-size:0.72rem; background:{score_bg}; color:{score_fg}; padding:2px 8px; border-radius:12px; font-weight:800; border:{score_bdr};">{status_text}</span>
                            </div>
                            <div style="font-size:1.1rem; font-weight:800; color:{text_color}; margin-bottom:12px; line-height:1.35; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{app_name}">
                                {app_name}
                            </div>
                            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:14px; background:{'rgba(255,255,255,0.02)' if c_dark else '#F8FAFC'}; padding:8px 12px; border-radius:10px;">
                                <div>
                                    <div style="color:{muted_color}; font-size:0.72rem; font-weight:600;">Installs Base</div>
                                    <div style="font-weight:800; color:{text_color}; font-size:0.92rem;">{installs_str}</div>
                                </div>
                                <div>
                                    <div style="color:{muted_color}; font-size:0.72rem; font-weight:600;">Terms Disclosed</div>
                                    <div style="font-weight:800; color:{text_color}; font-size:0.92rem;">{disclosure} / 5</div>
                                </div>
                            </div>
                            <div style="display:grid; grid-template-columns:1fr 1.15fr; gap:10px;">
                                <div style="background:{rank_bg}; border:{rank_bdr}; border-radius:10px; padding:8px 10px; text-align:center; display:flex; align-items:center; justify-content:center; gap:4px;">
                                    <span style="font-size:0.75rem; color:{rank_fg}; font-weight:700;">FinShield Rank</span>
                                    <span style="font-size:0.9rem; font-weight:900; color:{rank_fg};">🌿 {rank_num:02d}</span>
                                </div>
                                <div style="background:{score_bg}; border:{score_bdr}; border-radius:10px; padding:8px 10px; text-align:center; display:flex; align-items:center; justify-content:center; gap:2px;">
                                    <span style="font-size:0.75rem; color:{score_fg}; font-weight:700;">FinShield Score: </span>
                                    <strong style="font-size:1.05rem; font-weight:900; color:{score_fg};">{score}</strong>
                                    <span style="font-size:0.72rem; color:{score_fg}; opacity:0.8;">/100</span>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        with st.expander(f"🔍 View Details & Permissions for {app_name.split(':')[0]}"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(f"**Package ID:** `{app_id}`")
                                st.write(f"**Contacts Access:** {'🔴 Asks Contacts' if row['contacts'] else '🟢 No Contacts'}")
                                st.write(f"**SMS Reading:** {'🔴 Asks SMS' if row['sms'] else '🟢 No SMS'}")
                            with c2:
                                st.write(f"**Photos/Media:** {'🔴 Asks Photos' if row['photos'] else '🟢 No Photos'}")
                                st.write(f"**Harassment Mentions:** `{redflag_pct:.1f}%`")
                                st.write(f"**RBI Status:** {'🟢 Regulated NBFC Partner' if is_known else '⚠️ Unverified Partner'}")
                            
                            if st.button("Run Live Riskometer Audit ➔", key=f"btn_audit_app_{idx}", use_container_width=True):
                                st.session_state.active_package = app_id
                                st.session_state.show_inline_audit = app_id
                                st.rerun()
                                
                            if st.session_state.get("show_inline_audit") == app_id:
                                st.markdown("---")
                                st.success(f"✅ **Live Riskometer Audit Output for {app_name.split(':')[0]}**", icon="🛡️")
                                gauge_html = render_rbi_riskometer_card(row["risk_proba"], c_dark)
                                st.markdown(gauge_html, unsafe_allow_html=True)
                                
                                if row["reasons"]:
                                    st.markdown("**Key Risk Drivers & Audit Explanation:**")
                                    for item in row["reasons"]:
                                        if isinstance(item, (tuple, list)) and len(item) == 2:
                                            reason_text, is_bad = item
                                            icon = "🔴" if is_bad else "🟢"
                                            st.write(f"{icon} {reason_text}")

    except FileNotFoundError:
        st.warning(f"File {FEATURES_CSV_PATH} not found.")

# ==========================================
# TAB 4: LOAN CALCULATORS
# ==========================================
with tab_calculators:
    st.markdown("### 🧮 Financial Advisory Calculators")

    c_left, c_right = st.columns([1, 1], gap="large")

    with c_left:
        st.markdown("#### 💰 Personal Loan Prepayment Calculator")
        c_amt = st.number_input("Loan Amount (₹)", value=200000, step=10000, key="calc_amt")
        c_rate = st.number_input("Interest Rate (% p.a.)", value=18.0, step=0.5, key="calc_rate")
        c_tenure = st.number_input("Tenure (Months)", value=24, step=1, key="calc_tenure")

        if c_rate > 0 and c_tenure > 0:
            r = (c_rate / 12) / 100
            emi = (c_amt * r * ((1 + r) ** c_tenure)) / (((1 + r) ** c_tenure) - 1)
            tot_int = (emi * c_tenure) - c_amt
            savings = tot_int * 0.35

            st.markdown(
                f"""
                <div style="background:{t['card_bg']}; border:1px solid {t['card_border']}; border-radius:12px; padding:16px; margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:6px;"><span>Monthly EMI:</span><strong>₹{emi:,.0f}</strong></div>
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:6px;"><span>Total Interest Paid:</span><strong>₹{tot_int:,.0f}</strong></div>
                    <div style="display:flex; justify-content:space-between; font-size:0.95rem; color:#34D399; font-weight:700;"><span>Prepayment Interest Savings:</span><strong>₹{savings:,.0f}</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with c_right:
        st.markdown("#### ⚠️ Hidden Fees & True APR Detector")
        d_amt = st.number_input("Disbursed Amount (₹)", value=10000, step=1000, key="apr_disb")
        r_amt = st.number_input("Repayment Amount (₹)", value=12500, step=1000, key="apr_repay")
        d_days = st.number_input("Tenure (Days)", value=7, step=1, key="apr_days")

        if d_amt > 0 and d_days > 0:
            extra = r_amt - d_amt
            apr = (extra / d_amt / d_days) * 365 * 100

            st.markdown(
                f"""
                <div style="background:{t['card_bg']}; border:1px solid {t['card_border']}; border-radius:12px; padding:16px; margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:6px;"><span>Total Extra Fee & Interest:</span><strong>₹{extra:,.0f}</strong></div>
                    <div style="display:flex; justify-content:space-between; font-size:0.95rem; color:#F87171; font-weight:800;"><span>True Annualized APR:</span><strong>{apr:,.0f}% p.a.</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if d_days <= 7 or apr > 100:
                st.error("⚠️ Predatory 7-day loan alert! Annualized APR exceeds 100%. Violates RBI Digital Lending Directives.")
            else:
                st.success("✓ Terms within standard NBFC lending parameters.")

# ==========================================
# TAB 5: RBI GUIDELINES
# ==========================================
with tab_rbi:
    st.markdown("### 📜 RBI Digital Lending Guidelines 2026 Checklist")
    st.markdown(
        """
        - 🚫 **No Prohibited Data Access**: Lending apps are strictly forbidden from asking access to your phone contacts list, private photo gallery, or reading SMS.
        - 📄 **Key Fact Statement (KFS)**: Lenders must provide a standardized Key Fact Statement detailing all APR, processing fees, and penalties before agreement execution.
        - 🏦 **Direct Bank Account Transfer**: Loan disbursements and repayments must happen strictly between borrower's bank account and regulated bank/NBFC bank account.
        - 🔗 **Grievance Redressal**: Every app must publish a dedicated Grievance Redressal Officer contact and registered office address.
        """
    )
    st.markdown(
        """
        ---
        - 🏛️ **Verify Lenders on RBI Sachet**: [sachet.rbi.org.in](https://sachet.rbi.org.in)
        - 🚨 **File CyberCrime Complaint**: [cybercrime.gov.in](https://cybercrime.gov.in)
        """
    )

st.markdown('</div>', unsafe_allow_html=True)
