import re
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

MIN_REVIEWS_REQUIRED = 10  # matches the notebook's df["total_reviews"] >= 10 filter

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

def scrape_app(package_name: str) -> dict:
    """
    STUB. Replace with the real Play Store scraper. Must return a dict
    shaped like this — same keys, same types — so extract_features()
    below keeps working unchanged.
    """
    return {
        "package_name": package_name,
        "description": (
            "Get instant loans up to 50000 at low interest rate per annum. "
            "RBI-registered NBFC partner. Contact support@example.com for help."
        ),
        "privacy_policy": "https://example.com/privacy",
        "permissions": {
            "contacts": 1, "sms": 1, "microphone": 0,
            "location": 1, "photos_media_storage": 1,
        },
        "installs": "100,000+",
        "reviews": [
            "Fast and easy loan, good app",
            "They threatened to call my family and office, harassment",
            "Recovery agent called my boss, very unprofessional, blackmail",
            "good experience, quick disbursement",
            "Shared my contact list without permission, scam",
            "decent app works fine",
            "morphed my photo and sent to my contacts, fraud app",
            "smooth process, no complaints",
            "customer care never responds",
            "app crashed twice but loan was approved",
            "they called my office repeatedly, defame",
            "easy to use, would recommend",
        ],
    }

def disclosure_score(description: str, privacy_policy: str) -> int:
    """Same 5-point check as the notebook's disclosure_score()."""
    desc = str(description).lower()
    has_rate = bool(re.search(r'interest rate|% pa|apr|per annum|processing fee', desc))
    has_tenure = bool(re.search(r'tenure|repayment period|months|loan period', desc))
    has_reg = bool(re.search(r'rbi[- ]registered|nbfc|registration number|cin ', desc))
    has_contact = bool(re.search(r'customer care|grievance|support@|contact us|helpline', desc))
    pp = str(privacy_policy)
    has_pp = pp not in ('nan', '', 'None') and 'http' in pp
    return int(has_rate) + int(has_tenure) + int(has_reg) + int(has_contact) + int(has_pp)


def parse_installs(installs_text: str):
    """Same as the notebook's parse_installs()."""
    if not installs_text:
        return None
    digits = re.sub(r'[^0-9]', '', str(installs_text))
    return int(digits) if digits else 0


def extract_features(raw: dict) -> dict:
    """
    Turns raw scraped data into the exact 11 features the model expects.
    Raises a clear error if there aren't enough reviews — matches the
    notebook's own quality bar (>= 10 reviews).
    """
    reviews = raw.get("reviews", [])
    total_reviews = len(reviews)

    if total_reviews < MIN_REVIEWS_REQUIRED:
        raise ValueError(
            f"Only {total_reviews} reviews found — need at least "
            f"{MIN_REVIEWS_REQUIRED} for a reliable score (same bar the "
            f"model was trained with)."
        )

    redflag_hits = sum(1 for r in reviews if REDFLAG_PATTERN.search(r))
    review_redflag_score = round(redflag_hits / total_reviews, 3)

    sentiments = [analyzer.polarity_scores(r)['compound'] for r in reviews]
    avg_review_sentiment = sum(sentiments) / total_reviews

    strongly_negative = sum(1 for s in sentiments if s < -0.5)
    pct_strongly_negative_reviews = strongly_negative / total_reviews

    avg_review_length = sum(len(r.split()) for r in reviews) / total_reviews

    perms = raw.get("permissions", {})

    return {
        "contacts": perms.get("contacts", 0),
        "sms": perms.get("sms", 0),
        "microphone": perms.get("microphone", 0),
        "location": perms.get("location", 0),
        "photos_media_storage": perms.get("photos_media_storage", 0),
        "disclosure_score": disclosure_score(raw.get("description", ""), raw.get("privacy_policy", "")),
        "review_redflag_score": review_redflag_score,
        "avg_review_sentiment": avg_review_sentiment,
        "pct_strongly_negative_reviews": pct_strongly_negative_reviews,
        "avg_review_length": avg_review_length,
        "install_count": parse_installs(raw.get("installs")),
    }

@st.cache_data
def load_features_table():
    """Loads app_features_final.csv once and caches it (fast on every rerun)."""
    return pd.read_csv(FEATURES_CSV_PATH)


def get_app_choices():
    """Returns the list of app names available to pick from, or [] if the CSV is missing."""
    try:
        df = load_features_table()
        return sorted(df["app_name"].dropna().astype(str).unique().tolist())
    except FileNotFoundError:
        return []


def lookup_app_features(identifier: str):
    """
    Finds the row for this app (matched by app_name or app_id, case-insensitive)
    and returns just the 11 model-ready feature values as a dict.
    Returns None if the app isn't in the dataset.
    """
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
    return {col: row[col] for col in FEATURE_COLUMNS if col in row}

#model loading
import os
USE_FAKE_MODEL = not os.path.exists("predatory_loan_detector.pkl")

@st.cache_resource
def load_model():
    """
    @st.cache_resource makes Streamlit load the model file only ONCE,
    the first time the app runs, instead of reloading it from disk on
    every single click — makes the demo feel instant.
    """
    return joblib.load("predatory_loan_detector.pkl")

def _fake_predict(features: dict):
    """
    TEMPORARY stand-in for the real model, used only while
    predatory_loan_detector.pkl hasn't arrived yet. Same rough logic
    as the notebook's features (more red flags / less disclosure /
    more sensitive permissions = higher score), just not a real
    trained model. DELETE this function once the real .pkl is in.
    """
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
    if USE_FAKE_MODEL:
        return _fake_predict(features)

    model = load_model()
    row = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    proba = model.predict_proba(row)[0][1]  

    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = dict(zip(feature_names, classifier.coef_[0]))

    top = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    reasons = []
    for name, _coef in top:
        clean_name = name.replace("num__", "")
        reasons.append(explain_feature(clean_name, features.get(clean_name)))

    return proba, reasons

DEMO_FALLBACKS = {
    # "known.bad.loanapp": {
    #     "score": 0.91,
    #     "reasons": [("About 42% of reviews mention harassment or threats", True), ...],
    #     "features": {...},  # optional, only used for the technical details expander
    # },
}

def score_app(identifier: str):
    if identifier in DEMO_FALLBACKS:
        cached = DEMO_FALLBACKS[identifier]
        return cached["score"], cached["reasons"], True, cached.get("features", {})

    try:
        features = lookup_app_features(identifier)
        if features is None:
            raise ValueError(f"'{identifier}' isn't in the scraped dataset yet.")
        score, reasons = predict(features)
        return score, reasons, False, features
    except ValueError as e:
        st.warning(str(e))
        return None, None, None, None
    except FileNotFoundError:
        st.error(f"Can't find {FEATURES_CSV_PATH} — make sure it's in the same folder as this file.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Something went wrong scoring this app: {e}")
        return None, None, None, None

st.set_page_config(page_title="FINSHIELD-Loan App Risk Scorer", page_icon="🛡️", layout="centered")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def theme_colors(dark: bool) -> dict:
    if dark:
        return dict(
            app_bg="#0F172A", text="#E2E8F0", muted="#94A3B8",
            card_bg="#1E293B", card_border="#334155",
            red_bg="#3F1D1D", red_text="#FCA5A5",
            orange_bg="#3F2D12", orange_text="#FCD34D",
            green_bg="#0F2E1F", green_text="#6EE7B7",
            stat_bg="#CA8A04", stat_border="#EAB308", stat_text="#FFFFFF",
        )
    return dict(
        app_bg="#FAFAFA", text="#1E1E2E", muted="#6B7280",
        card_bg="#FFFFFF", card_border="#E5E7EB",
        red_bg="#FEE4E2", red_text="#B42318",
        orange_bg="#FEF3C7", orange_text="#B45309",
        green_bg="#D1FADF", green_text="#05603A",
        stat_bg="#FEF08A", stat_border="#FDE047", stat_text="#713F12",
    )


t = theme_colors(st.session_state.dark_mode)
button_red = "#EF4444"
accent = "#818CF8" if st.session_state.dark_mode else "#6366F1"

st.markdown(f"""
<style>
    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {t['app_bg']} !important;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background-color: {t['app_bg']} !important;
    }}
    .block-container {{ padding-top: 2rem; max-width: 700px; }}
    h1, h2, h3, p, span, label, .stMarkdown {{ color: {t['text']}; }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {t['card_bg']}; border: 1px solid {t['card_border']};
        border-radius: 16px; padding: 8px 12px;
    }}

    .stTextInput > div > div > input {{
        background-color: {t['app_bg']}; color: {t['text']};
        border: 1.5px solid {t['card_border']}; border-radius: 8px;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: {accent} !important; box-shadow: 0 0 0 1px {accent} !important;
    }}
    .stButton > button {{
        background-color: {button_red} !important; color: white !important;
        border: none !important; border-radius: 8px !important; font-weight: 600;
    }}
    .stButton > button:hover {{ opacity: 0.9; }}

    .stat-box {{
        border-radius: 10px; padding: 12px 14px; text-align: left;
    }}
    .stat-label {{ font-size: 0.78rem; margin-bottom: 2px; opacity: 0.85; }}
    .stat-value {{ font-size: 1.25rem; font-weight: 600; }}
    .verdict-card {{
        border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
    }}
    .summary-box {{
        background-color: {t['app_bg']}; border: 1px solid {t['card_border']};
        border-radius: 10px; padding: 16px 18px; font-size: 0.96rem; line-height: 1.6;
        color: {t['text']};
    }}
</style>
""", unsafe_allow_html=True)

main = st.container(border=True)
with main:
    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.title("FINSHIELD-Loan App Risk Scorer")
        st.caption("Check a Play Store lending app's predatory risk — before you install it, not after.")
    with top_r:
        st.toggle("🌙 Dark", key="dark_mode")

    if USE_FAKE_MODEL:
        st.info(
            "🔧 Running with a placeholder scoring formula (predatory_loan_detector.pkl not found yet). "
            "Good for testing the app flow — not a real prediction.",
            icon="🔧",
        )

    st.write("")
    app_choices = get_app_choices()
    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
    if app_choices:
        with col1:
            package_name = st.selectbox("Choose an app to check", options=app_choices)
        with col2:
            check_clicked = st.button("Check Risk", type="primary", use_container_width=True)
    else:
        with col1:
            package_name = st.text_input("Play Store package name", placeholder="e.g. com.example.loanapp")
        with col2:
            check_clicked = st.button("Check Risk", type="primary", use_container_width=True)
        st.caption(f"⚠️ {FEATURES_CSV_PATH} not found in this folder — showing free-text input instead of the app list.")

    if check_clicked:
        if not package_name:
            st.warning("Enter a package name first.")
        else:
            with st.spinner("Scoring app..."):
                score, reasons, used_fallback, features = score_app(package_name)

            if score is not None:
                st.write("")

                if score >= 0.6:
                    verdict, bg, fg = "High Risk", t["red_bg"], t["red_text"]
                elif score >= 0.3:
                    verdict, bg, fg = "Medium Risk", t["orange_bg"], t["orange_text"]
                else:
                    verdict, bg, fg = "Looks Legitimate", t["green_bg"], t["green_text"]

                st.markdown(
                    f'<div class="verdict-card" style="background-color:{bg}; color:{fg};">'
                    f'<div style="font-size:1.3rem; font-weight:700;">{verdict}</div>'
                    f'<div style="font-size:2rem; font-weight:700;">{score*100:.0f}% risk</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(max(score, 0.0), 1.0))
                if used_fallback:
                    st.caption("(cached result)")

                #boxes
                if features:
                    st.write("")
                    st.markdown("**App at a glance**")

                    installs = features.get("install_count")
                    disclosure = features.get("disclosure_score", 0)
                    redflag_pct = (features.get("review_redflag_score", 0) or 0) * 100
                    neg_pct = (features.get("pct_strongly_negative_reviews", 0) or 0) * 100
                    sentiment = features.get("avg_review_sentiment", 0)
                    length = features.get("avg_review_length", 0)

                    def tier_color(level):
                        return {
                            "red": (t["red_bg"], t["red_text"]),
                            "orange": (t["orange_bg"], t["orange_text"]),
                            "green": (t["green_bg"], t["green_text"]),
                        }[level]

                    def disclosure_tier(v):
                        return "red" if v <= 2 else "orange" if v == 3 else "green"

                    def redflag_tier(pct):
                        return "red" if pct >= 15 else "orange" if pct >= 5 else "green"

                    def negrev_tier(pct):
                        return "red" if pct >= 20 else "orange" if pct >= 10 else "green"

                    def installs_tier(v):
                        if v is None:
                            return "orange"
                        return "red" if v < 10000 else "orange" if v < 100000 else "green"

                    def sentiment_tier(v):
                        return "orange" if (v > 0.6 or v < -0.3) else "green"

                    def length_tier(v):
                        return "red" if v >= 20 else "orange" if v >= 10 else "green"

                    stats = [
                        ("📦 Installs", f"{installs:,}" if installs is not None else "—", installs_tier(installs)),
                        ("📝 Terms disclosed", f"{disclosure} / 5", disclosure_tier(disclosure)),
                        ("🚩 Harassment mentions", f"{redflag_pct:.0f}%", redflag_tier(redflag_pct)),
                        ("😠 Strongly negative reviews", f"{neg_pct:.0f}%", negrev_tier(neg_pct)),
                        ("🙂 Avg. review tone", f"{sentiment:+.2f}", sentiment_tier(sentiment)),
                        ("📏 Avg. review length", f"{length:.0f} words", length_tier(length)),
                    ]
                    b1, b2, b3 = st.columns(3)
                    for box, (label, value, level) in zip([b1, b2, b3, b1, b2, b3], stats):
                        bg_c, fg_c = tier_color(level)
                        with box:
                            st.markdown(
                                f'<div class="stat-box" style="background-color:{bg_c}; color:{fg_c};">'
                                f'<div class="stat-label">{label}</div>'
                                f'<div class="stat-value">{value}</div></div>',
                                unsafe_allow_html=True,
                            )
                            st.write("")

                    st.caption(
                        "🟢 fine · 🟡 minor concern · 🔴 risky  —  "
                        "**Terms disclosed**: how clearly the app states interest rate, tenure & registration (out of 5). "
                        "**Harassment mentions**: % of reviews describing threats or abusive recovery tactics. "
                        "**Strongly negative reviews**: % of reviews that are clearly unhappy. "
                        "**Avg. review tone**: how positive (+1) or negative (−1) reviews are overall. "
                        "**Avg. review length**: how detailed reviews tend to be, in words."
                    )
                #App summary
                st.write("")
                st.markdown("**App summary**")

                concerns = [s for s, flag in reasons if flag]
                positives = [s for s, flag in reasons if not flag]

                if score >= 0.6:
                    opener = "This app shows strong signs of predatory behavior and should be approached with real caution."
                elif score >= 0.3:
                    opener = "This app shows a few concerning signs — it's worth being cautious before using it."
                else:
                    opener = "This app looks largely legitimate based on the information available."

                summary_sentences = [opener]
                summary_sentences += [f"{s}." for s in concerns]
                if positives:
                    summary_sentences.append("On the positive side: " + " ".join(f"{p}." for p in positives))

                if score >= 0.6:
                    summary_sentences.append(
                        "If you're considering a loan, it's strongly recommended to look for a more "
                        "established, RBI-registered alternative instead."
                    )
                elif score >= 0.3:
                    summary_sentences.append(
                        "Proceed carefully, and double-check the lender's registration and terms before borrowing."
                    )
                else:
                    summary_sentences.append(
                        "No major red flags were found, but always confirm the lender's registration and "
                        "read the loan terms carefully before borrowing."
                    )

                st.markdown(
                    f'<div class="summary-box">{" ".join(summary_sentences)}</div>',
                    unsafe_allow_html=True,
                )
