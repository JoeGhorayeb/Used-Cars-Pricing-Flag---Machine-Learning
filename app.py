"""
Used Car Overpricing Detector — Lebanon Market
Streamlit GUI app.

Run with:
    pip install streamlit
    streamlit run app.py

Requires these files (saved by the notebook's "Save Artifacts" cell) to be
in the same folder as this script:
    best_model.joblib
    preprocessor.joblib
    model_freq_map.joblib
    feature_names.joblib
    threshold.joblib
    brand_list.joblib
    fueltype_list.joblib
    transmission_list.joblib
"""

import streamlit as st
import pandas as pd
import joblib


def html_block(raw_html: str) -> None:
    """
    Render raw HTML via st.markdown safely.

    Streamlit's markdown renderer treats any line indented 4+ spaces as a
    Markdown code block, which breaks nested HTML written with normal
    Python indentation (e.g. inside an `if` block). Stripping each line's
    leading/trailing whitespace before rendering avoids that entirely.
    """
    stripped = "\n".join(line.strip() for line in raw_html.strip().splitlines())
    st.markdown(stripped, unsafe_allow_html=True)


st.set_page_config(
    page_title="Souk Sense | Lebanon Car Pricing",
    page_icon="🛞",
    layout="centered",
    initial_sidebar_state="collapsed",
)

#styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --bg:        #12151C;
    --surface:   #1B2029;
    --surface-2: #232A35;
    --border:    #2E3644;
    --text:      #EDEFF3;
    --text-muted:#8B93A1;
    --brass:     #C89B3C;
    --brass-dim: #8A6B2A;
    --green:     #4CAF6D;
    --amber:     #D9A441;
    --red:       #E5595F;
}

/* base */
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.2rem; max-width: 760px; }

/* ---- masthead ---- */
.dash-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
}
.dash-header .mark {
    width: 44px; height: 44px;
    border-radius: 10px;
    background: linear-gradient(145deg, var(--brass), var(--brass-dim));
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.dash-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.55rem;
    margin: 0;
    letter-spacing: -0.02em;
    color: var(--text);
}
.dash-header p {
    margin: 2px 0 0 0;
    color: var(--text-muted);
    font-size: 0.86rem;
}
.odometer-rule {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--brass-dim);
    text-transform: uppercase;
    margin: 10px 0 24px 0;
}

/* ---- section labels ---- */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--brass);
    margin: 0 0 10px 2px;
}

/* ---- form surface ---- */
div[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 26px 26px 14px 26px;
}
label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
div[data-baseweb="input"], div[data-baseweb="select"] > div {
    background: var(--surface-2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
div[data-baseweb="select"] span { color: var(--text) !important; }
.stNumberInput input, .stTextInput input { color: var(--text) !important; }

.stFormSubmitButton button {
    background: var(--brass) !important;
    color: #1A1200 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em;
    padding: 0.6rem 0 !important;
    margin-top: 6px;
    transition: filter 0.15s ease;
}
.stFormSubmitButton button:hover { filter: brightness(1.08); }

/* ---- verdict card ---- */
.verdict-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent, var(--brass));
    border-radius: 12px;
    padding: 22px 26px;
    margin-top: 22px;
}
.verdict-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    margin: 0 0 2px 0;
    color: var(--accent, var(--text));
}
.verdict-sub {
    color: var(--text-muted);
    font-size: 0.83rem;
    margin: 0 0 18px 0;
}

/* ---- metrics row ---- */
.metric-row { display: flex; gap: 14px; margin-bottom: 20px; }
.metric-box {
    flex: 1;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
}
.metric-box .m-label {
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.metric-box .m-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.28rem;
    font-weight: 600;
    color: var(--text);
}
.metric-box .m-value.accent { color: var(--accent, var(--brass)); }

/* ---- deviation gauge ---- */
.gauge-wrap { margin-top: 4px; }
.gauge-label-row {
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.gauge-track {
    position: relative;
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg,
        var(--green) 0%, var(--green) 38%,
        var(--amber) 38%, var(--amber) 62%,
        var(--red) 62%, var(--red) 100%);
    opacity: 0.85;
}
.gauge-marker {
    position: absolute;
    top: -5px;
    width: 3px;
    height: 20px;
    background: var(--text);
    border-radius: 2px;
    box-shadow: 0 0 0 3px rgba(237,239,243,0.15);
}

/* footer note */
.footnote {
    color: var(--text-muted);
    font-size: 0.76rem;
    line-height: 1.5;
    margin-top: 16px;
}
.page-footer {
    margin-top: 34px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.74rem;
    text-align: center;
}
</style>
"""
html_block(CSS)


@st.cache_resource
def load_artifacts():
    model = joblib.load('joblib-files/best_model.joblib')
    preprocessor = joblib.load('joblib-files/preprocessor.joblib')
    model_freq_map = joblib.load('joblib-files/model_freq_map.joblib')
    feature_names = joblib.load('joblib-files/feature_names.joblib')
    threshold = joblib.load('joblib-files/threshold.joblib')
    brands = joblib.load('joblib-files/brand_list.joblib')
    fuel_types = joblib.load('joblib-files/fueltype_list.joblib')
    transmissions = joblib.load('joblib-files/transmission_list.joblib')
    return model, preprocessor, model_freq_map, feature_names, threshold, brands, fuel_types, transmissions


try:
    (model, preprocessor, model_freq_map, feature_names,
     threshold, brands, fuel_types, transmissions) = load_artifacts()
except FileNotFoundError:
    html_block('<div class="dash-header"><div class="mark">🛞</div>'
               '<div><h1>Souk Sense</h1><p>Lebanon used-car pricing dashboard</p></div></div>')
    st.error(
        "Model artifacts not found. Run the notebook's final cells first "
        "(the ones under **'Save Artifacts for the GUI App'**) so this app "
        "has a trained model to load, then restart this app from the same folder."
    )
    st.stop()


def categorize(actual_price, predicted_price, thresh):
    dev = (actual_price - predicted_price) / predicted_price * 100
    if dev > thresh:
        return 'Overpriced', dev
    elif dev < -thresh:
        return 'Underpriced', dev
    else:
        return 'Fairly Priced', dev


def predict_car_pricing(brand, model_name, year, mileage, engine_size, fuel_type,
                         transmission, tax, mpg, listing_price_usd):
    input_df = pd.DataFrame([{
        'model': model_name,
        'year': year,
        'transmission': transmission,
        'mileage': mileage,
        'fuelType': fuel_type,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engine_size,
        'Brand': brand,
    }])

    input_df['model_freq'] = input_df['model'].map(model_freq_map).fillna(0)
    input_df = input_df.drop(columns=['model'])

    input_encoded = preprocessor.transform(input_df)
    input_encoded = pd.DataFrame(input_encoded, columns=feature_names)

    predicted_price = model.predict(input_encoded)[0]
    category, deviation_pct = categorize(listing_price_usd, predicted_price, threshold)

    return predicted_price, deviation_pct, category

#header
html_block(
    """
    <div class="dash-header">
        <div class="mark">🛞</div>
        <div>
            <h1>Souk Sense</h1>
            <p>Fair-price estimation for the Lebanese used-car market</p>
        </div>
    </div>
    <div class="odometer-rule">
        <span>ML-Driven Valuation</span>
        <span>Scaled for LB Import Pricing</span>
    </div>
    """
)

#form
html_block('<div class="section-label">Vehicle Details</div>')

with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", brands)
        model_name = st.text_input("Model", placeholder="e.g. Corolla, A3, Golf")
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2019, step=1)
        mileage = st.number_input("Mileage", min_value=0, value=30000, step=1000)
        engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=6.0, value=1.6, step=0.1)

    with col2:
        fuel_type = st.selectbox("Fuel Type", fuel_types)
        transmission = st.selectbox("Transmission", transmissions)
        tax = st.number_input("Annual Tax (£)", min_value=0, value=145, step=5)
        mpg = st.number_input("MPG", min_value=0.0, value=50.0, step=1.0)
        listing_price_usd = st.number_input("Listing Price (USD, Lebanon-scaled)", min_value=0, value=20000, step=500)

    submitted = st.form_submit_button("Check Pricing", use_container_width=True)

#results
if submitted:
    if not model_name.strip():
        st.warning("Please enter a model name.")
    else:
        predicted_price, deviation_pct, category = predict_car_pricing(
            brand, model_name.strip(), year, mileage, engine_size, fuel_type,
            transmission, tax, mpg, listing_price_usd
        )

        accent_map = {"Overpriced": "var(--red)", "Underpriced": "var(--green)", "Fairly Priced": "var(--amber)"}
        icon_map = {"Overpriced": "▲ OVERPRICED", "Underpriced": "▼ UNDERPRICED", "Fairly Priced": "● FAIRLY PRICED"}
        accent = accent_map[category]

        # clamp deviation into a -50%..+50% gauge range for the marker position
        gauge_min, gauge_max = -50, 50
        clamped = max(gauge_min, min(gauge_max, deviation_pct))
        marker_pct = (clamped - gauge_min) / (gauge_max - gauge_min) * 100

        html_block(
            f"""
            <div class="verdict-card" style="--accent:{accent}; border-left-color:{accent};">
                <p class="verdict-title" style="color:{accent};">{icon_map[category]}</p>
                <p class="verdict-sub">Based on {len(feature_names)} market features vs. this listing</p>

                <div class="metric-row">
                    <div class="metric-box">
                        <div class="m-label">Estimated Fair Price</div>
                        <div class="m-value">${predicted_price:,.0f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="m-label">Listed At</div>
                        <div class="m-value">${listing_price_usd:,.0f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="m-label">Deviation</div>
                        <div class="m-value accent" style="color:{accent};">{deviation_pct:+.1f}%</div>
                    </div>
                </div>

                <div class="gauge-wrap">
                    <div class="gauge-label-row">
                        <span>−{abs(gauge_min)}%</span>
                        <span>UNDER · FAIR · OVER (±{threshold:.0f}% band)</span>
                        <span>+{gauge_max}%</span>
                    </div>
                    <div class="gauge-track">
                        <div class="gauge-marker" style="left: calc({marker_pct}% - 1.5px);"></div>
                    </div>
                </div>

                <p class="footnote">
                    A listing is flagged Overpriced / Underpriced when it deviates more than
                    ±{threshold:.0f}% from the model's fair-price estimate. This model has an
                    R² of ~0.96 on held-out UK listings, scaled to Lebanese market pricing
                    (~1.75× import duty + VAT multiplier).
                </p>
            </div>
            """
        )

html_block(
    """
    <div class="page-footer">
        Trained on real UK used-car listings (Audi, BMW, Ford, Hyundai, Mercedes,
        Skoda, Toyota, Vauxhall, Volkswagen) · price-scaled to approximate the Lebanese market.
    </div>
    """
)