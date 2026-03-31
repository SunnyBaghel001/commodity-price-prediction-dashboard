import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ------------------ CONFIG ------------------

st.set_page_config(page_title="Commodity Intelligence", layout="wide")

# ------------------ CSS ------------------

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #FFFFFF;
}
.stTabs [role="tab"] {
    font-size: 18px;
    padding: 10px;
}
.metric-card {
    background-color: #1E222A;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------

@st.cache_data
def load_data():
    return pd.read_csv("final_cleaned_data.csv")

df = load_data()

# ------------------ LOAD MODEL ------------------

@st.cache_resource
def load_model():
    return joblib.load("price_model_compressed.pkl")

model = load_model()
le_commodity = joblib.load("le_commodity.pkl")
le_state = joblib.load("le_state.pkl")
le_season = joblib.load("le_season.pkl")
commodity_median_map = joblib.load("commodity_median.pkl")
state_median_map = joblib.load("state_median.pkl")

# ------------------ TITLE ------------------

st.title("📊 Commodity Price Analytics Dashboard")
st.caption("AI-powered price prediction & market insights")

# ------------------ SIDEBAR ------------------

st.sidebar.header("🔧 Selection")

commodity = st.sidebar.selectbox("Commodity", le_commodity.classes_)
state = st.sidebar.selectbox("State", le_state.classes_)
month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.slider("Day", 1, 31, 15)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info("""
ML-powered commodity price prediction system.

Model:
- Random Forest / XGBoost  
Accuracy: ~76% R²
""")

# ------------------ FILTER DATA ------------------

filtered_df = df[(df['Commodity'] == commodity) & (df['State'] == state)]

if filtered_df.empty:
    st.warning("⚠️ No exact match found — showing nearest available data")

    # fallback: use only commodity
    filtered_df = df[df['Commodity'] == commodity]

    if filtered_df.empty:
        # fallback: use full dataset
        filtered_df = df
# ------------------ MIN/MAX FROM DATA ------------------

min_price = int(filtered_df['Min_Price'].median())
max_price = int(filtered_df['Max_Price'].median())
price_spread = max_price - min_price

# ------------------ FEATURES ------------------

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    else:
        return 'Monsoon'

season = get_season(month)

commodity_enc = le_commodity.transform([commodity])[0]
state_enc = le_state.transform([state])[0]
season_enc = le_season.transform([season])[0]

commodity_median = commodity_median_map.get(commodity_enc, df['Modal_Price'].median())
state_median = state_median_map.get(state_enc, df['Modal_Price'].median())

# ------------------ PREDICTION ------------------

input_data = pd.DataFrame([{
    'Commodity': commodity_enc,
    'State': state_enc,
    'Month': month,
    'Day': day,
    'Season': season_enc,
    'Price_Spread': price_spread,
    'Commodity_Median_Price': commodity_median,
    'State_Median_Price': state_median
}])

with st.spinner("🔮 Predicting price..."):
    prediction = model.predict(input_data)[0]
    per_kg = prediction / 100

# ------------------ PREDICTION DISPLAY ------------------

st.markdown("### 💰 Predicted Modal Price")

st.markdown(f"""
<div style='text-align:center;
            background: linear-gradient(135deg, #1f4037, #2c7744);
            padding:25px;
            border-radius:15px;
            margin-bottom:20px;'>
    <h1 style='color:white;'>₹ {round(prediction,2)}</h1>
    <p style='color:#dcdcdc;'>per quintal</p>
    <p style='color:#a0f0d0;'>≈ ₹ {round(per_kg,2)} per kg</p>
</div>
""", unsafe_allow_html=True)

# ------------------ SMART INSIGHTS ------------------

if prediction < min_price:
    st.info("📉 Price is expected to be lower than usual market range.")
elif prediction > max_price:
    st.warning("📈 Price may increase beyond current market trends.")
else:
    st.success("✅ Price is within expected market range.")

# ------------------ KPI CARDS ------------------

col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='metric-card'>Min Price<br><b>₹ {min_price}</b></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'>Max Price<br><b>₹ {max_price}</b></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'>Spread<br><b>₹ {price_spread}</b></div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ MARKET SUMMARY ------------------

st.markdown("### 📊 Market Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Modal Price", f"₹ {int(df['Modal_Price'].mean())}")
col2.metric("Max Observed Price", f"₹ {int(df['Modal_Price'].max())}")
col3.metric("Min Observed Price", f"₹ {int(df['Modal_Price'].min())}")

# ------------------ TABS ------------------

tab1, tab2, tab3 = st.tabs(["📈 Trends", "📍 States", "🥕 Commodities"])

# ------------------ TREND ------------------

with tab1:
    st.subheader(f"Monthly Trend — {commodity} in {state}")

    monthly = filtered_df.groupby('Month')['Modal_Price'].mean().reset_index()
    monthly = monthly.sort_values('Month')

    fig = px.line(monthly, x='Month', y='Modal_Price',
                  markers=True, template="plotly_dark")

    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

# ------------------ STATE ------------------

with tab2:
    st.subheader(f"Top States for {commodity}")

    state_data = df[df['Commodity'] == commodity]
    state_data = state_data.groupby('State')['Modal_Price'].mean().reset_index()
    state_data = state_data.sort_values(by='Modal_Price', ascending=False).head(8)

    fig = px.bar(state_data, x='State', y='Modal_Price',
                 color='Modal_Price', template="plotly_dark")

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ------------------ COMMODITY ------------------

with tab3:
    st.subheader("Top Commodities (Overall)")

    commodity_data = df.groupby('Commodity')['Modal_Price'].mean().reset_index()
    commodity_data = commodity_data.sort_values(by='Modal_Price', ascending=False).head(8)

    fig = px.bar(commodity_data, x='Commodity', y='Modal_Price',
                 color='Modal_Price', template="plotly_dark")

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ------------------ DOWNLOAD ------------------

st.download_button(
    label="📥 Download Dataset",
    data=df.to_csv(index=False),
    file_name="commodity_data.csv",
    mime="text/csv"
)

# ------------------ DATA VIEW ------------------

with st.expander("🔍 View Dataset"):
    st.dataframe(df.head(100))

# ------------------ FOOTER ------------------

st.markdown("---")
st.caption("🚀 ML-powered commodity pricing system | Dynamic analytics | R² ≈ 0.76")
