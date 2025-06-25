
import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open('sonar_model.sav', 'rb'))

# Page configuration
st.set_page_config(page_title="Sonar Rock vs Mine", page_icon="🌊", layout="wide")

# Custom CSS for background and styling
page_bg_img = """
<style>
section[data-testid="stSidebar"] {
    overflow-y: auto;
    max-height: 100vh;
    padding-bottom: 2rem;
}
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1629904853716-f0bc54eea481");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-height: 100vh;
    position: relative;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(255,255,255,0.3); /* 💡 Light overlay for readability */
    z-index: -1;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
.block-container {
    background: #1e1e2f;
    color: #ffffff;
    border-radius: 1rem;
    padding: 2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    margin-top: 0;
    margin-bottom: 2rem;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4461/4461634.png", width=100)
    st.title("🧪 Enter Sonar Features")
    st.markdown("Please enter **60 numeric values** from the sonar signal.")

    input_data = []
    for i in range(60):
        val = st.number_input(f"Feature {i+1}", key=f"input_{i}", step=0.01, format="%.4f")
        input_data.append(val)

# Main content
st.title("🌊 Sonar Signal Object Detection")
st.markdown("This ML model classifies an object detected by sonar as either a **Rock 🪨** or a **Mine 🚢**.")

# Prediction
if st.sidebar.button("🔍 Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)

    st.subheader("🎯 Prediction Result")
    if prediction == 'R':
        st.markdown("""
        <div style='background-color:#2a2a3b; color:#d9e6ff; padding: 16px; border-radius: 10px; font-size: 18px;'>
        ℹ️ <b>This object is a Rock 🪨</b>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='background-color:#3b2a2a; color:#ffdddd; padding: 16px; border-radius: 10px; font-size: 18px;'>
    🚨 <b>This object is a Mine 🚢</b>
    </div>
    """, unsafe_allow_html=True)
