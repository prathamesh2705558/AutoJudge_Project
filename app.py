# ==========================================
#  REQUIREMENT 5: WEB UI CODE (STREAMLIT)
# ==========================================
import streamlit as st
import joblib
import re
import numpy as np

# Load the trained artifacts
clf = joblib.load('model_class.pkl')
reg = joblib.load('model_score.pkl')
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="AutoJudge Pro", page_icon="🏆")

st.title("🏆 CP Problem Predictor")
st.markdown("Predicts **Codeforces Ratings** and Difficulty.")

# UI: Input Section
col1, col2 = st.columns(2)
with col1:
    desc = st.text_area("Problem Description", height=200, placeholder="Paste description...")
with col2:
    inp_desc = st.text_area("Input Description", height=90, placeholder="e.g. integer N...")
    out_desc = st.text_area("Output Description", height=90, placeholder="e.g. print sum...")

if st.button("Analyze Problem", type="primary"):
    if desc:
        # Preprocessing (must match training)
        full_text = desc + " " + inp_desc + " " + out_desc
        clean = re.sub(r'<.*?>', '', full_text.lower())
        
        # Feature Extraction
        text_features = vectorizer.transform([clean]).toarray()
        len_feature = scaler.transform([[len(clean.split())]])
        final_features = np.hstack((text_features, len_feature))
        
        # Inference
        raw_class = clf.predict(final_features)[0]
        raw_score = reg.predict(final_features)[0]
        
        # Post-Processing (Safety Logic)
        pred_class = str(raw_class).strip().title()
        final_rating = int(round(raw_score / 100) * 100)
        
        # UI Coloring Logic
        label_color = "#000000"
        if pred_class == "Easy":
            final_rating = np.clip(final_rating, 800, 1000)
            label_color = "#008000"
        elif pred_class == "Medium":
            final_rating = np.clip(final_rating, 1100, 1500)
            label_color = "#DAA520"
        else: # Hard
            final_rating = np.clip(final_rating, 1600, 3000)
            label_color = "#FF0000"

        # UI: Display Results
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("### Difficulty")
            st.markdown(f"<h2 style='color: {label_color};'>{pred_class}</h2>", unsafe_allow_html=True)
        with m2:
            st.markdown("### Predicted Rating")
            st.markdown(f"<h2 style='color: #1E90FF;'>{final_rating}</h2>", unsafe_allow_html=True)
            st.progress(min(final_rating / 3500, 1.0))
    else:
        st.warning("Please problem text first.")