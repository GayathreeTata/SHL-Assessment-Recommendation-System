import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from PIL import Image

# --- SETTINGS ---
st.set_page_config(
    page_title="🔍 SHL Assessment Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    return pd.read_csv("SHL_catalog.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- STYLING ---
def styled_card(title, skills, test_type, description, score, url):
    emoji = {
        "Coding": "💻", "Cognitive": "🧠", "Personality": "😊",
        "Communication": "🗣️", "Aptitude": "📝"
    }.get(test_type, "📋")
    
    return f"""
    <div style="
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: {'#2d3741' if st.get_option('theme.base') == 'dark' else '#f0f2f6'};
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    ">
        <h3>{emoji} {title}</h3>
        <p><b>Skills:</b> {skills}</p>
        <p><b>Type:</b> {test_type} {emoji}</p>
        <p><b>Description:</b> {description[:150]}...</p>
        <p><b>🔗 URL:</b> <a href="{url}" target="_blank">View Assessment</a></p>
        <div style="background: #e0e0e0; border-radius: 5px; height: 10px;">
            <div style="background: #4CAF50; width: {score*100}%; height: 10px; border-radius: 5px;"></div>
        </div>
        <p><b>Match:</b> {score:.0%}</p>
    </div>
    """

# --- MAIN APP ---
def main():
    # --- HEADER ---
    st.title("🔍 SHL Assessment Recommender")
    st.markdown("""
    <style>
    .big-font { font-size:18px !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Find the best SHL assessments for your job needs!</p>', unsafe_allow_html=True)

    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.header("⚙️ Filters")
        min_duration = st.slider("Minimum Duration (mins)", 10, 60, 20)
        remote_only = st.checkbox("Remote Testing Only", True)

    # --- LOAD DATA ---
    catalog_df = load_data()
    catalog_df['combined'] = catalog_df.apply(
        lambda row: f"{row['Assessment Name']} {row['Skills']} {row['Description']}", axis=1
    )
    model = load_model()
    corpus_embeddings = model.encode(catalog_df['combined'].tolist(), convert_to_tensor=True)

    # --- USER INPUT ---
    user_query = st.text_input(
        "🔎 Describe your job role or required skills:",
        placeholder="e.g., 'Python developer with SQL experience'"
    )

    if user_query:
        # --- SEARCH ---
        with st.spinner("🔍 Finding best matches..."):
            query_embedding = model.encode(user_query, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_k = min(5, len(catalog_df))
            top_indices = torch.topk(cosine_scores, k=top_k).indices

        # --- DISPLAY RESULTS ---
        st.subheader("🎯 Top Recommendations")
        for idx in top_indices:
            idx = idx.item()
            row = catalog_df.iloc[idx]
            
            # Skip if doesn't match filters
            if remote_only and row['Remote Testing Support'] != 'Yes':
                continue
            if row['Duration'] < min_duration:
                continue
            
            # Display styled card
            st.markdown(
                styled_card(
                    row['Assessment Name'],
                    row['Skills'],
                    row['Test Type'],
                    row['Description'],
                    cosine_scores[idx].item(),
                    row['URL']
                ),
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
