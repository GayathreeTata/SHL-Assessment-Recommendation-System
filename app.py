import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# --- SETUP ---
st.set_page_config(
    page_title="🌟 SHL Assessment Genius",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("SHL_catalog.csv")
    # Add mock columns for demo (replace with real data)
    df["Job Type"] = np.random.choice(["Remote", "Hybrid", "Full-time"], size=len(df))
    df["Location"] = np.random.choice(["US", "UK", "India", "Global"], size=len(df))
    df["Experience"] = np.random.choice(["Fresher", "Early Pro", "Professional"], size=len(df))
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- STYLED CARD ---
def assessment_card(title, skills, test_type, score, url, job_type, location, exp_level):
    color_map = {
        "Remote": "#FF6B6B", 
        "Hybrid": "#4ECDC4", 
        "Full-time": "#FFD166"
    }
    exp_color = {
        "Fresher": "#06D6A0",
        "Early Pro": "#118AB2",
        "Professional": "#073B4C"
    }
    
    return f"""
    <div style="
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        background: linear-gradient(145deg, {color_map[job_type]} 0%, #f8f9fa 100%);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        border-left: 5px solid {exp_color[exp_level]};
    ">
        <div style="display: flex; justify-content: space-between;">
            <h3 style="color: #2b2d42; margin: 0;">{title}</h3>
            <span style="background: {exp_color[exp_level]}; 
                     color: white; 
                     padding: 3px 10px; 
                     border-radius: 20px;
                     font-size: 12px;">
                {exp_level}
            </span>
        </div>
        <p style="color: #6c757d;"><b>📍 {location}</b> | 🕒 {job_type}</p>
        <p><b>🔧 Skills:</b> {skills}</p>
        <p><b>📝 Test Type:</b> {test_type}</p>
        <div style="background: #e9ecef; border-radius: 10px; height: 10px; margin: 10px 0;">
            <div style="background: linear-gradient(90deg, #FF9A8B 0%, #FF6B6B 100%); 
                        width: {score*100}%; 
                        height: 10px; 
                        border-radius: 10px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <a href="{url}" target="_blank" style="
                background: #4361ee;
                color: white;
                padding: 8px 15px;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
            ">View Assessment</a>
            <span style="font-weight: bold; color: #2b2d42;">Match: {score:.0%}</span>
        </div>
    </div>
    """

# --- MAIN APP ---
def main():
    # --- HEADER ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');
    .header {
        font-family: 'Poppins', sans-serif;
        color: #4361ee;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    <h1 class="header">🌟 SHL Assessment Genius</h1>
    """, unsafe_allow_html=True)

    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        
        # Job Type Filter
        job_types = st.multiselect(
            "Job Type:",
            options=["Remote", "Hybrid", "Full-time"],
            default=["Remote", "Hybrid"]
        )
        
        # Location Filter
        locations = st.multiselect(
            "Location:",
            options=["US", "UK", "India", "Global"],
            default=["Global"]
        )
        
        # Experience Filter
        experience = st.multiselect(
            "Experience Level:",
            options=["Fresher", "Early Pro", "Professional"],
            default=["Early Pro", "Professional"]
        )
        
        # Duration Slider
        min_duration = st.slider(
            "⏳ Minimum Duration (mins):",
            10, 60, 30
        )

    # --- LOAD DATA ---
    catalog_df = load_data()
    model = load_model()
    corpus_embeddings = model.encode(
        catalog_df['Assessment Name'] + " " + 
        catalog_df['Skills'] + " " + 
        catalog_df['Description'],
        convert_to_tensor=True
    )

    # --- SEARCH BAR ---
    user_query = st.text_input(
        "🔎 Describe your job role or skills needed:",
        placeholder="E.g.: 'Data analyst with SQL experience for remote work'"
    )

    if user_query:
        with st.spinner("✨ Finding your perfect assessments..."):
            query_embedding = model.encode(user_query, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            catalog_df["Match Score"] = cosine_scores.numpy()
            
            # Apply filters
            filtered_df = catalog_df[
                (catalog_df["Job Type"].isin(job_types)) &
                (catalog_df["Location"].isin(locations)) &
                (catalog_df["Experience"].isin(experience)) &
                (catalog_df["Duration"] >= min_duration)
            ].sort_values("Match Score", ascending=False).head(5)

        # --- RESULTS ---
        st.markdown(f"### 🎯 Top {len(filtered_df)} Matches")
        if filtered_df.empty:
            st.warning("No assessments match your filters. Try broadening your search!")
        else:
            for _, row in filtered_df.iterrows():
                st.markdown(
                    assessment_card(
                        title=row["Assessment Name"],
                        skills=row["Skills"],
                        test_type=row["Test Type"],
                        score=row["Match Score"],
                        url=row["URL"],
                        job_type=row["Job Type"],
                        location=row["Location"],
                        exp_level=row["Experience"]
                    ),
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
