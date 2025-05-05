import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import asyncio

# Set up asyncio event loop policy for Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Load the mock dataset
@st.cache_data
def load_data():
    catalog_df = pd.read_csv("SHL_catalog.csv")
    return catalog_df

# Combine row features into a single string
def combine_row(row):
    parts = [
        str(row["Assessment Name"]),
        str(row["Duration"]),
        str(row["Remote Testing Support"]),
        str(row["Adaptive/IRT"]),
        str(row["Test Type"]),
        str(row["Skills"]),
        str(row["Description"]),
    ]
    return ' '.join(parts)

# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Function to get icon based on test type
def get_test_type_icon(test_type):
    icons = {
        "Cognitive": "🧠",
        "Personality": "👤",
        "Skill": "📚",
        "Behavioral": "💡",
        "Verbal": "💬",
        "Numerical": "🔢",
        "Logical": "🧩"
    }
    return icons.get(test_type.split()[0], "📝")

# Function to display colored assessment card
def display_assessment_card(assessment, score):
    # Get icons
    test_type_icon = get_test_type_icon(assessment['Test Type'])
    remote_icon = "🌐" if assessment['Remote Testing Support'] == 'Yes' else "🏢"
    adaptive_icon = "🔄" if assessment['Adaptive/IRT'] == 'Yes' else "➡️"
    duration_icon = "⏱️"
    skill_icon = "🛠️"
    
    # Create a container with colored border
    with st.container():
        # Header with assessment name and score
        st.markdown(
            f"""
            <div style="
                background-color: #f0f2f6;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #4e79a7;
                margin-bottom: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            ">
                <h3 style="color: #2c3e50; margin:0;">{test_type_icon} {assessment['Assessment Name']}</h3>
                <p style="color: #7f8c8d; margin:0;">⭐ Match score: <span style="color: #e74c3c; font-weight:bold;">{score:.2f}</span></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Assessment details in a two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                <div style="margin-bottom: 15px;">
                    <p style="margin:5px 0;"><span style="color: #3498db; font-weight:bold;">{skill_icon} Skills:</span> {assessment['Skills']}</p>
                    <p style="margin:5px 0;"><span style="color: #3498db; font-weight:bold;">{test_type_icon} Test Type:</span> {assessment['Test Type']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div style="margin-bottom: 15px;">
                    <p style="margin:5px 0;"><span style="color: #3498db; font-weight:bold;">{remote_icon} Remote Testing:</span> <span style="color: {'#27ae60' if assessment['Remote Testing Support'] == 'Yes' else '#e74c3c'}">{assessment['Remote Testing Support']}</span></p>
                    <p style="margin:5px 0;"><span style="color: #3498db; font-weight:bold;">{adaptive_icon} Adaptive/IRT:</span> <span style="color: {'#27ae60' if assessment['Adaptive/IRT'] == 'Yes' else '#e74c3c'}">{assessment['Adaptive/IRT']}</span></p>
                    <p style="margin:5px 0;"><span style="color: #3498db; font-weight:bold;">{duration_icon} Duration:</span> <span style="color: #27ae60;">{assessment['Duration']} mins</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Description in an expandable section
        with st.expander("📝 Description", expanded=False):
            st.markdown(
                f"""
                <div style="
                    background-color: #f9f9f9;
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 3px solid #9b59b6;
                    margin: 10px 0;
                ">
                    <p style="margin:0; color: #34495e;">{assessment['Description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # URL with clickable link
        st.markdown(
            f"""
            <div style="margin-top: 10px;">
                <a href="{assessment['URL']}" target="_blank" style="
                    background-color: #3498db;
                    color: white;
                    padding: 8px 15px;
                    text-decoration: none;
                    border-radius: 5px;
                    display: inline-block;
                    font-weight: bold;
                ">
                🔗 View Assessment Details
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")

# Main Streamlit app
def main():
    # Custom CSS for the app
    st.markdown(
        """
        <style>
            .stTextInput input, .stTextArea textarea {
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 12px;
            }
            .stButton button {
                background-color: #3498db;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                border: none;
                font-weight: bold;
            }
            .stButton button:hover {
                background-color: #2980b9;
                color: white;
            }
            [data-testid="stExpander"] .st-emotion-cache-1qrv4ga {
                font-weight: bold;
                font-size: 1.1rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🔍 SHL Assessment Recommendation System")
    st.markdown("Find the best SHL assessments for your job requirements")
    
    # Load data and model
    catalog_df = load_data()
    catalog_df['combined'] = catalog_df.apply(combine_row, axis=1)
    model = load_model()
    corpus = catalog_df['combined'].tolist()
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    # User input section with a nice header
    with st.container():
        st.subheader("📝 Enter Your Job Description")
        user_query = st.text_area(
            "Describe the job role, required skills, or assessment needs:",
            height=150,
            placeholder="Example: We need cognitive ability tests for graduate hiring that can be administered remotely and take less than 30 minutes..."
        )
    
    if user_query:
        # Encode the query
        query_embedding = model.encode(user_query, convert_to_tensor=True)
        
        # Calculate cosine similarities
        cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_k = min(5, len(corpus))
        top_results = torch.topk(cosine_scores, k=top_k)
        
        # Display results with a nice header
        st.subheader("🎯 Top Matching Assessments")
        st.markdown(f"Found {len(top_results[0])} assessments that match your requirements:")
        
        # Progress bar for processing
        with st.spinner('🔎 Finding the best assessments for you...'):
            # Display each assessment as a card
            for score, idx in zip(top_results[0], top_results[1]):
                idx = idx.item()
                assessment = catalog_df.iloc[idx]
                display_assessment_card(assessment, score.item())

if __name__ == "__main__":
    main()
