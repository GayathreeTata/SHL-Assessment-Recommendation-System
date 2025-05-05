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

# Main Streamlit app
def main():
    st.title("💡SHL Assessment Recommendation System")
    
    # Load data and model
    catalog_df = load_data()
    catalog_df['combined'] = catalog_df.apply(combine_row, axis=1)
    model = load_model()
    
    # Generate embeddings for the catalog
    corpus = catalog_df['combined'].tolist()
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    # User input
    user_query = st.text_input(" Enter your search query here:🔍")
    
    if user_query:
        # Encode the query
        query_embedding = model.encode(user_query, convert_to_tensor=True)
        
        # Calculate cosine similarities
        cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_k = min(5, len(corpus))
        top_results = torch.topk(cosine_scores, k=top_k)
        
        # Display results
        st.subheader("🎯Top 5 Matching Assessments:")
        
        for score, idx in zip(top_results[0], top_results[1]):
            idx = idx.item()
            assessment = catalog_df.iloc[idx]
            
            st.write(f"*Assessment:* {assessment['Assessment Name']}")
            st.write(f"*Skills:* {assessment['Skills']}")
            st.write(f"*Test Type:* {assessment['Test Type']}")
            st.write(f"*Description:* {assessment['Description']}")
            st.write(f"*Remote Testing Support:* {assessment['Remote Testing Support']}")
            st.write(f"*Adaptive/IRT:* {assessment['Adaptive/IRT']}")
            st.write(f"*Duration:* {assessment['Duration']} mins")
            st.write(f"*URL:* {assessment['URL']}")
            st.write(f"*Similarity Score:* {score.item():.4f}")
            st.write("---")

if _name_ == "_main_":
    main()
