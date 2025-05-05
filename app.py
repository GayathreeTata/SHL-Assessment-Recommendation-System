import streamlit as st
import pandas as pd
import asyncio
from query_functions import query_handling_using_LLM_updated

st.set_page_config(page_title="SHL Assessment Recommendation System", layout="centered")
st.markdown(
    """
    <h1 style='text-align:center;'>🧠 SHL Assessment Recommendation System</h1>
    <p style='text-align:center;color:#888;'>Find the best assessments based on your query using AI!</p>
    <hr>
    """,
    unsafe_allow_html=True
)

query = st.text_input("🔍 Enter your search query here:", placeholder="e.g. Python SQL coding test")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("🤖 Thinking... Fetching the best matches for you!"):
            try:
                # Safely wrap LLM call with isolated event loop
                def run_query():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(query_handling_using_LLM_updated(query))

                df = run_query()

                if isinstance(df, pd.DataFrame) and not df.empty:
                    if "Score" in df.columns:
                        df = df.drop(columns=["Score"])
                    if "Duration" in df.columns:
                        df = df.rename(columns={"Duration": "Duration in mins"})
                    display_cols = [
                        "Assessment Name", "Skills", "Test Type", "Description",
                        "Remote Testing Support", "Adaptive/IRT", "Duration in mins", "URL"
                    ]
                    df = df[[c for c in display_cols if c in df.columns]]
                    df["URL"] = df["URL"].apply(
                        lambda x: f"<a href='{x}' target='_blank'>🔗 View</a>" if pd.notna(x) else ""
                    )
                    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.warning("😕 No assessments matched your query. Try rephrasing it!")
            except Exception as e:
                st.error(f"🚨 Something went wrong: {e}")
