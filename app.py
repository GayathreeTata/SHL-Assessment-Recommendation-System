# ──────────────────────────────────────────────────────────────────────────────
# File: app.py
# ──────────────────────────────────────────────────────────────────────────────

# 1) VERY FIRST THINGS: create and set a brand‑new event loop
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

# 2) Now import all your normal libraries
import streamlit as st
import pandas as pd
from query_functions import query_handling_using_LLM_updated

# 3) Streamlit page config & header
st.set_page_config(page_title="SHL Assessment Recommendation System", layout="centered")
st.markdown(
    """
    <h1 style='text-align:center;'>🧠 SHL Assessment Recommendation System</h1>
    <p style='text-align:center;color:#888;'>Find the best assessments based on your query using AI!</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# 4) The input widget
query = st.text_input("🔍 Enter your search query here:", placeholder="e.g. Python SQL coding test")

# 5) The “Search” button & results
if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("🤖 Thinking... Fetching the best matches for you!"):
            try:
                # <-- THIS CALL WILL NOW ALWAYS SEE an event loop
                df = query_handling_using_LLM_updated(query)

                # Your existing DataFrame cleanup & display logic
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if "Score" in df.columns:
                        df = df.drop(columns=["Score"])
                    if "Duration" in df.columns:
                        df = df.rename(columns={"Duration": "Duration in mins"})
                    display_cols = [
                        "Assessment Name","Skills","Test Type","Description",
                        "Remote Testing Support","Adaptive/IRT","Duration in mins","URL"
                    ]
                    df = df[[c for c in display_cols if c in df.columns]]
                    df["URL"] = df["URL"].apply(
                        lambda x: f"<a href='{x}' target='_blank'>🔗 View</a>" if pd.notna(x) else ""
                    )

                    # Render styled HTML table
                    table_html = "<table>…</table>"  # your existing HTML builder
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.warning("😕 No assessments matched your query. Try rephrasing it!")
            except Exception as e:
                st.error(f"🚨 Something went wrong: {e}")
