# SHL-Assessment-Recommendation-System
This challenge is an AI-powered advice device constructed to signify the maximum applicable SHL exams primarily based totally on person queries, task descriptions, or unstructured enter data. It combines NLP strategies and Large Language Models (LLMs) to offer accurate, efficient, and context-conscious consequences via an intuitive frontend interface.

**Feautures** :
Recommends SHL assessments based on:
Job descriptions
Unstructured URLs or text input
Custom user queries
NLP-based semantic similarity matching using Sentence-BERT
Contextual feature extraction and filtering using Gemini 1.5 Pro (LLM)
Ranking and scoring using cosine similarity
Top recommendations output with relevance filtering
Streamlit-based frontend for an interactive user experience


Tech Stack 🛠️
Natural Language Processing:
Sentence-BERT for growing embeddings of tests and queries
Cosine similarity for rating the maximum applicable tests
LLM Integration:
Gemini 1.five Pro used to extract dependent features (task title, skills, period, etc.) from unstructured input
Post-processing and filtering of tips primarily based totally on constraints like period and talent match
Frontend:
Streamlit software for user-pleasant interplay and show of results

**Working**
Data Analysis and processing:

A mock dataset of fifty SHL-like checks is used, every containing:
Assessment name, URL, duration, take a look at type, skills, description, far flung support, and adaptive/IRT support.
A "combined" column is created through concatenating all columns right into a unmarried string for embedding.

NLP Embedding and Retrieval:

Sentence-BERT is used to transform each dataset entries and enter queries into vector embeddings.
Cosine similarity is calculated to perceive the pinnacle matching checks.

LLM  AI Enhancement (Gemini 1.five Pro):

Accepts task descriptions, URLs, or unstructured queries.
Extracts significant dependent functions like task role, required skills, anticipated duration, etc.
This facts is used to generate greater correct embeddings.
After retrieving pinnacle candidates, Gemini re-filters primarily based totally on constraints and relevance.

Evaluation Metrics:

Performance is evaluated the use of metrics which includes Recall@five and MAP@five.
The hybrid (NLP + LLM) technique outperforms the natural NLP baseline in each metrics.
Streamlit Interface:

Users can enter queries immediately in an internet interface.
Receives and shows the pinnacle encouraged checks at the side of their details.


* Evaluation Metrics 📊
NLP Model - Recall@5 = 0.85 and MAP@5 = 0.71
NLP + LLM Model - Recall@5 = 1.0 and MAP@5 = 1.000




  
