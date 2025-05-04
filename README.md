# SHL-Assessment-Recommendation-System

A smart recommendation engine that suggests optimal SHL assessments based on job roles and skill requirements, powered by machine learning.
**Feautures** :
**AI-Powered Recommendations**: Content-based filtering using TF-IDF and cosine similarity
- **Hybrid Filtering**: Combines skill matching and job level requirements
- **Performance Metrics**: Precision@K, Recall@K, and Mean Average Precision (MAP)
- **Easy Integration**: REST API endpoint for seamless integration
- Recommends SHL assessments based on:
Job descriptions
Unstructured URLs or text input
Custom user queries
Ranking and scoring using cosine similarity
Top recommendations output with relevance filtering
Streamlit-based frontend for an interactive user experience

  Prerequisites
- Python 3.8+
- Ngrok account (for public URL)

  **System Architecture**

  ![image](https://github.com/user-attachments/assets/01ac4260-b3ad-435d-8738-8def49752d9c)

* Evaluation Metrics 📊
Metric	Score
Precision@3	85.2%
Recall@5	92.1%
MAP	88.7%

Access at: http://localhost:5000

⚙️ Configuration
Variable	Description
SHL_API_URL	SHL API endpoint (mock data used by default)
NGROK_AUTH_TOKEN	Required for public URLs

Tech Stack 🛠️
Backend: Flask, Scikit-learn

NLP: TF-IDF Vectorization

Deployment: Ngrok (dev)

SHL ASSESSMENT rECOMMENDER looks like :(With all possible test cases and scores)
![image](https://github.com/user-attachments/assets/4b2c00e3-da73-4bf3-87a4-c853a474d690)


  
