from flask import Flask, render_template_string, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import numpy as np
from collections import defaultdict
!pip install pyngrok
from pyngrok import ngrok

app = Flask(__name__)

SHL_API_URL = "https://api.shl.com/assessments" 
API_KEY = "your_shl_api_key_here"  # Should be stored securely in production

# Mock data for demonstration (same as before)
def get_shl_assessments():
    """Fetch SHL assessments from their API or use mock data"""
    return [
        {
            "id": "1",
            "name": "Verify Interactive",
            "description": "Cognitive ability test measuring verbal, numerical, and inductive reasoning",
            "skills": ["cognitive", "verbal", "numerical", "reasoning"],
            "job_level": ["entry", "mid"],
            "duration": 45,
            "popularity": 8.5
        },
        {
                "id": "2",
                "name": "OPQ32",
                "description": "Personality questionnaire measuring behavioral preferences at work",
                "skills": ["personality", "behavior", "work_preferences"],
                "job_level": ["all"],
                "duration": 30,
                "popularity": 9.0
            },
            {
                "id": "3",
                "name": "SJT Professional",
                "description": "Situational Judgment Test for professional roles",
                "skills": ["judgment", "decision_making", "professional_skills"],
                "job_level": ["mid", "senior"],
                "duration": 25,
                "popularity": 7.8
            },
            {
                "id": "4",
                "name": "Motivational Questionnaire",
                "description": "Measures what drives and motivates individuals at work",
                "skills": ["motivation", "drivers", "engagement"],
                "job_level": ["all"],
                "duration": 20,
                "popularity": 7.2
            },
            {
                "id": "5",
                "name": "Deductive Reasoning",
                "description": "Measures logical reasoning and problem-solving skills",
                "skills": ["cognitive", "logical_reasoning", "problem_solving"],
                "job_level": ["entry", "mid", "senior"],
                "duration": 35,
                "popularity": 8.1
            }
        ]

def preprocess_data(assessments):
    """Convert assessments data to a format suitable for recommendation"""
    processed = []
    for assessment in assessments:
        processed.append({
            "id": assessment["id"],
            "content": f"{assessment['name']} {assessment['description']} {' '.join(assessment['skills'])} {' '.join(assessment['job_level'])}"
        })
    return processed

# Recommendation engine
class SHLRecommender:
    def __init__(self):
        self.assessments = get_shl_assessments()
        self.processed_data = preprocess_data(self.assessments)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform([item['content'] for item in self.processed_data])

    def recommend_by_query(self, query, top_n=3):
        """Recommend assessments based on text query"""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_indices = similarities.argsort()[::-1][:top_n]

        recommendations = []
        for idx in related_indices:
          recommendations.append({
    "assessment": self.assessments[idx],  # Corrected to match variable name
    "similarity_score": float(similarities[idx])
          })


        return recommendations

    def recommend_by_job_level(self, job_level, top_n=3):
        """Recommend assessments based on job level"""
        matched = [a for a in self.assessments if job_level in a['job_level']]
        matched_sorted = sorted(matched, key=lambda x: x['popularity'], reverse=True)
        return matched_sorted[:top_n]

    def hybrid_recommendation(self, query=None, job_level=None, top_n=3):
        """Combine content-based and popularity-based recommendations"""
        if query and job_level:
            content_recs = self.recommend_by_query(query, top_n*2)
            level_recs = self.recommend_by_job_level(job_level, top_n*2)

            # Combine and deduplicate
            combined = content_recs + [{"assessment": a, "similarity_score": 0} for a in level_recs]
            unique_recs = {}
            for rec in combined:
                aid = rec['assessment']['id']
                if aid not in unique_recs:
                    unique_recs[aid] = rec
                else:
                    # Boost score if present in both
                    unique_recs[aid]['similarity_score'] += 0.5

            # Sort by combined score
            sorted_recs = sorted(unique_recs.values(),
                               key=lambda x: (x['similarity_score'], x['assessment']['popularity']),
                               reverse=True)
            return [r['assessment'] for r in sorted_recs[:top_n]]

        elif query:
            return [r['assessment'] for r in self.recommend_by_query(query, top_n)]
        elif job_level:
            return self.recommend_by_job_level(job_level, top_n)
        else:
            return sorted(self.assessments, key=lambda x: x['popularity'], reverse=True)[:top_n]

# Evaluation metrics 
class EvaluationMetrics:
    @staticmethod
    def precision_at_k(recommendations, relevant_items, k):
        """Calculate precision at K"""
        top_k = recommendations[:k]
        relevant_set = set(relevant_items)
        relevant_and_retrieved = [item for item in top_k if item['id'] in relevant_set]
        return len(relevant_and_retrieved) / k

    @staticmethod
    def recall_at_k(recommendations, relevant_items, k):
        """Calculate recall at K"""
        top_k = recommendations[:k]
        relevant_set = set(relevant_items)
        relevant_and_retrieved = [item for item in top_k if item['id'] in relevant_set]
        return len(relevant_and_retrieved) / len(relevant_items) if relevant_items else 0

    @staticmethod
    def mean_average_precision(recommendations, relevant_items):
        """Calculate mean average precision"""
        relevant_set = set(relevant_items)
        average_precision = 0.0
        num_relevant = 0

        for k in range(1, len(recommendations)+1):
            if recommendations[k-1]['id'] in relevant_set:
                num_relevant += 1
                precision_at_k = num_relevant / k
                average_precision += precision_at_k

        return average_precision / len(relevant_items) if relevant_items else 0

    @staticmethod
    
    def evaluate(recommender, test_cases):
        """Evaluate the recommender system with multiple test cases"""
        results = {
            'precision@3': [],
            'recall@3': [],
            'map': []
        }

        for case in test_cases:
            query = case.get('query')
            job_level = case.get('job_level')
            relevant_ids = case['relevant_ids']

            recommendations = recommender.hybrid_recommendation(query, job_level, top_n=5)
            count = 0
            recall_score = count/len(relevant_ids)

            results['precision@3'].append(
                EvaluationMetrics.precision_at_k(recommendations, relevant_ids, 3)
            )
            results['recall@3'].append(
                EvaluationMetrics.recall_at_k(recommendations, relevant_ids, 3)
            )
            results['map'].append(
                EvaluationMetrics.mean_average_precision(recommendations, relevant_ids)
            )

        # Calculate averages
        avg_metrics = {
            'avg_precision@3': sum(results['precision@3']) / len(results['precision@3']),
            'avg_recall@3': sum(results['recall@3']) / len(results['recall@3']),
            'avg_map': sum(results['map']) / len(results['map'])
        }

        return avg_metrics

# Create test cases for evaluation
def create_test_cases(assessments):
    """Create test cases for evaluation based on assessment data"""
    # In a real scenario, these would come from user testing or historical data
    test_cases = [
        {
            "query": "cognitive reasoning test",
            "job_level": "entry",
            "relevant_ids": ["1", "5"]  # Verify Interactive and Deductive Reasoning
        },
        {
            "query": "personality questionnaire",
            "job_level": "all",
            "relevant_ids": ["2", "4"]  # OPQ32 and Motivational Questionnaire
        },
        {
            "query": "professional skills assessment",
            "job_level": "senior",
            "relevant_ids": ["3"]  # SJT Professional
        },
        {
            "query": "logical reasoning",
            "job_level": "mid",
            "relevant_ids": ["1", "5"],  # Verify Interactive and Deductive Reasoning
            "description": "Test for logical reasoning assessments"
        },
        {
            "query": "work motivation",
            "job_level": "all",
            "relevant_ids": ["4"],  # Motivational Questionnaire
            "description": "Test for motivation assessments"
        }
    ]


    return test_cases
