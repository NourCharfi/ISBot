import os
import requests
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from whoosh.qparser import QueryParser
from chatbot.data_processing import ix, responses, urls, preprocess_text, vectorizer, tfidf_matrix, file_paths
from chatbot.models import nb_classifier, knn_classifier
from chatbot.config import shortcuts, shortcut_urls
from chatbot.embeddings_utils import get_best_match_with_fasttext
from dotenv import load_dotenv

# Load environment variables for API key
load_dotenv()
API_KEY = os.getenv('OPENROUTER_API_KEY')
API_URL = 'https://openrouter.ai/api/v1/chat/completions'

def call_openrouter_api(query):
    """Call OpenRouter API to generate a response."""
    try:
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost:8080',
            'X-Title': 'ISBOT'
        }
        payload = {
            'model': 'meta-llama/llama-3.1-8b-instruct:free',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': query}
            ]
        }
        response = requests.post(API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        answer = data['choices'][0]['message']['content'].strip()
        return answer
    except requests.RequestException as e:
        print(f"OpenRouter API request failed: {e}")
        return "Sorry, I could not connect to the OpenRouter API."

def search_in_index(query):
    """Search the Whoosh index for a matching question."""
    with ix.searcher() as searcher:
        query_obj = QueryParser("question", ix.schema).parse(query)
        results = searcher.search(query_obj, limit=1)
        return {"answer": results[0]['answer'], "url": results[0]['url']} if results else None

def get_shortcut_url(shortcut):
    """Get the URL for a shortcut command."""
    path = shortcut_urls.get(shortcut)
    return f"https://isetsf.rnu.tn{path}" if path else None

def check_new_questions(user_input, user_id):
    """Check if the question exists in new_questions.json for the user."""
    try:
        if not os.path.exists('data/new_questions.json'):
            return None
        
        with open('data/new_questions.json', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry['question'].lower() == user_input.lower() and entry['user_id'] == user_id:
                        return entry['response']
        return None
    except Exception as e:
        print(f"Error checking new_questions.json: {e}")
        return None

def get_response(user_input, user_id):
    """Process user input and return the best matching response."""
    saved_response = check_new_questions(user_input, user_id)
    if saved_response:
        return {
            "answer": saved_response,
            "url": None,
            "similarity": 1.0,
            "category": "saved_data",
            "is_shortcut": False,
            "method": "exact_match",
            "source": "local"
        }

    if user_input.lower() in ['hello', 'hi', 'hey'] or user_input in shortcuts:
        return {
            "answer": shortcuts.get(user_input, "Hello! How can I assist you today?"),
            "url": get_shortcut_url(user_input),
            "similarity": 1.0,
            "category": "shortcut",
            "is_shortcut": True,
            "method": "shortcut",
            "source": "local"
        }
    if user_input.startswith('/'):
        return {
            "answer": "Commande inconnue. Tapez /help pour la liste.",
            "url": None,
            "similarity": 0.0,
            "category": "shortcut",
            "is_shortcut": True,
            "method": "shortcut",
            "source": "local"
        }

    processed_input = preprocess_text(user_input)
    input_tfidf = vectorizer.transform([processed_input])
    input_dense = input_tfidf.toarray()

    # TF-IDF approach
    similarities = cosine_similarity(input_tfidf, tfidf_matrix)
    best_match_idx = similarities.argmax()
    max_similarity = similarities[0, best_match_idx]
    category_tfidf = nb_classifier.predict(input_tfidf)[0]

    if max_similarity > 0.65:
        return {
            "answer": responses[best_match_idx],
            "url": f"https://isetsf.rnu.tn{urls[best_match_idx]}" if urls[best_match_idx] else None,
            "file_path": file_paths[best_match_idx] if file_paths[best_match_idx] else None,
            "similarity": float(max_similarity),
            "category": category_tfidf,
            "is_shortcut": False,
            "method": "tfidf",
            "source": "local"
        }

    # FastText approach
    ft_idx, ft_sim = get_best_match_with_fasttext(user_input)
    if ft_sim > 0.8:
        return {
            "answer": responses[ft_idx],
            "url": f"https://isetsf.rnu.tn{urls[ft_idx]}" if urls[ft_idx] else None,
            "file_path": file_paths[ft_idx] if file_paths[ft_idx] else None,
            "similarity": float(ft_sim),
            "category": category_tfidf,
            "is_shortcut": False,
            "method": "fasttext",
            "source": "local"
        }

    # KNN approach
    distances, indices = knn_classifier.kneighbors(input_dense, n_neighbors=1)
    if distances[0][0] < 0.7:
        idx = indices[0][0]
        category_knn = nb_classifier.predict(input_tfidf)[0]
        return {
            "answer": responses[idx],
            "url": f"https://isetsf.rnu.tn{urls[idx]}" if urls[idx] else None,
            "file_path": file_paths[idx] if file_paths[idx] else None,
            "similarity": float(1.0 - distances[0][0]),
            "category": category_knn,
            "is_shortcut": False,
            "method": "knn",
            "source": "local"
        }

    # Index search fallback
    search_result = search_in_index(user_input)
    if search_result:
        return {
            "answer": search_result['answer'],
            "url": f"https://isetsf.rnu.tn{search_result['url']}" if search_result['url'] else None,
            "similarity": 0.5,
            "category": category_tfidf,
            "is_shortcut": False,
            "method": "index_search",
            "source": "local"
        }

    # OpenRouter API fallback
    api_response = call_openrouter_api(user_input)
    response_dict = {
        "answer": api_response,
        "url": None,
        "similarity": 0.0,
        "category": "external_api",
        "is_shortcut": False,
        "method": "External Chatbot",
        "source": "local"
    }
    save_new_question(user_input, response_dict, user_id=user_id)
    return response_dict

def save_new_question(user_input, response, rating=None, user_id=None):
    """Save new questions and responses to a file."""
    try:
        if not os.path.exists("data"):
            os.makedirs("data")
        
        if os.path.exists('data/new_questions.json'):
            with open('data/new_questions.json', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry['question'].lower() == user_input.lower() and entry['user_id'] == user_id:
                            return True
        
        answer = response.get('answer') if isinstance(response, dict) else response
        entry = {
            "question": user_input,
            "response": answer,
            "rating": rating,
            "user_id": user_id,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open('data/new_questions.json', 'a', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
        return True
    except Exception as e:
        print(f"Error saving question: {e}")
        return False