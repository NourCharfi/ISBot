import json
import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
import os
from gensim.models import FastText

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stemmer and stop words
stemmer_fr = SnowballStemmer('french')
stop_words_fr = set(stopwords.words('french'))

# Whoosh schema
schema = Schema(question=TEXT(stored=True), answer=TEXT(stored=True), url=TEXT(stored=True))

# Initialize Whoosh index
if not os.path.exists("indexdir"):
    os.makedirs("indexdir")
ix = create_in("indexdir", schema)

# Global variables
questions = []
responses = []
urls = []
file_paths = []
categories = []
vectorizer = None
tfidf_matrix = None
fasttext_model = None
fasttext_question_vectors = []

def load_data():
    """Load data from JSON and index it."""
    global questions, responses, urls, file_paths, categories
    with open('data/data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    writer = ix.writer()
    for entry in data:
        writer.add_document(
            question=entry['question'],
            answer=entry['answer'],
            url=entry.get('url', '')  # Make url optional
        )
        main_question = entry['question']
        variations = entry.get('question_variations', [])
        questions.extend([main_question] + variations)
        responses.extend([entry['answer']] * (len(variations) + 1))
        urls.extend([entry.get('url', '')] * (len(variations) + 1))
        file_paths.extend([entry.get('file_path', '')] * (len(variations) + 1))
        categories.extend([entry.get('category', 'general')] * (len(variations) + 1))
    writer.commit()

def preprocess_text(text):
    """Preprocess text for analysis."""
    try:
        lang = detect(text)
    except:
        lang = 'fr'
    
    stemmer = stemmer_fr if lang == 'fr' else SnowballStemmer('english')
    stop_words = stop_words_fr if lang == 'fr' else set(stopwords.words('english'))
    
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = [stemmer.stem(word) for word in word_tokenize(text) if word not in stop_words]
    return ' '.join(tokens)

def get_document_vector_fasttext(doc, model):
    """Generate document vector by averaging FastText word vectors."""
    words = preprocess_text(doc).split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# Load data
load_data()

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2)
processed_questions = [preprocess_text(q) for q in questions]
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# Train or load FastText model
fasttext_model_path = 'models/fasttext.model'
if os.path.exists(fasttext_model_path):
    fasttext_model = FastText.load(fasttext_model_path)
else:
    tokenized_questions = [preprocess_text(q).split() for q in questions]
    fasttext_model = FastText(tokenized_questions, vector_size=100, window=5, min_count=1, workers=4)
    if not os.path.exists('models'):
        os.makedirs('models')
    fasttext_model.save(fasttext_model_path)

# Pre-calculate FastText vectors
fasttext_question_vectors = np.array([get_document_vector_fasttext(q, fasttext_model) for q in questions])