from dotenv import load_dotenv
import os

load_dotenv()

shortcuts = {
    "🕒 Horaires": "Voici les horaires des cours. Consultez le lien pour plus de détails.",
    "📞 Contact": "Pour contacter l'administration: Email: admin@iset.tn, Tél: +216 XX XXX XXX",
    "📝 Inscription": "Les inscriptions sont ouvertes du 1er au 30 septembre. Consultez le guide d'inscription.",
    "📚 Bibliothèque": "La bibliothèque est ouverte du lundi au vendredi de 8h à 18h",
    "📖 Examens": "Le calendrier des examens est disponible via le lien ci-dessous.",
}

shortcut_urls = {
    "🕒 Horaires": "/programmes/horaires",
    "📞 Contact": "/contacts/administration",
    "📝 Inscription": "/admissions/procedure-inscription",
    "📚 Bibliothèque": "/services/bibliotheque",
    "📖 Examens": "/programmes/calendrier-examens",
}

MODEL_PATHS = {
    "fasttext": "models/fasttext.model",
    "nb_classifier": "models/nb_classifier.pkl",
    "knn_classifier": "models/knn_classifier.pkl",
    "vectorizer": "models/vectorizer.pkl"
}

SIMILARITY_THRESHOLDS = {
    "tfidf": 0.65,
    "fasttext": 0.8,
    "knn": 0.7
}

DATA_PATH = "data/data.json"
NEW_QUESTIONS_PATH = "data/new_questions.json"
INDEX_DIR = "indexdir"

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')