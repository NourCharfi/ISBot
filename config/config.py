import os
from pathlib import Path

# Chemins des dossiers
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Configuration du modèle
MODEL_CONFIG = {
    "vectorizer": {
        "max_features": 5000,
        "ngram_range": (1, 2)
    },
    "knn": {
        "n_neighbors": 5,
        "weights": "distance"
    },
    "svm": {
        "kernel": "linear",
        "C": 1.0
    }
}

# Configuration de l'API
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True
}

# Configuration du prétraitement
PREPROCESSING_CONFIG = {
    "stopwords": ["le", "la", "les", "un", "une", "des", "et", "ou", "mais", "donc", "car", "ni"],
    "min_word_length": 2,
    "max_word_length": 20
}

# Configuration de la base de données
DATABASE_CONFIG = {
    "path": DATA_DIR / "chatbot.db"
}

# Création des dossiers s'ils n'existent pas
for directory in [DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 