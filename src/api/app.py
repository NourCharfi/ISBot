from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path

# Ajout du chemin parent au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.chatbot_model import ChatbotModel
from src.preprocessing.text_processor import TextProcessor
from config.config import API_CONFIG, MODELS_DIR

app = Flask(__name__)
CORS(app)

# Initialisation du modèle et du processeur de texte
model = None
text_processor = TextProcessor()

def load_model():
    """Charge le modèle entraîné."""
    global model
    model_path = MODELS_DIR / "chatbot_model.joblib"
    if model_path.exists():
        model = ChatbotModel.load(str(model_path))
    else:
        raise FileNotFoundError("Le modèle n'a pas été trouvé. Veuillez d'abord entraîner le modèle.")

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint pour interagir avec le chatbot.
    
    Format de la requête:
    {
        "question": "Votre question ici"
    }
    
    Format de la réponse:
    {
        "answer": "Réponse du chatbot",
        "confidence": 0.95
    }
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'error': 'La question est requise'
            }), 400
            
        question = data['question']
        
        # Prétraitement de la question
        processed_question = ' '.join(text_processor.process(question))
        
        # Prédiction
        answer, confidence = model.predict(processed_question)
        
        return jsonify({
            'answer': answer,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint pour vérifier l'état de l'API."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    try:
        load_model()
        app.run(
            host=API_CONFIG['host'],
            port=API_CONFIG['port'],
            debug=API_CONFIG['debug']
        )
    except Exception as e:
        print(f"Erreur lors du démarrage de l'API: {str(e)}")
        sys.exit(1) 