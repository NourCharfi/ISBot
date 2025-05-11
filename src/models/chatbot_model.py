from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
from typing import List, Dict, Any, Tuple
import joblib
import os
from pathlib import Path

class ChatbotModel:
    def __init__(self, model_type: str = 'knn'):
        """
        Initialise le modèle du chatbot.
        
        Args:
            model_type (str): Type de modèle à utiliser ('knn' ou 'svm')
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        if model_type == 'knn':
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        else:  # svm
            self.model = SVC(
                kernel='linear',
                probability=True
            )
            
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        
        self.classes = None
        self.answers = None

    def train(self, questions: List[str], labels: List[str], answers: Dict[str, str]):
        """
        Entraîne le modèle sur les données fournies.
        
        Args:
            questions (List[str]): Liste des questions d'entraînement
            labels (List[str]): Liste des labels correspondants
            answers (Dict[str, str]): Dictionnaire des réponses par label
        """
        self.pipeline.fit(questions, labels)
        self.classes = self.model.classes_
        self.answers = answers

    def predict(self, question: str) -> Tuple[str, float]:
        """
        Prédit la réponse pour une question donnée.
        
        Args:
            question (str): Question à traiter
            
        Returns:
            Tuple[str, float]: (Réponse prédite, Score de confiance)
        """
        # Prédiction de la classe
        label = self.pipeline.predict([question])[0]
        
        # Calcul du score de confiance
        if self.model_type == 'knn':
            distances, _ = self.model.kneighbors(
                self.vectorizer.transform([question])
            )
            confidence = 1 / (1 + distances[0][0])  # Conversion de la distance en score
        else:  # svm
            confidence = np.max(self.pipeline.predict_proba([question])[0])
        
        # Récupération de la réponse
        answer = self.answers.get(label, "Désolé, je ne comprends pas votre question.")
        
        return answer, confidence

    def save(self, path: str):
        """
        Sauvegarde le modèle sur le disque.
        
        Args:
            path (str): Chemin où sauvegarder le modèle
        """
        model_data = {
            'pipeline': self.pipeline,
            'classes': self.classes,
            'answers': self.answers
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: str) -> 'ChatbotModel':
        """
        Charge un modèle depuis le disque.
        
        Args:
            path (str): Chemin du modèle à charger
            
        Returns:
            ChatbotModel: Instance du modèle chargé
        """
        model_data = joblib.load(path)
        model = cls()
        model.pipeline = model_data['pipeline']
        model.classes = model_data['classes']
        model.answers = model_data['answers']
        return model 