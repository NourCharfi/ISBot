# ISET Chatbot

Un chatbot intelligent pour guider les utilisateurs du site de l'ISET, développé dans le cadre du projet Machine Learning DSIR 1.

## Fonctionnalités

- Réponses aux questions des étudiants sur divers sujets
- Navigation assistée vers les pages pertinentes du site
- Support multilingue (Français/Anglais)
- Système de feedback pour amélioration continue
- Suggestions proactives basées sur le contexte

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd iset-chatbot
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Télécharger les ressources NLTK :
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Structure du Projet

```
iset-chatbot/
├── data/                   # Données d'entraînement et de test
├── models/                 # Modèles entraînés
├── src/
│   ├── preprocessing/      # Prétraitement des textes
│   ├── models/            # Implémentation des modèles ML
│   ├── api/               # API Flask
│   └── utils/             # Utilitaires
├── tests/                 # Tests unitaires
├── web/                   # Interface utilisateur
└── config/               # Fichiers de configuration
```

## Utilisation

1. Lancer l'API :
```bash
python src/api/app.py
```

2. Lancer l'interface web :
```bash
streamlit run web/app.py
```

## Évaluation

Le chatbot est évalué sur :
- Précision des réponses
- Temps de réponse
- Satisfaction utilisateur
- Taux de résolution des requêtes

## Contribution

Les contributions sont les bienvenues ! Veuillez suivre les étapes suivantes :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence [LICENSE_TYPE]. Voir le fichier `LICENSE` pour plus de détails.
