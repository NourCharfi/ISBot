import streamlit as st
import requests
import json
from pathlib import Path
import sys

# Ajout du chemin parent au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import API_CONFIG

# Configuration de la page
st.set_page_config(
    page_title="ISET Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input {
        background-color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
    }
    .chat-message .confidence {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("🤖 ISET Chatbot")
st.markdown("""
Bienvenue sur le chatbot de l'ISET ! Je suis là pour répondre à vos questions sur :
- Les programmes universitaires
- Les procédures administratives
- Les horaires des cours
- Et bien plus encore !
""")

# Initialisation de l'historique des messages dans la session
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Fonction pour envoyer un message au chatbot
def send_message(message):
    try:
        response = requests.post(
            f"http://{API_CONFIG['host']}:{API_CONFIG['port']}/api/chat",
            json={"question": message}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur lors de la communication avec le chatbot : {str(e)}")
        return None

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div>👤 Vous :</div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot">
                <div>🤖 Chatbot :</div>
                <div>{message["content"]}</div>
                <div class="confidence">Confiance : {message["confidence"]:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

# Zone de saisie du message
user_input = st.text_input("Votre question :", key="user_input")

if user_input:
    # Ajout du message de l'utilisateur à l'historique
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Envoi du message au chatbot
    response = send_message(user_input)
    
    if response:
        # Ajout de la réponse du chatbot à l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "confidence": response["confidence"]
        })
    
    # Rafraîchissement de la page pour afficher les nouveaux messages
    st.experimental_rerun()

# Pied de page
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Développé dans le cadre du projet Machine Learning DSIR 1</p>
    <p>© 2024 ISET</p>
</div>
""", unsafe_allow_html=True) 