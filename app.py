from flask import Flask, request, jsonify, render_template, redirect, url_for, session, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import os
import json
import datetime
import requests
import logging
from logging.handlers import RotatingFileHandler
from chatbot.data_processing import preprocess_text, vectorizer, tfidf_matrix
from chatbot.chatbot_logic import get_response, save_new_question
from chatbot.database import db, User, ChatSession, ChatMessage, SharedChat
import secrets

# Allow OAuth2 to work with HTTP in development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
GOOGLE_REDIRECT_URI = 'http://localhost:8080/auth/callback'  # Updated port to 8080
SCOPES = ['https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile', 'openid']

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a file handler
handler = RotatingFileHandler('oauth_debug.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(console_handler)

# Create database tables
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login')
def login():
    """Initiate Google OAuth login."""
    logger.info("Starting OAuth login process")
    try:
        # Clear any existing session data
        session.clear()
        
        logger.debug(f"GOOGLE_CLIENT_ID: {GOOGLE_CLIENT_ID}")
        logger.debug(f"GOOGLE_REDIRECT_URI: {GOOGLE_REDIRECT_URI}")
        
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uris": [GOOGLE_REDIRECT_URI],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token"
                }
            },
            scopes=SCOPES
        )
        flow.redirect_uri = GOOGLE_REDIRECT_URI
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='select_account'  # Force account selection
        )
        logger.info(f"Generated authorization URL: {authorization_url}")
        logger.info(f"Generated state: {state}")
        
        session['state'] = state
        return redirect(authorization_url)
    except Exception as e:
        logger.error(f"Error in login route: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/auth/callback')
def auth_callback():
    """Handle Google OAuth callback."""
    logger.info("Received OAuth callback")
    logger.debug(f"Full request URL: {request.url}")
    logger.debug(f"Request args: {request.args}")
    
    try:
        state = session.get('state')
        logger.debug(f"Session state: {state}")
        
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uris": [GOOGLE_REDIRECT_URI],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token"
                }
            },
            scopes=SCOPES,
            state=state
        )
        flow.redirect_uri = GOOGLE_REDIRECT_URI
        
        logger.info("Attempting to fetch token")
        flow.fetch_token(authorization_response=request.url)
        logger.info("Successfully fetched token")

        credentials = flow.credentials
        logger.debug(f"Access token: {credentials.token[:10]}...")  # Only log first 10 chars for security
        
        user_info = requests.get(
            'https://www.googleapis.com/oauth2/v1/userinfo',
            headers={'Authorization': f'Bearer {credentials.token}'}
        ).json()
        logger.info(f"Retrieved user info for: {user_info.get('email')}")

        email = user_info.get('email')
        user = User.query.filter_by(email=email).first()
        if not user:
            logger.info(f"Creating new user for email: {email}")
            user = User(email=email, name=user_info.get('name', ''))
            db.session.add(user)
            db.session.commit()

        login_user(user)
        logger.info(f"Successfully logged in user: {email}")
        return redirect(url_for('chat'))
        
    except Exception as e:
        logger.error(f"Error in auth_callback: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/logout')
@login_required
def logout():
    """Log out the current user."""
    # Clear all session data
    session.clear()
    logout_user()
    return redirect(url_for('chat'))

@app.route('/', methods=['GET', 'POST'])
def chat():
    """Handle chat interface and user input."""
    if not current_user.is_authenticated:
        return render_template('chat.html', chat_history=[], session_id=None)

    session_id = request.args.get('session_id')
    if session_id:
        current_session = ChatSession.query.filter_by(id=int(session_id), user_id=current_user.id).first()
    else:
        # Get the most recent session or create a new one if none exists
        current_session = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.id.desc()).first()
        if not current_session:
            current_session = ChatSession(
                user_id=current_user.id,
                date=datetime.datetime.now().isoformat()
            )
            db.session.add(current_session)
            db.session.commit()

    # Preprocess chat history to parse bot_response
    chat_history = current_session.messages if current_session else []
    processed_history = []
    for msg in chat_history:
        try:
            processed_msg = {
                'user_message': msg.user_message,
                'bot_response': json.loads(msg.bot_response),
                'timestamp': msg.timestamp
            }
            processed_history.append(processed_msg)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing bot_response for message ID {msg.id}: {e}")
            continue

    if request.method == 'POST' and current_user.is_authenticated:
        user_input = request.form.get('message')
        if user_input:
            response = get_response(user_input, current_user.id)
            timestamp = datetime.datetime.now().isoformat()

            chat_entry = ChatMessage(
                session_id=current_session.id,
                user_message=user_input,
                bot_response=json.dumps(response),
                timestamp=timestamp
            )
            db.session.add(chat_entry)
            db.session.commit()

            # Save to new_questions.json if similarity is low and not a shortcut
            if response['similarity'] < 0.8 and not response.get('is_shortcut', False):
                save_new_question(user_input, response['answer'], user_id=current_user.id)

            processed_history = []
            for msg in current_session.messages:
                try:
                    processed_msg = {
                        'user_message': msg.user_message,
                        'bot_response': json.loads(msg.bot_response),
                        'timestamp': msg.timestamp
                    }
                    processed_history.append(processed_msg)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing bot_response for message ID {msg.id}: {e}")
                    continue

    return render_template('chat.html', chat_history=processed_history, session_id=current_session.id if current_session else None)

@app.route('/about')
def about():
    """Render about page."""
    return render_template('about.html')

@app.route('/rate', methods=['POST'])
@login_required
def rate_response():
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'rating' not in data or 'response' not in data:
            return jsonify({'status': 'error', 'message': 'Missing required data'}), 400

        question = data['question']
        rating = data['rating']
        response = data['response']

        if rating == 1:  # Positive rating
            # Save to data.json
            if not save_to_data_json(question, response):
                return jsonify({'status': 'error', 'message': 'Failed to save to data.json'}), 500
            # Remove from new_questions.json if it exists
            remove_from_new_questions(question)
        else:  # Negative rating
            # Remove from both data.json and new_questions.json
            if not remove_from_data_json(question):
                app.logger.warning(f"Failed to remove question from data.json: {question}")
            if not remove_from_new_questions(question):
                app.logger.warning(f"Failed to remove question from new_questions.json: {question}")

        return jsonify({'status': 'success'})
    except Exception as e:
        app.logger.error(f"Error in rate_response: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def remove_from_data_json(question):
    """Remove a question from data.json"""
    try:
        data_file = 'data/data.json'  # Fixed path to match save_to_data_json
        if not os.path.exists(data_file):
            app.logger.warning(f"data.json not found at {data_file}")
            return False
            
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter out the question
        original_length = len(data)
        data = [entry for entry in data if entry['question'].lower() != question.lower()]
        
        if len(data) < original_length:
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            app.logger.info(f"Removed question from data.json: {question}")
            return True
        else:
            app.logger.info(f"Question not found in data.json: {question}")
            return False
            
    except Exception as e:
        app.logger.error(f"Error removing from data.json: {str(e)}")
        return False

def save_to_data_json(question, response):
    """Save positively rated Q&A to data.json."""
    try:
        data_file = 'data/data.json'
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Create or load existing data
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
            
        # Check if question already exists
        for entry in data:
            if entry['question'].lower() == question.lower():
                # Update existing entry
                entry['answer'] = response
                entry['user_id'] = current_user.id
                entry['timestamp'] = datetime.datetime.now().isoformat()
                break
        else:
            # Add new entry if question doesn't exist
            new_entry = {
                "id": len(data) + 1,
                "category": "user_rated",
                "question": question,
                "question_variations": [],
                "answer": response,
                "url": "",
                "user_id": current_user.id,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data.append(new_entry)
            
        # Save updated data
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Error saving to data.json: {str(e)}", exc_info=True)
        return False

def remove_from_new_questions(question):
    """Remove negatively rated question from new_questions.json."""
    try:
        new_questions_file = 'data/new_questions.json'
        if not os.path.exists(new_questions_file):
            return True
            
        # Read all lines
        with open(new_questions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Filter out the question
        filtered_lines = []
        for line in lines:
            if line.strip():
                try:
                    entry = json.loads(line)
                    if not (entry['question'].lower() == question.lower() and entry['user_id'] == current_user.id):
                        filtered_lines.append(line)
                except json.JSONDecodeError:
                    continue
                    
        # Write back filtered content
        with open(new_questions_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
            
        return True
    except Exception as e:
        logger.error(f"Error removing from new_questions.json: {str(e)}", exc_info=True)
        return False

@app.route('/new_chat', methods=['POST'])
@login_required
def new_chat():
    """Create a new chat session."""
    new_session = ChatSession(
        user_id=current_user.id,
        date=datetime.datetime.now().isoformat()
    )
    db.session.add(new_session)
    db.session.commit()
    return jsonify({"status": "success", "session_id": new_session.id})

@app.route('/delete_chat', methods=['POST'])
@login_required
def delete_chat():
    """Delete a chat session."""
    data = request.get_json()
    session_id = data.get('session_id')
    session = ChatSession.query.filter_by(id=int(session_id), user_id=current_user.id).first()
    if session:
        db.session.delete(session)
        db.session.commit()
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Session not found"}), 404

@app.route('/get_sessions', methods=['GET'])
@login_required
def get_sessions():
    """Return all chat sessions for the current user, with a friendly name as title."""
    import datetime
    import calendar
    sessions = ChatSession.query.filter_by(user_id=current_user.id).all()
    result = []
    for s in sessions:
        # Use first user message if available and not empty
        if s.messages and len(s.messages) > 0 and s.messages[0].user_message and s.messages[0].user_message.strip():
            first_msg = s.messages[0].user_message.strip()
            title = first_msg[:40] + ('...' if len(first_msg) > 40 else '')
        else:
            # Fallback: Chat_May09
            try:
                dt = datetime.datetime.fromisoformat(s.date)
                month_name = dt.strftime('%b')  # e.g., 'May'
                day = f"{dt.day:02d}"
                title = f"Chat_{month_name}{day}"
            except Exception:
                title = "Nouveau chat"
        result.append({
            "id": s.id,
            "date": s.date,
            "title": title,
            "messages": [{"user": m.user_message, "bot": json.loads(m.bot_response), "timestamp": m.timestamp} for m in s.messages]
        })
    return jsonify(result)

@app.route('/api/share_chat/<int:session_id>', methods=['POST'])
@login_required
def api_share_chat(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not session:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    shared = SharedChat.query.filter_by(session_id=session_id).first()
    if not shared:
        token = secrets.token_urlsafe(16)
        shared = SharedChat(token=token, session_id=session_id, created_at=datetime.datetime.now())
        db.session.add(shared)
        db.session.commit()
    share_url = url_for('view_shared_chat', token=shared.token, _external=True)
    return jsonify({'status': 'success', 'share_url': share_url})

@app.route('/share/<token>')
def view_shared_chat(token):
    shared = SharedChat.query.filter_by(token=token).first()
    if not shared:
        abort(404)
    session = ChatSession.query.filter_by(id=shared.session_id).first()
    if not session:
        abort(404)
    # Parse chat history for display (public, read-only)
    chat_history = session.messages if session else []
    processed_history = []
    for msg in chat_history:
        try:
            processed_msg = {
                'user_message': msg.user_message,
                'bot_response': json.loads(msg.bot_response),
                'timestamp': msg.timestamp
            }
            processed_history.append(processed_msg)
        except Exception:
            continue
    return render_template('shared_chat.html', chat_history=processed_history)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))  # Updated port to 8080
    app.run(host='0.0.0.0', port=port, debug=True)