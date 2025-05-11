from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=True)

    def get_id(self):
        return str(self.id)

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    date = db.Column(db.String(50), nullable=False)
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id', ondelete='CASCADE'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)

class SharedChat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(64), unique=True, nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id', ondelete='CASCADE'), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)