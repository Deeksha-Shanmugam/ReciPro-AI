from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import os
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://recidivism_user:1234@localhost/recidivism_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)
from datetime import datetime

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)  # Make sure this is long enough for the hash
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    person_id = db.Column(db.String(100), nullable=False)
    prediction_result = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    input_data = db.Column(db.JSON)