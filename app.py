from flask import Flask, Response, render_template, request, redirect, url_for, flash, session

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from functools import wraps
from io import StringIO
import pandas as pd
import os
from models import User, PredictionHistory, db
from logic import train_model, predict_logic

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://recidivism_user:1234@localhost/recidivism_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def init_model(data_path):
    """Initialize model only if it doesn't exist"""
    model_exists = (
        os.path.exists('models/model_params.json') and
        os.path.exists('models/encrypted_weights.bin') and
        os.path.exists('models/scaler.pkl') and
        os.path.exists('models/private_context.bin') and
        os.path.exists('models/public_context.bin') and
        os.path.exists('models/encoders.pkl')
    )
    
    if not model_exists:
        print("Training new model...")
        train_model(data_path)
        print("Model training completed!")
    else:
        print("Loading existing model...")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Get form data
        input_data = {
            'Person_ID': request.form['Person_ID'],
            'Sex_Code_Text': request.form['Sex_Code_Text'],
            'Ethnic_Code_Text': request.form['Ethnic_Code_Text'],  # Will be encrypted
            'DateOfBirth': request.form['DateOfBirth'],
            'MaritalStatus': request.form['MaritalStatus'],
            'LegalStatus': request.form['LegalStatus'],
            'CustodyStatus': request.form['CustodyStatus'],
            'RecSupervisionLevel': int(request.form['RecSupervisionLevel']),  # Will be encrypted
            'RawScore': float(request.form['RawScore'])  # Will be encrypted
        }
        
        # Make prediction (encryption happens inside)
        risk_probability, risk_level = predict_logic(input_data)
        
        # Save to database
        pred_record = PredictionHistory(
            user_id=session['user_id'],
            person_id=input_data['Person_ID'],
            prediction_result=float(risk_probability),
            risk_level=risk_level,
            input_data=input_data
        )
        db.session.add(pred_record)
        db.session.commit()
        
        # Store in session
        session['prediction'] = {
            'risk_probability': float(risk_probability),
            'risk_level': risk_level,
            'person_id': input_data['Person_ID']
        }
        
        return redirect(url_for('results'))
    
    # Return template for GET request
    return render_template('predict.html')

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        user = User(username=username, password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('predict'))
        
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/results')
@login_required
def results():
    prediction = session.get('prediction')
    if prediction is None:
        flash('No prediction data found', 'error')
        return redirect(url_for('predict'))
    
    risk_labels = {
        'Low Risk': 'bg-green-100 text-green-800',
        'Medium Risk': 'bg-yellow-100 text-yellow-800',
        'High Risk': 'bg-red-100 text-red-800'
    }
    
    if 'prediction_date' not in prediction:
        prediction['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('results.html', 
                         prediction=prediction,
                         risk_labels=risk_labels)

@app.route('/history')
@login_required
def history():
    predictions = PredictionHistory.query.filter_by(
        user_id=session['user_id']
    ).order_by(PredictionHistory.prediction_date.desc()).all()
    
    risk_labels = {
        'Low Risk': 'bg-green-100 text-green-800',
        'Medium Risk': 'bg-yellow-100 text-yellow-800',
        'High Risk': 'bg-red-100 text-red-800'
    }
    
    return render_template('history.html', 
                         predictions=predictions,
                         risk_labels=risk_labels)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/export_history')
@login_required
def export_history():
    predictions = PredictionHistory.query.filter_by(
        user_id=session['user_id']
    ).order_by(PredictionHistory.prediction_date.desc()).all()
    
    data = []
    for pred in predictions:
        row = {
            'Person ID': pred.person_id,
            'Risk Probability': pred.prediction_result,
            'Risk Level': pred.risk_level,
            'Prediction Date': pred.prediction_date,
            **pred.input_data
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=prediction_history.csv'}
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
    data_path = 'D:\\Data privacy and security analytics\\CAT 1\\data\\subset2.csv' # Replace with your actual data path
    init_model(data_path)  # Initialize once before starting Flask
    app.run(debug=True, use_reloader=False)

