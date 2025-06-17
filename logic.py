from datetime import datetime
from flask import json
import numpy as np
import pandas as pd
import tenseal as ts
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def sigmoid_approximation(z):
    """Better sigmoid approximation for encrypted computation"""
    return 0.5 + 0.197 * z - 0.004 * z**3

def preprocess_data(df, is_training=True, encoders=None, scaler=None):
    """Enhanced data preprocessing with better feature handling and strict type conversion"""
    try:
        # Handle age calculation
        if 'DateOfBirth' in df.columns:
            df['Age'] = df['DateOfBirth'].apply(calculate_age)
        
        # Define features
        features = [
            'Age', 'Sex_Code_Text', 'Ethnic_Code_Text', 
            'MaritalStatus', 'LegalStatus', 'CustodyStatus', 
            'RecSupervisionLevel', 'RawScore'
        ]
        
        # Ensure all features exist
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create working copy
        X = df[features].copy()
        
        # Convert Age to numeric and handle missing values
        X.loc[:, 'Age'] = pd.to_numeric(X['Age'], errors='coerce')
        age_mean = X['Age'].mean()
        X.loc[:, 'Age'] = X['Age'].fillna(age_mean)
        
        # Handle categorical features
        categorical_cols = ['Sex_Code_Text', 'Ethnic_Code_Text', 'MaritalStatus', 
                          'LegalStatus', 'CustodyStatus', 'RecSupervisionLevel']
        
        if is_training:
            encoders = {}
            for col in categorical_cols:
                encoders[col] = LabelEncoder()
                X.loc[:, col] = encoders[col].fit_transform(X[col].fillna('Unknown'))
        
        else:
            if encoders is None:
                raise ValueError("Encoders must be provided for prediction")
            for col in categorical_cols:
                # During prediction, map unknown values to an existing category
                unknown_value = encoders[col].classes_[0] if 'Unknown' not in encoders[col].classes_ else 'Unknown'
                X.loc[:, col] = X[col].fillna(unknown_value)
                # Handle unseen categories by mapping to an existing one
                X.loc[:, col] = X[col].map(lambda x: x if x in encoders[col].classes_ else unknown_value)
                X.loc[:, col] = encoders[col].transform(X[col])

        
        # Handle numeric features
        numeric_cols = ['Age', 'RawScore']
        X.loc[:, 'RawScore'] = pd.to_numeric(X['RawScore'], errors='coerce')
        rawscore_mean = X['RawScore'].mean()
        X.loc[:, 'RawScore'] = X['RawScore'].fillna(rawscore_mean)
        
        if is_training:
            scaler = StandardScaler()
            X.loc[:, numeric_cols] = scaler.fit_transform(X[numeric_cols])
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for prediction")
            X.loc[:, numeric_cols] = scaler.transform(X[numeric_cols])
        
        # Create target variable
        y = None
        if is_training and 'DecileScore' in df.columns:
            y = (df['DecileScore'] >= 6).astype(int)
        
        # Final type check - ensure X is all float numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
        
        # Convert X to numpy float array
        X_array = X.values.astype(float)
        
        # Return appropriate values depending on training mode
        if is_training:
            return X_array, np.array(y, dtype=float), encoders, scaler
        else:
            return X_array, None, encoders, scaler
    
    except Exception as e:
        print(f"Preprocessing error: {e}")
        raise

def generate_context():
    """Generate and configure TenSEAL encryption context while preserving secret key"""
    # Create context with specific parameters
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    
    # Configure context
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()
    
    # IMPORTANT: Create a copy of the context without secret key for external use
    # but keep the original with secret key for decryption operations
    public_context = context.copy()
    public_context.make_context_public()
    
    # Return both contexts
    return context, public_context

class HomomorphicLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=7):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.private_context = None  # Context with secret key for decryption
        self.public_context = None   # Public context for sharing
        self.encrypted_weights = None

    def encrypt_data(self, X):
        """Encrypt feature matrix using context"""
        try:
            X_array = X if isinstance(X, np.ndarray) else np.array(X, dtype=float)
            
            encrypted_vectors = []
            for row in X_array:
                scaled_row = row / np.max(np.abs(row)) if np.max(np.abs(row)) != 0 else row
                # Use private context to enable later decryption
                encrypted_vectors.append(ts.ckks_vector(self.private_context, scaled_row.tolist()))
            #print(encrypted_vectors)
            return encrypted_vectors
        except Exception as e:
            print(f"Encryption error: {e}")
            raise 

    def approximate_sigmoid(self, x):
        """Simplified sigmoid approximation for encrypted computation"""
        return ts.ckks_vector(self.private_context, [0.5]) + 0.25 * x

    def fit(self, X, y, contexts):
        """Train model with plaintext data, ensuring type consistency"""
        self.private_context, self.public_context = contexts
        
        
        # Debug: Print input types
        #print(f"X type: {type(X)}, X.dtype: {X.dtype if hasattr(X, 'dtype') else 'No dtype'}")
        #print(f"y type: {type(y)}, y.dtype: {y.dtype if hasattr(y, 'dtype') else 'No dtype'}")
        
        # Ensure X and y are numeric arrays 
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=float)
        if X.dtype != np.float64:
            X = X.astype(float)
            
        if not isinstance(y, np.ndarray):
            y = np.array(y, dtype=float)
        if y.dtype != np.float64:
            y = y.astype(float)
        
        n_samples, n_features = X.shape
        
        # Initialize weights with float type explicitly
        self.weights = np.random.randn(n_features).astype(float) * 0.01
        
        print("Starting training ...")
        print(f"X shape: {X.shape}, y shape: {y.shape}, weights shape: {self.weights.shape}")
        
        for epoch in range(self.iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights)
            predictions = sigmoid_approximation(linear_model)
            
            # Calculate gradient
            error = predictions - y
            gradient = np.dot(X.T, error) / n_samples
            
            # Ensure gradient is float
            if gradient.dtype != np.float64:
                gradient = gradient.astype(float)
                
            # Update weights with explicit casting
            self.weights = self.weights - (self.learning_rate * gradient)
            
            # Calculate cost
            epsilon = 1e-9  # To avoid log(0)
            cost = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
            print(f"Epoch {epoch+1}/{self.iterations}, Cost: {cost:.6f}")
        
        # After training, encrypt the weights for secure prediction
        # Use private context to enable later decryption
        self.encrypted_weights = ts.ckks_vector(self.private_context, self.weights.tolist())
        return self.encrypted_weights
    
    def predict_proba(self, X):
        """Make predictions with encrypted data"""
        if self.private_context is None or self.encrypted_weights is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure X is a numpy array with float type
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=float)
        
        # Encrypt input data
        encrypted_X = self.encrypt_data(X)
        probabilities = []
        
        for enc_x in encrypted_X:
            # Compute encrypted prediction (homomorphic operation)
            linear_pred = enc_x.dot(self.encrypted_weights)
            
            try:
                # Decrypt only after secure computation is done
                # This will work because we used private_context with secret key
                linear_value = linear_pred.decrypt()[0]
                
                # Apply sigmoid approximation on decrypted value
                prob_value = sigmoid_approximation(linear_value)
                probabilities.append(prob_value)
            except ValueError as e:
                print(f"Decryption error: {e}")
                # Alternative method: if we have weights available in plaintext
                if self.weights is not None:
                    row = enc_x.decrypt()
                    linear_value = np.dot(row, self.weights)
                    prob_value = sigmoid_approximation(linear_value)
                    probabilities.append(prob_value)
                else:
                    raise
            
        return np.array(probabilities)
    
    def save_model(self, model_dir='models'):
        """Save model components"""
        os.makedirs(model_dir, exist_ok=True)
        
        params = {
            'learning_rate': self.learning_rate,
            'iterations': self.iterations,
            'weights': self.weights.tolist() if self.weights is not None else None
        }
        
        with open(os.path.join(model_dir, 'model_params.json'), 'w') as f:
            json.dump(params, f)
        
        # Save both contexts
        if self.private_context is not None:
            private_context_bytes = self.private_context.serialize(save_secret_key=True)
            with open(os.path.join(model_dir, 'private_context.bin'), 'wb') as f:
                f.write(private_context_bytes)
                
        if self.public_context is not None:
            public_context_bytes = self.public_context.serialize()
            with open(os.path.join(model_dir, 'public_context.bin'), 'wb') as f:
                f.write(public_context_bytes)
        
        if self.encrypted_weights is not None:
            encrypted_weights_bytes = self.encrypted_weights.serialize()
            with open(os.path.join(model_dir, 'encrypted_weights.bin'), 'wb') as f:
                f.write(encrypted_weights_bytes)

    @classmethod
    def load_model(cls, model_dir='models'):
        """Load model components"""
        model = cls()
        
        # Load model parameters
        with open(os.path.join(model_dir, 'model_params.json'), 'r') as f:
            params = json.load(f)
        
        model.learning_rate = params['learning_rate']
        model.iterations = params['iterations']
        if params['weights'] is not None:
            model.weights = np.array(params['weights'])
        
        try:
            # Load private context (with secret key)
            with open(os.path.join(model_dir, 'private_context.bin'), 'rb') as f:
                private_context_bytes = f.read()
                model.private_context = ts.context_from(private_context_bytes)
            
            # Load public context
            with open(os.path.join(model_dir, 'public_context.bin'), 'rb') as f:
                public_context_bytes = f.read()
                model.public_context = ts.context_from(public_context_bytes)
            
            # Load encrypted weights
            with open(os.path.join(model_dir, 'encrypted_weights.bin'), 'rb') as f:
                encrypted_weights_bytes = f.read()
                model.encrypted_weights = ts.ckks_vector_from(model.private_context, encrypted_weights_bytes)
                
        except Exception as e:
            print(f"Error loading model components: {e}")
            # Generate new contexts if loading fails
            model.private_context, model.public_context = generate_context()
            print("Generated new encryption contexts")
            
        return model


# Update train_model to use the new context handling
def train_model(data_path):
    """Train model with proper context handling and type management"""
    try:
        # Create encryption contexts - both private and public
        private_context, public_context = generate_context()
        
        # Load and preprocess data
        df = pd.read_csv(data_path)
        X, y, encoders, scaler = preprocess_data(df)
        
        # Debug prints
        #print(f"X type after preprocessing: {type(X)}, shape: {X.shape}, dtype: {X.dtype}")
        #print(f"y type after preprocessing: {type(y)}, shape: {y.shape}, dtype: {y.dtype}")
        
        # Train model with both contexts
        model = HomomorphicLogisticRegression(learning_rate=0.01, iterations=7)
        model.fit(X, y, (private_context, public_context))
        
        # Evaluate on plaintext
        train_pred = model.predict_proba(X)
        train_accuracy = np.mean((train_pred >= 0.5) == y)
        print(f"Training Accuracy: {train_accuracy:.2%}")
        
        # Save components
        os.makedirs('models', exist_ok=True)
        model.save_model('models')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(encoders, 'models/encoders.pkl')
        
        return model
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        raise

def predict_logic(input_data):
    """Make predictions using loaded model - maintaining encryption boundaries"""
    try:
        # Load model and components
        model = HomomorphicLogisticRegression.load_model('models')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/encoders.pkl')
        
        # Prepare input
        input_df = pd.DataFrame([input_data])
        X_scaled, _, _, _ = preprocess_data(input_df, is_training=False, 
                                          encoders=encoders, scaler=scaler)
        
        # Get encrypted prediction (data remains encrypted during computation)
        risk_prob = model.predict_proba(X_scaled)[0]
        risk_prob = round(risk_prob, 2)
        
        # Determine risk level
        if risk_prob < 0.3:
            risk_level = "Low Risk"
        elif risk_prob < 0.7:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        print(f"Prediction probability: {risk_prob:.4f}")
        return float(risk_prob), risk_level
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise

def calculate_age(date_of_birth):
    """Calculate age safely from various input types"""
    try:
        # Convert to datetime if not already
        if isinstance(date_of_birth, pd.Series):
            date_of_birth = pd.to_datetime(date_of_birth.iloc[0])
        elif not isinstance(date_of_birth, datetime):
            date_of_birth = pd.to_datetime(date_of_birth)
        
        # Calculate age
        return (datetime.now() - date_of_birth).days / 365.25
    except Exception as e:
        print(f"Age calculation error: {e}")
        return 30  # Default age if calculation fails