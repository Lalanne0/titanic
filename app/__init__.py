from flask import Flask
from config import Config
import logging
from logging.handlers import RotatingFileHandler
import os

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure logs directory exists
    if not os.path.exists('logs'):
        os.mkdir('logs')

    # Configure logging
    file_handler = RotatingFileHandler('logs/titanic_predictor.log', 
                                     maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Titanic Predictor Startup')

    # Register blueprints
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    return app

# app/models/preprocessing.py
import pandas as pd
import numpy as np

def preprocess_cabin(df):
    df = df.copy()
    
    df['Deck'] = df['Cabin'].str[0] if 'Cabin' in df.columns else None
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)

    deck_mapping = {
        'A': 'ABC', 'B': 'ABC', 'C': 'ABC',
        'D': 'DE', 'E': 'DE',
        'F': 'FG', 'G': 'FG',
        'T': 'Other'
    }

    df['Deck_category'] = df['Deck'].map(deck_mapping)
    df['Deck_category'] = df['Deck_category'].fillna('Unknown')

    if 'Pclass' in df.columns and 'Fare' in df.columns:
        fare_means = df.groupby('Pclass')['Fare'].transform('mean')
        df['Fare_ratio'] = df['Fare'] / fare_means
        
        conditions = [
            (df['Pclass'] == 1) & (df['Fare_ratio'] > 1.5),
            (df['Pclass'] == 1) & (df['Fare_ratio'] <= 1.5),
            (df['Pclass'] == 2) & (df['Fare_ratio'] > 1.2),
            (df['Pclass'] == 2) & (df['Fare_ratio'] <= 1.2),
            (df['Pclass'] == 3) & (df['Fare_ratio'] > 1),
            (df['Pclass'] == 3) & (df['Fare_ratio'] <= 1)
        ]
        choices = ['ABC', 'DE', 'DE', 'FG', 'FG', 'FG']
        
        df.loc[df['Deck_category'] == 'Unknown', 'Deck_category'] = pd.Series(
            np.select(conditions, choices, default='Unknown'))
            
    return df

def preprocess_title(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

def preprocess_family(df):
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    return df

def preprocess_data(df):
    df = preprocess_cabin(df)
    df = preprocess_title(df)
    df = preprocess_family(df)
    df['Embarked'] = df['Embarked'].fillna('S')
    return df

# app/models/predictor.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

class TitanicPredictor:
    def __init__(self):
        self.numeric_features = ['Age', 'Fare_ratio', 'FamilySize', 'FarePerPerson']
        self.categorical_features = ['Pclass', 'Sex', 'Embarked', 'Has_Cabin', 
                                   'Deck_category', 'Title', 'IsAlone']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), self.numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), self.categorical_features)
            ])
        
        # Load all models
        self.models = {
            'random_forest': joblib.load('app/models/random_forest_model.pkl'),
            'xgboost': joblib.load('app/models/xgboost_model.pkl'),
            'lightgbm': joblib.load('app/models/lightgbm_model.pkl')
        }
        
    def predict(self, data, model_name='random_forest'):
        """Make prediction using the specified model."""
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess the data
        df = preprocess_data(df)
        
        # Select features
        X = df[self.numeric_features + self.categorical_features]
        
        # Transform features
        X_processed = self.preprocessor.transform(X)
        
        # Get prediction and probability
        model = self.models[model_name]
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0].max()
        
        return {
            'survived': bool(prediction),
            'probability': float(probability),
            'model_used': model_name
        }

# app/routes.py
from flask import Blueprint, request, jsonify, render_template
from app.models.predictor import TitanicPredictor
import logging

bp = Blueprint('main', __name__)
predictor = TitanicPredictor()

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'Pclass': int(request.form['pclass']),
            'Sex': request.form['sex'],
            'Age': float(request.form['age']),
            'SibSp': int(request.form['sibsp']),
            'Parch': int(request.form['parch']),
            'Fare': float(request.form['fare']),
            'Embarked': request.form['embarked'],
            'Cabin': request.form.get('cabin', ''),
            'Name': request.form.get('name', 'Unknown')
        }
        
        # Get model selection
        model_name = request.form.get('model', 'random_forest')
        
        # Make prediction
        result = predictor.predict(data, model_name)
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(result)
        else:
            return render_template('result.html', 
                                 prediction=result['survived'],
                                 probability=result['probability'],
                                 model_used=result['model_used'],
                                 data=data)
                                 
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')
    
# run.py
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=False)