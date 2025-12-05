import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


HOME_FIELD_ADJ = 100
MODEL_PATH = 'epl_predictor_model.pkl'

_model = None
_scaler = None
_model_loaded = False

def expected_score(rating_a, rating_b):
    """
    Calculate expected score using Elo formula.
    Returns probability of team A winning.
    """
    return 1 / (1 + 10**((rating_b - rating_a) / 400))


def prepare_training_data(df_processed):

    df_processed['EloDiff'] = (df_processed['HomeElo_Before'] + HOME_FIELD_ADJ) - df_processed['AwayElo_Before']
    
    df_processed['HomeWinProb'] = df_processed.apply(
        lambda row: expected_score(row['HomeElo_Before'] + HOME_FIELD_ADJ, row['AwayElo_Before']),
        axis=1
    )
    
    
    df_processed['HomeEloStrength'] = df_processed['HomeElo_Before']
    df_processed['AwayEloStrength'] = df_processed['AwayElo_Before']
    df_processed['AvgElo'] = (df_processed['HomeElo_Before'] + df_processed['AwayElo_Before']) / 2
    
    
    feature_cols = ['EloDiff', 'HomeWinProb', 'HomeEloStrength', 'AwayEloStrength', 'AvgElo']
    X = df_processed[feature_cols].values
    
    
    label_map = {'A': 0, 'D': 1, 'H': 2}
    y = df_processed['FTR'].map(label_map).values
    
    return X, y, feature_cols


def train_model(df_processed):
    print("Preparing training data...")
    X, y, feature_cols = prepare_training_data(df_processed)
    
    
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    print(f"Training on {len(X)} matches...")
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    return model, scaler


def save_model(model, scaler, filepath=MODEL_PATH):
    model_data = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")



def load_model(filepath=MODEL_PATH):
    global _model, _scaler, _model_loaded
    
    if _model_loaded:
        return _model, _scaler
    
    if not os.path.exists(filepath):
        print(f"Warning: Model file not found at {filepath}. Using Elo-based fallback.")
        return None, None
    
    try:
        model_data = joblib.load(filepath)
        _model = model_data['model']
        _scaler = model_data['scaler']
        _model_loaded = True
        print(f"Model loaded successfully from {filepath}")
        return _model, _scaler
    except Exception as e:
        print(f"Error loading model: {e}. Using Elo-based fallback.")
        return None, None
    


def predict_score(home_team_elo, away_team_elo):

    base_goals = 1.4
    
    
    home_adjustment = (home_team_elo + HOME_FIELD_ADJ - 1500) / 400
    away_adjustment = (away_team_elo - 1500) / 400
    
    
    home_expected = base_goals + home_adjustment * 0.5
    away_expected = base_goals + away_adjustment * 0.5
    
    
    home_expected = max(0.5, home_expected)
    away_expected = max(0.5, away_expected)
    

    home_goals = round(home_expected)
    away_goals = round(away_expected)
    
    return f"{home_goals}-{away_goals}"



def predict_match_outcome(home_team_elo, away_team_elo, return_all_probs=False):

    model, scaler = load_model()
    
    elo_diff = (home_team_elo + HOME_FIELD_ADJ) - away_team_elo
    home_win_prob = expected_score(home_team_elo + HOME_FIELD_ADJ, away_team_elo)
    avg_elo = (home_team_elo + away_team_elo) / 2
    
    features = np.array([[
        elo_diff,
        home_win_prob,
        home_team_elo,
        away_team_elo,
        avg_elo
    ]])
    
    if model is not None and scaler is not None:
        features_scaled = scaler.transform(features)
        probabilities = model.predict_proba(features_scaled)[0]
        
        prob_dict = {'A': probabilities[0], 'D': probabilities[1], 'H': probabilities[2]}

    else:

        p_home = home_win_prob
        

        closeness = 1 - abs(elo_diff) / 800  
        p_draw = max(0.15, min(0.35, 0.25 + closeness * 0.1))
        

        remaining_prob = 1 - p_draw
        p_home_adj = p_home * remaining_prob
        p_away_adj = (1 - p_home) * remaining_prob
        
        prob_dict = {'H': p_home_adj, 'D': p_draw, 'A': p_away_adj}
    

    prediction_key = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[prediction_key]
    
    scoreline = predict_score(home_team_elo, away_team_elo)
    
    
    if prediction_key == 'H':
        prediction_text = f"Home Win ({scoreline})"
    elif prediction_key == 'D':
        prediction_text = f"Draw ({scoreline})"
    else:
        prediction_text = f"Away Win ({scoreline})"
    
    
    confidence_str = f"{confidence * 100:.2f}%"
    
    if return_all_probs:
        all_probs = {
            'home_win': f"{prob_dict['H'] * 100:.1f}%",
            'draw': f"{prob_dict['D'] * 100:.1f}%",
            'away_win': f"{prob_dict['A'] * 100:.1f}%"
        }
        return prediction_text, confidence_str, all_probs
    
    return prediction_text, confidence_str



def initialize_model(df_processed):
    if not os.path.exists(MODEL_PATH):
        print("No existing model found. Training new model...")
        model, scaler = train_model(df_processed)
        save_model(model, scaler)
        
        global _model, _scaler, _model_loaded
        _model = model
        _scaler = scaler
        _model_loaded = True
    else:
        print(f"Model already exists at {MODEL_PATH}")
        load_model()




if __name__ == '__main__':

    print("Testing prediction")
    
    test_cases = [
        (1850, 1650, "Strong Home Team vs Weak Away Team"),
        (1500, 1500, "Evenly Matched Teams"),
        (1400, 1700, "Weak Home Team vs Strong Away Team")
    ]
    
    for home_elo, away_elo, description in test_cases:
        prediction, confidence = predict_match_outcome(home_elo, away_elo)
        print(f"\n{description} (Home: {home_elo}, Away: {away_elo})")
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence}")