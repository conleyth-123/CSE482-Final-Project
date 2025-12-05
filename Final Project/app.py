import json
from flask import Flask, render_template, request, jsonify
from data_processing import load_and_process_data, load_all_epl_data, calculate_elo_ratings
from model import predict_match_outcome, initialize_model

app = Flask(__name__)


print("Loading EPL data...")
TEAMS, SUMMARY_STATS, PERFORMANCE_TRENDS, TEAM_FORM, TEAM_SEASON_WINS = load_and_process_data()


print("Initializing prediction model...")
df = load_all_epl_data()
df_processed, _, _ = calculate_elo_ratings(df)
initialize_model(df_processed)

trends_json = json.dumps(PERFORMANCE_TRENDS)

@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        teams=TEAMS,
        summary_stats=SUMMARY_STATS,
        performance_trends=PERFORMANCE_TRENDS, 
        performance_trends_json=trends_json
    )

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to receive team selection and return a prediction."""
    data = request.get_json()
    home_team = data.get('home_team')
    away_team = data.get('away_team')

    home_elo = TEAM_FORM.get(home_team, 1500)
    away_elo = TEAM_FORM.get(away_team, 1500)

    prediction, confidence, all_probs = predict_match_outcome(home_elo, away_elo, return_all_probs=True)

    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'home_team_form': f"ELO: {home_elo}",
        'away_team_form': f"ELO: {away_elo}",
        'probabilities': all_probs
    })

@app.route('/api/team_history/<team_name>', methods=['GET'])
def team_history(team_name):
    team_data = TEAM_SEASON_WINS.get(team_name)
    
    if team_data:
        seasons = list(team_data.keys())
        wins = list(team_data.values())
        
        return jsonify({
            'labels': seasons,
            'data': wins,
            'team_name': team_name
        })
    return jsonify({'error': f'Win history for team {team_name} not found.'}), 404

if __name__ == '__main__':
    app.run(debug=True)