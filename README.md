Season Performance Trends for Premier League Teams

Final project for CSE 482: Big Data. An analysis of English Premier League (EPL) team performance from the 2010/11 through 2019/20 seasons, paired with a machine learning model that predicts match outcomes.

Author: Thomas Conley
GitHub: https://github.com/conleyth-123/CSE482-Final-Project.git

Overview

This project explores ten seasons of EPL match data to answer two questions: does a league-wide performance pattern exist across teams, and can match outcomes be predicted from team skill levels? The analysis combines a season-by-season win tracker with an Elo-based skill rating system that feeds into a Logistic Regression match predictor, all delivered through a Flask web app.

Features


Season win trends — Tracks each team's total wins per season across 2010–2020, revealing a clear split between consistent top performers and bottom-table teams cycling through promotion and relegation.
Elo rating system — Every team starts at a base rating of 1500, with ratings updated after each match based on result and opponent strength. Home teams receive a 100-point boost to reflect home-field advantage.
League-wide stats — Aggregate metrics including total matches, average goals per match, and home vs. away win rates.
Match outcome predictor — A Logistic Regression model trained on Elo differences and win probabilities, predicting Home Win / Draw / Away Win along with a likely final score. Falls back to Elo-based logic if the trained model file is unavailable.


Tech Stack


Python
Flask
Pandas / NumPy
scikit-learn
joblib


How It Works


Data prep & Elo calculation — Ten seasons of CSV match data are merged and sorted chronologically. Elo ratings are recalculated match-by-match.
League stats — Summary statistics are computed across the full dataset (total matches, scoring averages, win rates).
Win trend tracking — Home and away wins are tallied per team per season, with missing seasons (due to relegation) recorded as zero rather than omitted.
Model training — Elo differences, win probabilities, and raw Elo scores are standardized and used to train a Logistic Regression classifier on historical match results.


Key Results


3,628 matches analyzed across ten seasons.
Average of 2.75 goals per match and 1.38 points per game.
Home teams won 45.6% of matches versus 29.7% for away teams.
A clear divide emerged between top-performing teams (e.g., Liverpool, Manchester United, Chelsea), which stayed clear of relegation but often dipped the season after a strong finish, and bottom-performing teams, whose win totals followed a recurring rise-and-fall pattern tied to promotion and relegation.
The match predictor consistently favored top-performing teams with high confidence, and predicted draws rather than losses in close cases — reinforcing the performance gap found in the win-trend analysis.

