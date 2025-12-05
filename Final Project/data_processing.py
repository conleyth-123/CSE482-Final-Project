import pandas as pd
import numpy as np
import math
import os 

K_FACTOR = 30
ELO_INIT = 1500
HOME_FIELD_ADJ = 100

DATA_DIR = 'data'


def update_elo(R_A, R_B, S_A, K):
    E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
    R_A_new = R_A + K * (S_A - E_A)
    return R_A_new


def calculate_elo_ratings(df):

    all_teams = set(df['HomeTeam'])
    all_teams = all_teams.union(set(df['AwayTeam']))

    elo_ratings = {}
    for team in all_teams:
        elo_ratings[team] = ELO_INIT

    elo_history = []

    df['HomeElo_Before'] = np.nan
    df['AwayElo_Before'] = np.nan

    for index, row in df.iterrows():

        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        R_H = elo_ratings.get(home_team, ELO_INIT)
        R_A = elo_ratings.get(away_team, ELO_INIT)

        df.loc[index, 'HomeElo_Before'] = R_H
        df.loc[index, 'AwayElo_Before'] = R_A

        R_H_adj = R_H + HOME_FIELD_ADJ

        ftr = row['FTR']
        if ftr == 'H':
            S_H = 1.0
        else:
            if ftr == 'D':
                S_H = 0.5
            else:
                S_H = 0.0

        R_H_new = update_elo(R_H_adj, R_A, S_H, K_FACTOR)
        R_H_new = R_H_new - HOME_FIELD_ADJ

        R_A_new = update_elo(R_A, R_H_adj, 1 - S_H, K_FACTOR)

        elo_ratings[home_team] = R_H_new
        elo_ratings[away_team] = R_A_new

        match_number = index + 1
        season_year = row['Date'].year

        if row['Date'].month < 7:
            season = f"{season_year - 1}/{str(season_year)[-2:]}"
        else:
            season = f"{season_year}/{str(season_year + 1)[-2:]}"

        matchweek = math.ceil((match_number % 380) / (380 / 38))
        if matchweek == 0:
            matchweek = 38

        elo_history.append({'Season': season, 'Matchweek': matchweek, 'Team': home_team, 'Elo': R_H_new})
        elo_history.append({'Season': season, 'Matchweek': matchweek, 'Team': away_team, 'Elo': R_A_new})

    elo_history_df = pd.DataFrame(elo_history)
    final_elo_ratings = elo_ratings

    return df, final_elo_ratings, elo_history_df


def load_all_epl_data():

    epl_files = [
        'epl_10_11.csv',
        'epl_11_12.csv',
        'epl_12_13.csv',
        'epl_13_14.csv',
        'epl_14_15.csv',
        'epl_15_16.csv',
        'epl_16_17.csv',
        'epl_17_18.csv',
        'epl_18_19.csv',
        'epl_19_20.csv'
    ]

    all_data = []

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Directory not found: '{DATA_DIR}'. Please ensure the 'data' folder exists."
        )

    for filename in epl_files:
        full_path = os.path.join(DATA_DIR, filename)

        try:
            df_season = pd.read_csv(full_path, encoding='latin-1')
            print(f"Loaded {filename}: {len(df_season)} rows")
            all_data.append(df_season)

        except FileNotFoundError:
            print(f"File not found: {full_path}. Skipping.")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if len(all_data) == 0:
        raise ValueError(
            "No objects to concatenate. Could not load any EPL files."
        )

    df = pd.concat(all_data, ignore_index=True)

    print(f"Total rows before date parsing: {len(df)}")
    print(f"Sample dates before parsing: {df['Date'].head(10).tolist()}")

    original_dates = df['Date'].copy()

    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

    mask = df['Date'].isna()

    df.loc[mask, 'Date'] = pd.to_datetime(
        original_dates[mask],
        format='%d/%m/%y',
        errors='coerce'
    )

    valid_dates = df['Date'].notna().sum()
    print(f"Valid dates after parsing: {valid_dates} / {len(df)}")

    if valid_dates == 0:
        print("WARNING: No dates were parsed successfully.")
        print(df.iloc[0]['Date'])

    df = df.sort_values(by='Date')
    df = df.reset_index(drop=True)

    df = df.dropna(
        subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    )

    return df


def load_and_process_data():

    df = load_all_epl_data()

    df_processed, final_elo_ratings, elo_history_df = calculate_elo_ratings(df)

    home_teams = set(df_processed['HomeTeam'])
    away_teams = set(df_processed['AwayTeam'])
    teams_list_raw = list(home_teams.union(away_teams))
    teams_list_raw = sorted(teams_list_raw)

    teams_list = []
    for team in teams_list_raw:
        teams_list.append(team.title())

    total_goals = df_processed['FTHG'].sum() + df_processed['FTAG'].sum()
    total_matches = len(df_processed)

    home_wins = (df_processed['FTR'] == 'H').sum()
    away_wins = (df_processed['FTR'] == 'A').sum()

    min_year = df_processed['Date'].min().year
    max_year = df_processed['Date'].max().year

    def season_func(d):
        if d.month < 7:
            return f"{d.year-1}/{str(d.year)[-2:]}"
        else:
            return f"{d.year}/{str(d.year+1)[-2:]}"

    df_processed['Season'] = df_processed['Date'].apply(season_func)

    unique_seasons = sorted(df_processed['Season'].unique())

    min_season = unique_seasons[0]
    max_season = unique_seasons[-1]
    num_seasons = len(unique_seasons)

    def point_calc_home(x):
        if x == 'H':
            return 3
        else:
            if x == 'D':
                return 1
            else:
                return 0

    def point_calc_away(x):
        if x == 'H':
            return 0
        else:
            if x == 'D':
                return 1
            else:
                return 3

    df_processed['HomePoints'] = df_processed['FTR'].apply(point_calc_home)
    df_processed['AwayPoints'] = df_processed['FTR'].apply(point_calc_away)

    total_points = df_processed['HomePoints'].sum() + df_processed['AwayPoints'].sum()

    summary_stats = {
        'Overall PPG': f"{total_points / total_matches / 2 :.2f}",
        'Avg Goals Scored': f"{total_goals / total_matches :.2f}",
        'Overall Home Win Rate': f"{(home_wins / total_matches) * 100 :.1f}%",
        'Overall Away Win Rate': f"{(away_wins / total_matches) * 100 :.1f}%",
        'Seasons Analyzed': f"{min_season} to {max_season} ({num_seasons} Seasons)",
        'Total Matches': f"{total_matches:,}"
    }

    last_year = df_processed['Date'].dt.year.max()
    last_season_df = df_processed.loc[df_processed['Date'].dt.year == last_year]

    last_season_teams = set(last_season_df['HomeTeam'])
    last_season_teams = last_season_teams.union(set(last_season_df['AwayTeam']))

    current_team_elo = {}
    for team, elo in final_elo_ratings.items():
        if team in last_season_teams:
            current_team_elo[team] = elo

    sorted_elo = sorted(
        current_team_elo.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_n = min(4, len(sorted_elo))
    top_tier_teams = []

    for team, elo in sorted_elo[:top_n]:
        top_tier_teams.append(team)

    elo_history_df['Tier'] = 'lower_tier'
    for team in top_tier_teams:
        elo_history_df.loc[elo_history_df['Team'] == team, 'Tier'] = 'top_tier'

    last_season_str = elo_history_df['Season'].max()
    last_season_df = elo_history_df[elo_history_df['Season'] == last_season_str]

    matchweek_elo = last_season_df.groupby(['Matchweek', 'Tier'])['Elo']
    matchweek_elo = matchweek_elo.mean()
    matchweek_elo = matchweek_elo.reset_index()

    trends_data = matchweek_elo.pivot(
        index='Matchweek',
        columns='Tier',
        values='Elo'
    )
    trends_data = trends_data.reset_index()

    matchweeks = trends_data['Matchweek'].tolist()

    top_tier_elo = []
    for e in trends_data.get('top_tier', [np.nan] * len(matchweeks)).tolist():
        if pd.notna(e):
            top_tier_elo.append(f"{e:.2f}")

    lower_tier_elo = []
    for e in trends_data.get('lower_tier', [np.nan] * len(matchweeks)).tolist():
        if pd.notna(e):
            lower_tier_elo.append(f"{e:.2f}")

    matchweeks = matchweeks[:len(top_tier_elo)]

    performance_trends = {
        'matchweeks': matchweeks,
        'top_tier_elo': top_tier_elo,
        'lower_tier_elo': lower_tier_elo,
        'chart_label': f'Average Elo over Season Matchweeks ({last_season_str} Season)'
    }

    team_form = {}

    for k, v in final_elo_ratings.items():
        team_form[k.title()] = round(v)

    df_processed['Season'] = df_processed['Date'].apply(season_func)

    home_wins_df = df_processed[df_processed['FTR'] == 'H'].copy()
    away_wins_df = df_processed[df_processed['FTR'] == 'A'].copy()

    home_wins_count = home_wins_df.groupby(['Season', 'HomeTeam']).size()
    home_wins_count = home_wins_count.reset_index(name='HomeWins')
    home_wins_count = home_wins_count.rename(columns={'HomeTeam': 'Team'})

    away_wins_count = away_wins_df.groupby(['Season', 'AwayTeam']).size()
    away_wins_count = away_wins_count.reset_index(name='AwayWins')
    away_wins_count = away_wins_count.rename(columns={'AwayTeam': 'Team'})

    total_wins_per_season = pd.merge(
        home_wins_count,
        away_wins_count,
        on=['Season', 'Team'],
        how='outer'
    )

    total_wins_per_season = total_wins_per_season.fillna(0)
    total_wins_per_season['TotalWins'] = (
        total_wins_per_season['HomeWins'] + total_wins_per_season['AwayWins']
    )

    team_season_wins = total_wins_per_season.pivot(
        index='Team',
        columns='Season',
        values='TotalWins'
    )

    team_season_wins = team_season_wins.fillna(0)
    team_season_wins = team_season_wins.astype(int)

    team_season_wins = team_season_wins.apply(
        lambda x: x.to_dict(),
        axis=1
    )
    team_season_wins = team_season_wins.to_dict()

    final_dict = {}
    for k, v in team_season_wins.items():
        final_dict[k.title()] = v

    team_season_wins = final_dict

    return teams_list, summary_stats, performance_trends, team_form, team_season_wins


if __name__ == '__main__':

    try:
        teams, summary, trends, form, season_wins = load_and_process_data()
        print("Data loaded successfully!")
        print(f"Total Teams: {len(teams)}")
        print(f"Summary Stats: {summary}")

        example = list(form.items())[:5]
        print(f"Example Team Form (Elo): {example}")

        first_team = list(season_wins.keys())[0]
        seasons = list(season_wins[first_team].keys())
        print(f"Seasons with data: {seasons}")

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
