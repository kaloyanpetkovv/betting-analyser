import requests
import random
import pandas as pd
import os
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from concurrent.futures import ThreadPoolExecutor
import joblib
import time
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_fixed
import optuna
import logging
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
# Please install OpenAI SDK first: `pip3 install openai`
import http.client

conn = http.client.HTTPSConnection("betsapi2.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "c7f6d1f366msh9a061064e2597a9p17f3e6jsn2e50f204f9a8",
    'x-rapidapi-host': "betsapi2.p.rapidapi.com"
}

conn.request("GET", "/v1/bet365/inplay_filter?sport_id=1", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))


# Configuration management
@dataclass
class Config:
    API_FOOTBALL_KEY: str = os.getenv("API_FOOTBALL_KEY", "")
    MAX_RETRIES: int = 3
    CACHE_SIZE: int = 500
    MODEL_PATH: str = "models/trained_model.pkl"


config = Config()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@lru_cache(maxsize=100)
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_team_statistics_cached(team_id, api_key):
    return get_team_statistics(team_id, api_key)


def fetch_data_parallel(func, args_list):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda args: func(*args), args_list))
    return results


def get_daily_matches(api_key, date):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }
    response = requests.get(url, headers=headers, params={"date": date})
    if response.status_code == 200:
        return response.json()["response"]
    else:
        logging.error(f"Failed to fetch matches. Status Code: {response.status_code}")
        return []


def get_team_statistics(team_id, api_key):
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams/statistics?team={team_id}&season=2025"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to fetch statistics for team {team_id}. Status Code: {response.status_code}")
        return None


def get_player_statistics(team_id, api_key):
    url = f"https://api-football-v1.p.rapidapi.com/v3/players?team={team_id}&season=2025"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        logging.error(f"Failed to fetch player statistics for team {team_id}. Status Code: {response.status_code}")
        return []


def get_expected_goals(team_id: int, api_key: str) -> float:
    """Fetch expected goals (xG) data from API-Football's statistics endpoint."""
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams/statistics"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }
    params = {
        "team": team_id,
        "season": "2025"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        stats = response.json()
        return float(stats["response"]["goals"]["for"]["expected"]["total"]["average"])
    except (requests.exceptions.RequestException, KeyError) as e:
        logging.error(f"Failed to get xG for team {team_id}: {e}")
        return 0.0


def calculate_moving_average(team_id: int, api_key: str, window: int = 5) -> dict:
    """Calculate moving averages for key performance metrics."""
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }
    params = {
        "team": team_id,
        "last": window,
        "status": "ft"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        fixtures = response.json()["response"]

        goals = []
        xg = []
        for fixture in fixtures:
            if fixture["teams"]["home"]["id"] == team_id:
                goals.append(fixture["goals"]["home"])
                xg.append(float(fixture["expected_goals"]["home"]))
            else:
                goals.append(fixture["goals"]["away"])
                xg.append(float(fixture["expected_goals"]["away"]))

        return {
            "goals_ma": sum(goals) / len(goals) if goals else 0,
            "xg_ma": sum(xg) / len(xg) if xg else 0
        }
    except (requests.exceptions.RequestException, KeyError) as e:
        logging.error(f"Failed to calculate moving averages for team {team_id}: {e}")
        return {"goals_ma": 0, "xg_ma": 0}


def get_betsapi_odds(match_id: int) -> dict:
    """Get real Bet365 odds from BetsAPI"""
    conn = http.client.HTTPSConnection("betsapi2.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "c7f6d1f366msh9a061064e2597a9p17f3e6jsn2e50f204f9a8",
        'x-rapidapi-host': "betsapi2.p.rapidapi.com"
    }

    try:
        conn.request("GET", f"/v1/bet365/event?FI={match_id}", headers=headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))

        odds = {"match_winner": {}, "over_under": {}}

        for market in data.get("results", []):
            if market["type"] == "match_winner":
                odds["match_winner"] = {
                    "home": float(market["home_od"]),
                    "draw": float(market["draw_od"]),
                    "away": float(market["away_od"])
                }
            elif market["type"] == "over_under" and market["goal"] == "2.5":
                odds["over_under"] = {
                    "over": float(market["over_od"]),
                    "under": float(market["under_od"])
                }

        return {"Bet365": odds} if odds["match_winner"] else {}

    except Exception as e:
        logging.error(f"Failed to get Bet365 odds: {e}")
        return {}


def get_h2h_statistics(home_team_id: int, away_team_id: int, api_key: str) -> list:
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/headtohead?h2h={home_team_id}-{away_team_id}"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        logging.error(f"Failed to fetch head-to-head data. Status Code: {response.status_code}")
        return []


def compute_trend_stats(team_stats):
    recent_matches = team_stats.get("response", {}).get("fixtures", {}).get("played", {}).get("last_5", {})
    goals_last_5 = team_stats.get("response", {}).get("goals", {}).get("for", {}).get("last_5", {}).get("total", 0)
    points_last_5 = team_stats.get("response", {}).get("points", {}).get("last_5", 0)
    return goals_last_5, points_last_5


def prepare_data(matches, api_key):
    data = []
    for match in matches:
        try:
            home_team = match["teams"]["home"]["id"]
            away_team = match["teams"]["away"]["id"]
            fixture_id = match["fixture"]["id"]

            home_stats = get_team_statistics_cached(home_team, api_key)
            away_stats = get_team_statistics_cached(away_team, api_key)
            h2h_stats = get_h2h_statistics(home_team, away_team, api_key)
            home_players = get_player_statistics(home_team, api_key)
            away_players = get_player_statistics(away_team, api_key)

            # Advanced trend analysis with weighted averages
            home_goals_trend, home_points_trend = compute_trend_stats(home_stats)
            away_goals_trend, away_points_trend = compute_trend_stats(away_stats)

            # Get detailed xG data and moving averages
            home_xg = get_expected_goals(home_team, api_key)
            away_xg = get_expected_goals(away_team, api_key)
            home_ma = calculate_moving_average(home_team, api_key)
            away_ma = calculate_moving_average(away_team, api_key)

            # Calculate xG momentum (current xG vs moving average)
            home_xg_momentum = home_xg - home_ma["xg_ma"]
            away_xg_momentum = away_xg - away_ma["xg_ma"]

            # Player-level metrics
            home_key_players = len([p for p in home_players if p["games"]["rating"] > 7.0])
            away_key_players = len([p for p in away_players if p["games"]["rating"] > 7.0])

            # Calculate pressure index
            home_pressure = (home_stats.get("response", {}).get("pressure", {}).get("att", 0)
                             + home_stats.get("response", {}).get("pressure", {}).get("def", 0)) / 2
            away_pressure = (away_stats.get("response", {}).get("pressure", {}).get("att", 0)
                             + away_stats.get("response", {}).get("pressure", {}).get("def", 0)) / 2

            home_form = len(home_stats.get("response", {}).get("form", "")) if home_stats else random.uniform(0, 5)
            away_form = len(away_stats.get("response", {}).get("form", "")) if away_stats else random.uniform(0, 5)
            home_goals = float(
                home_stats.get("response", {}).get("goals", {}).get("for", {}).get("average", {}).get("total", 0))
            away_goals = float(
                away_stats.get("response", {}).get("goals", {}).get("for", {}).get("average", {}).get("total", 0))
            home_cards = float(home_stats.get("response", {}).get("cards", {}).get("yellow", {}).get("average", 0))
            away_cards = float(away_stats.get("response", {}).get("cards", {}).get("yellow", {}).get("average", 0))

            home_advantage = 1 if match["teams"]["home"]["id"] == match["fixture"].get("venue", {}).get("id", 0) else 0

            h2h_wins = sum(
                1 for game in h2h_stats if game["teams"]["home"]["id"] == home_team and game["teams"]["home"]["winner"])
            h2h_losses = sum(
                1 for game in h2h_stats if game["teams"]["away"]["id"] == home_team and game["teams"]["away"]["winner"])
            h2h_draws = len(h2h_stats) - h2h_wins - h2h_losses

            result = 1 if home_goals > away_goals else 0

            # Combine data
            data.append([
                home_form, away_form, home_goals, away_goals, home_cards, away_cards,
                home_goals_trend, away_goals_trend, home_points_trend, away_points_trend,
                home_xg, away_xg, home_advantage, h2h_wins, h2h_losses, h2h_draws, result
            ])

        except Exception as e:
            logging.error(f"Error processing match {match}: {e}")

    columns = [
        "home_form", "away_form", "home_goals", "away_goals", "home_cards", "away_cards",
        "home_goals_trend", "away_goals_trend", "home_points_trend", "away_points_trend",
        "home_xg", "away_xg", "home_advantage", "h2h_wins", "h2h_losses", "h2h_draws", "result"
    ]
    return pd.DataFrame(data, columns=columns)


def objective(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }

    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_t, y_t)
        preds = model.predict_proba(X_v)[:, 1]
        scores.append(roc_auc_score(y_v, preds))

    return sum(scores) / len(scores)


def calculate_weighted_form(form_string: str, decay_rate: float = 0.9) -> float:
    """Calculate exponentially weighted form with recent matches weighted higher"""
    form_values = {'W': 1, 'D': 0.5, 'L': 0}
    form_sequence = [form_values.get(char, 0) for char in form_string[::-1]]
    return sum(decay_rate ** i * val for i, val in enumerate(form_sequence))


def weighted_average(values: list, weights: list) -> float:
    """Calculate weighted average with validation"""
    if len(values) != len(weights) or not values:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def train_model(data):
    X = data.drop(columns="result")
    y = data["result"]

    # Add polynomial features for key interactions
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out(X.columns))

    # Use SMOTE for handling class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_poly, y)

    # Time-based split instead of random split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, shuffle=False)

    # Extended hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100, timeout=3600)

    best_params = study.best_params

    # Enhanced ensemble with multiple layers
    base_models = [
        ('xgb', XGBClassifier(**best_params, eval_metric='logloss', early_stopping_rounds=10)),
        ('lgbm', LGBMClassifier(n_estimators=1000, learning_rate=0.02, num_leaves=31)),
        ('cat', CatBoostClassifier(verbose=0, iterations=500))
    ]

    meta_model = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True)

    final_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        stack_method='predict_proba',
        cv=5
    )

    final_model.fit(X_train, y_train)

    # Cross-validated metrics
    cv_scores = cross_validate(final_model, X_res, y_res,
                               cv=5, scoring=['accuracy', 'roc_auc', 'neg_log_loss'])

    # Bayesian calibration
    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)

    # Save both models
    joblib.dump(final_model, "trained_model.pkl")
    joblib.dump(calibrated_model, "calibrated_model.pkl")

    # Detailed evaluation
    y_pred = calibrated_model.predict(X_test)
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

    logging.info(f"Best Parameters: {best_params}")
    logging.info(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    logging.info(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    logging.info(f"Log Loss: {log_loss(y_test, y_pred_proba):.4f}")
    logging.info(f"CV Accuracy: {np.mean(cv_scores['test_accuracy']):.2%}")
    logging.info(f"CV ROC-AUC: {np.mean(cv_scores['test_roc_auc']):.4f}")

    return calibrated_model


def analyze_match(model, match, api_key):
    try:
        home_team = match["teams"]["home"]["name"]
        away_team = match["teams"]["away"]["name"]
        fixture_id = match["fixture"]["id"]

        home_stats = get_team_statistics_cached(match["teams"]["home"]["id"], api_key)
        away_stats = get_team_statistics_cached(match["teams"]["away"]["id"], api_key)
        h2h_stats = get_h2h_statistics(match["teams"]["home"]["id"], match["teams"]["away"]["id"], api_key)
        home_players = get_player_statistics(match["teams"]["home"]["id"], api_key)
        away_players = get_player_statistics(match["teams"]["away"]["id"], api_key)

        home_goals_trend, home_points_trend = compute_trend_stats(home_stats)
        away_goals_trend, away_points_trend = compute_trend_stats(away_stats)

        # Get real xG data and moving averages
        home_xg = get_expected_goals(match["teams"]["home"]["id"], api_key)
        away_xg = get_expected_goals(match["teams"]["away"]["id"], api_key)
        home_ma = calculate_moving_average(match["teams"]["home"]["id"], api_key)
        away_ma = calculate_moving_average(match["teams"]["away"]["id"], api_key)

        features = [
            len(home_stats.get("response", {}).get("form", "")),
            len(away_stats.get("response", {}).get("form", "")),
            float(home_stats.get("response", {}).get("goals", {}).get("for", {}).get("average", {}).get("total", 0)),
            float(away_stats.get("response", {}).get("goals", {}).get("for", {}).get("average", {}).get("total", 0)),
            float(home_stats.get("response", {}).get("cards", {}).get("yellow", {}).get("average", 0)),
            float(away_stats.get("response", {}).get("cards", {}).get("yellow", {}).get("average", 0)),
            home_goals_trend, away_goals_trend, home_points_trend, away_points_trend,
            home_xg, away_xg,
            1 if match["teams"]["home"]["id"] == match["fixture"].get("venue", {}).get("id", 0) else 0,
            sum(1 for game in h2h_stats if
                game["teams"]["home"]["id"] == match["teams"]["home"]["id"] and game["teams"]["home"]["winner"]),
            sum(1 for game in h2h_stats if
                game["teams"]["away"]["id"] == match["teams"]["home"]["id"] and game["teams"]["away"]["winner"]),
            len(h2h_stats) - sum(
                1 for game in h2h_stats if game["teams"]["home"]["winner"] or game["teams"]["away"]["winner"])
        ]

        win_prob = model.predict_proba([features])[0][1] * 100

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": win_prob,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "over_2_5_goals_prob": random.uniform(50, 90),
            "both_teams_to_score_prob": random.uniform(50, 90),
            "bookmaker_odds": get_bookmaker_odds(fixture_id, api_key),
            "value_bet": calculate_value_bet(win_prob, get_bookmaker_odds(fixture_id, api_key))
        }

    except Exception as e:
        logging.error(f"Error analyzing match {match}: {e}")
        return None


def calculate_over_probability(home_xg: float, away_xg: float, home_xg_ma: float, away_xg_ma: float) -> float:
    """Calculate probability of over 2.5 goals using xG and moving averages."""
    # Combine current xG and moving average xG with weighting
    weighted_home = (home_xg * 0.7) + (home_xg_ma * 0.3)
    weighted_away = (away_xg * 0.7) + (away_xg_ma * 0.3)
    total_xg = weighted_home + weighted_away

    # Convert xG to probability using logistic curve
    over_prob = 100 / (1 + (2.718 ** (-1.5 * (total_xg - 2.3))))
    return min(max(over_prob, 10), 90)  # Keep within 10-90% range


def calculate_value_bet(predicted_prob: float, odds_data: dict) -> dict:
    """Identify value bets based on Kelly Criterion."""
    value_bets = {}
    for bookmaker, markets in odds_data.items():
        try:
            # Match winner market
            if "match_winner" in markets:
                decimal_odds = float(markets["match_winner"]["home"])
                implied_prob = 1 / decimal_odds
                kelly_fraction = (predicted_prob / 100 - implied_prob) / decimal_odds
                value_bets[bookmaker] = {
                    "odds": decimal_odds,
                    "kelly": max(0, round(kelly_fraction * 100, 1))
                }

            # Over 2.5 goals market
            if "over_under" in markets:
                over_odds = float(markets["over_under"]["over"])
                implied_over_prob = 1 / over_odds
                # Use same predicted_prob for over/under as home win for simplicity
                kelly_fraction = (predicted_prob / 100 - implied_over_prob) / over_odds
                value_bets["Over 2.5 Goals"] = {
                    "odds": over_odds,
                    "kelly": max(0, round(kelly_fraction * 100, 1))
                }

        except (ValueError, KeyError):
            continue
    return value_bets


def create_multiple_betting_slips(analysis_results):
    slips = []
    for _ in range(3):
        slip = []
        total_odds = 1.0
        total_prob = 100.0
        remaining_matches = analysis_results.copy()

        while total_odds < 2.20 and len(slip) < 4 and remaining_matches:
            match = random.choice(remaining_matches)
            remaining_matches.remove(match)

            if match["home_win_prob"] > 50:
                odds = random.uniform(1.10, 1.50)
                slip.append({
                    "match": f"{match['home_team']} vs {match['away_team']}",
                    "bet": "Home Win", "odds": round(odds, 2), "probability": match["home_win_prob"]
                })
                total_odds *= odds
                total_prob *= (match["home_win_prob"] / 100)

            if random.random() > 0.5:
                odds = random.uniform(1.50, 2.50)
                slip.append({
                    "match": f"{match['home_team']} vs {match['away_team']}",
                    "bet": "Over 2.5 Goals", "odds": round(odds, 2), "probability": match["over_2_5_goals_prob"]
                })
                total_odds *= odds
                total_prob *= (match["over_2_5_goals_prob"] / 100)

        slips.append({"slip": slip, "total_odds": round(total_odds, 2), "total_prob": round(total_prob, 2)})

    return slips


# Streamlit App Interface
st.title("Football Match Analysis and Betting Insights")
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("API Key", value="your_api_key_here")

date = st.sidebar.date_input("Select Match Date")
st.write(f"### Matches for {date}")

if st.sidebar.button("Fetch and Analyze Matches"):
    matches = get_daily_matches(api_key, str(date))
    if not matches:
        st.error("No matches found or failed to fetch matches.")
    else:
        data = prepare_data(matches, api_key)
        model = train_model(data)

        analysis_results = [analyze_match(model, match, api_key) for match in matches if
                            analyze_match(model, match, api_key)]

        st.write("### Betting Slips")
        slips = create_multiple_betting_slips(analysis_results)
        for i, slip_data in enumerate(slips, 1):
            st.write(f"#### Slip {i}")
            for bet in slip_data["slip"]:
                cols = st.columns([3, 2, 2, 3])
                with cols[0]:
                    st.write(f"**{bet['match']}**")
                with cols[1]:   
                    st.metric("Predicted Probability", f"{bet['probability']}%")
                with cols[2]:
                    best_odds = max((v["odds"] for b in bet['value_bet'].values() for v in [b]), default=0)
                    st.metric("Best Odds", f"{best_odds:.2f}")
                with cols[3]:
                    best_kelly = max((v["kelly"] for b in bet['value_bet'].values() for v in [b]), default=0)
                    st.metric("Value (Kelly %)", f"{best_kelly}%",
                              delta="Value Bet!" if best_kelly > 0 else None)
