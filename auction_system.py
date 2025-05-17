# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random

# Initialize session state
if 'auction_data' not in st.session_state:
    st.session_state.auction_data = None
if 'squad' not in st.session_state:
    st.session_state.squad = []
if 'budget' not in st.session_state:
    st.session_state.budget = 100.0  # in crores
if 'failed_players' not in st.session_state:
    st.session_state.failed_players = []

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cricket_data_2025.csv')
    except FileNotFoundError:
        st.error("Error: 'cricket_data_2025.csv' not found. Please ensure the file is in the correct directory.")
        st.stop()

    numeric_columns = [
        "Matches_Batted", "Not_Outs", "Runs_Scored", "Batting_Average", "Balls_Faced", "Batting_Strike_Rate",
        "Centuries", "Half_Centuries", "Fours", "Sixes", "Catches_Taken", "Stumpings", "Matches_Bowled",
        "Balls_Bowled", "Runs_Conceded", "Wickets_Taken", "Bowling_Average", "Economy_Rate", "Bowling_Strike_Rate",
        "Four_Wicket_Hauls", "Five_Wicket_Hauls"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    def assign_role(row):
        if row['Stumpings'] > 0:
            return "Wicketkeeper"
        elif row['Runs_Scored'] > 100 and row['Wickets_Taken'] > 5:
            return "All-Rounder"
        elif row['Wickets_Taken'] > 5:
            return "Bowler"
        elif row['Runs_Scored'] > 100:
            return "Batsman"
        return "Unknown"

    df['Role'] = df.apply(assign_role, axis=1)
    foreign_players = ["Aaron Hardie", "Pat Cummins", "David Warner", "Andre Russell", "Rashid Khan"]
    df['Foreign'] = df['Player_Name'].apply(lambda x: 1 if x in foreign_players else 0)

    return df

def calculate_ratings(df):
    def calculate_rating(row):
        batting_metrics = ['Batting_Average', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries', 'Fours', 'Sixes']
        batting_weights = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]
        batting_score = sum(weight * (row[metric] / df[metric].max()) if df[metric].max() > 0 else 0
                            for metric, weight in zip(batting_metrics, batting_weights))

        bowling_metrics = ['Wickets_Taken', 'Four_Wicket_Hauls', 'Five_Wicket_Hauls']
        bowling_weights = [0.6, 0.2, 0.2]
        bowling_score = sum(weight * (row[metric] / df[metric].max()) if df[metric].max() > 0 else 0
                            for metric, weight in zip(bowling_metrics, bowling_weights))

        if row['Bowling_Average'] > 0:
            bowling_score += 0.2 * (1 - row['Bowling_Average'] / df['Bowling_Average'].max())
        if row['Economy_Rate'] > 0:
            bowling_score += 0.2 * (1 - row['Economy_Rate'] / df['Economy_Rate'].max())

        fielding_score = 0.5 * (row['Catches_Taken'] / df['Catches_Taken'].max() if df['Catches_Taken'].max() > 0 else 0)
        fielding_score += 0.5 * (row['Stumpings'] / df['Stumpings'].max() if df['Stumpings'].max() > 0 else 0)

        if row['Role'] == 'Batsman':
            return 0.8 * batting_score + 0.2 * fielding_score
        elif row['Role'] == 'Bowler':
            return 0.8 * bowling_score + 0.2 * fielding_score
        elif row['Role'] == 'All-Rounder':
            return 0.4 * batting_score + 0.4 * bowling_score + 0.2 * fielding_score
        elif row['Role'] == 'Wicketkeeper':
            return 0.6 * batting_score + 0.4 * fielding_score
        else:
            return 0

    df['Rating'] = df.apply(calculate_rating, axis=1)
    df['Rating'] = df['Rating'] * 100
    return df

def predict_prices(df):
    X = df[['Rating', 'Matches_Batted', 'Matches_Bowled']]

    def assign_price(rating):
        if rating > 90:
            base_price = random.uniform(10, 20) * 10000000
        elif rating > 70:
            base_price = random.uniform(2, 10) * 10000000
        else:
            base_price = random.uniform(0.5, 2) * 10000000
        noise = random.uniform(-0.1, 0.1) * base_price
        return np.clip(base_price + noise, 0.5 * 10000000, 20 * 10000000)

    df['Price_INR'] = df['Rating'].apply(assign_price)
    y = df['Price_INR']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df['Predicted_Price_INR'] = model.predict(X)
    df['Predicted_Price_Crore'] = df['Predicted_Price_INR'] / 10000000
    df['Min_Price_Crore'] = df['Predicted_Price_Crore'] * 0.8
    df['Max_Price_Crore'] = df['Predicted_Price_Crore'] * 1.2

    return df

def get_target_players(df, budget, squad):
    ROLE_CONSTRAINTS = {"Batsman": 4, "Bowler": 3, "All-Rounder": 2, "Wicketkeeper": 2}
    available_players = df[~df['Player_Name'].isin([p['Player_Name'] for p in squad])]
    role_counts = {role: 0 for role in ROLE_CONSTRAINTS}
    for player in squad:
        if player['Role'] in role_counts:
            role_counts[player['Role']] += 1
    remaining_needs = {role: max(0, ROLE_CONSTRAINTS[role] - role_counts[role]) for role in ROLE_CONSTRAINTS}

    target_players = []
    for role, need in remaining_needs.items():
        if need > 0:
            role_players = available_players[available_players['Role'] == role]
            role_players = role_players.sort_values('Rating', ascending=False).head(need * 2)
            target_players.extend(role_players.to_dict('records'))

    high_rated = available_players.sort_values('Rating', ascending=False).head(5)
    target_players.extend(high_rated.to_dict('records'))

    seen_names = set()
    unique_players = []
    for player in target_players:
        if player['Player_Name'] not in seen_names:
            seen_names.add(player['Player_Name'])
            unique_players.append(player)

    return unique_players[:10]

def get_replacement_options(df, failed_player, budget, squad):
    same_role = df[(df['Role'] == failed_player['Role']) &
                   (~df['Player_Name'].isin([p['Player_Name'] for p in squad])) &
                   (df['Predicted_Price_Crore'] <= failed_player['Predicted_Price_Crore'] * 1.1)]
    same_role = same_role.sort_values('Rating', ascending=False)
    budget_options = df[(~df['Player_Name'].isin([p['Player_Name'] for p in squad])) &
                        (df['Predicted_Price_Crore'] <= budget)].sort_values('Rating', ascending=False)
    replacements = pd.concat([same_role, budget_options]).drop_duplicates('Player_Name')
    return replacements.head(5).to_dict('records')

def add_player_to_squad(player, actual_price):
    if actual_price <= st.session_state.budget:
        player_copy = player.copy()
        player_copy['Actual_Price_Crore'] = actual_price
        st.session_state.squad.append(player_copy)
        st.session_state.budget -= actual_price
        st.success(f"Added {player['Player_Name']} to squad for ₹{actual_price:.2f} crore")
    else:
        st.error(f"Cannot add {player['Player_Name']} - not enough budget")

def mark_player_failed(player):
    st.session_state.failed_players.append(player)
    st.warning(f"Failed to acquire {player['Player_Name']} - showing replacement options")

def main():
    st.title("🏏 IPL Auction Simulator")
    st.write("A real-time auction system for IPL team selection")

    if st.session_state.auction_data is None:
        with st.spinner("Loading and processing player data..."):
            df = load_data()
            df = calculate_ratings(df)
            df = predict_prices(df)
            st.session_state.auction_data = df

    df = st.session_state.auction_data
    budget = st.session_state.budget
    squad = st.session_state.squad

    st.subheader("💰 Current Budget")
    st.write(f"Remaining Budget: ₹{budget:.2f} crore")

    st.subheader("👥 Current Squad")
    if squad:
        squad_df = pd.DataFrame(squad)[['Player_Name', 'Role', 'Rating', 'Actual_Price_Crore']]
        st.dataframe(squad_df)
        role_counts = pd.DataFrame(squad)['Role'].value_counts().reset_index()
        role_counts.columns = ['Role', 'Count']
        st.bar_chart(role_counts.set_index('Role'))
    else:
        st.write("No players in squad yet")

    st.subheader("🎯 Target Players")
    targets = get_target_players(df, budget, squad)

    for player in targets:
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
        with col1:
            st.write(f"**{player['Player_Name']}** ({player['Role']})")
            st.write(f"Rating: {player['Rating']:.1f}")
        with col2:
            st.write(f"Price Range: ₹{player['Min_Price_Crore']:.2f} - ₹{player['Max_Price_Crore']:.2f} crore")
            actual_price = st.number_input(f"Enter price for {player['Player_Name'].split()[0]}",
                                           min_value=0.1,
                                           max_value=player['Max_Price_Crore'] * 2,
                                           value=round(player['Predicted_Price_Crore'], 2),
                                           key=f"price_input_{player['Player_Name']}")
        with col3:
            if st.button(f"Buy {player['Player_Name'].split()[0]}", key=f"buy_{player['Player_Name']}"):
                add_player_to_squad(player, actual_price)
                st.rerun()
        with col4:
            if st.button(f"Fail {player['Player_Name'].split()[0]}", key=f"fail_{player['Player_Name']}"):
                mark_player_failed(player)
                st.rerun()

    if st.session_state.failed_players:
        st.subheader("🔄 Replacement Options")
        for failed_player in st.session_state.failed_players:
            st.write(f"❌ Failed to buy: **{failed_player['Player_Name']}** ({failed_player['Role']})")
        for failed_player in st.session_state.failed_players:
            st.write(f"Replacements for {failed_player['Player_Name']} ({failed_player['Role']}):")
            replacements = get_replacement_options(df, failed_player, budget, squad)

            for replacement in replacements:
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.write(f"**{replacement['Player_Name']}** ({replacement['Role']})")
                    st.write(f"Rating: {replacement['Rating']:.1f}")
                with col2:
                    st.write(f"Price: ₹{replacement['Predicted_Price_Crore']:.2f} crore")
                with col3:
                    actual_price = st.number_input(f"Enter price for {replacement['Player_Name'].split()[0]}",
                                                   min_value=0.1,
                                                   max_value=replacement['Max_Price_Crore'] * 2,
                                                   value=round(replacement['Predicted_Price_Crore'], 2),
                                                   key=f"replace_price_input_{replacement['Player_Name']}")
                    if st.button(f"Add {replacement['Player_Name'].split()[0]}",
                                 key=f"replace_{replacement['Player_Name']}"):
                        add_player_to_squad(replacement, actual_price)
                        st.rerun()

    if st.button("Reset Auction"):
        st.session_state.squad = []
        st.session_state.budget = 100.0
        st.session_state.failed_players = []
        st.rerun()

# Remaining players tracking
    st.subheader("📋 Remaining Available Players")
    bought_names = [p['Player_Name'] for p in st.session_state.squad]
    failed_names = [p['Player_Name'] for p in st.session_state.failed_players]
    target_names = [p['Player_Name'] for p in get_target_players(df, budget, squad)]
    remaining_players = df[~df['Player_Name'].isin(bought_names + failed_names + target_names)]
    st.dataframe(remaining_players[['Player_Name', 'Role', 'Rating', 'Predicted_Price_Crore']].sort_values(by='Rating', ascending=False))

    # Optional: Export remaining players to CSV
    csv = remaining_players.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Remaining Players CSV",
        data=csv,
        file_name='remaining_players.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    main()
