# ----------------------------------------------------------------------
# ⚽ Advanced Multi-Position Player Analysis App v10.1 (Enhanced Stats & Interactive Radars) ⚽
#
# Changes in this version:
# - Added statistical soundness: z-score normalization, position-specific comparisons
# - Integrated radar chart display for similar players via "Add to Radar"
# - Maintained all physical profile radars and hover interactions
# - Added minimum minutes filter (600 minutes)
# ----------------------------------------------------------------------

# --- 1. IMPORTS ---
import streamlit as st
import requests
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from datetime import date

# Plotly + HTML component for legend-hover interactivity
import plotly.graph_objects as go
import plotly.io as pio
import uuid
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')

# --- 2. APP CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Player Analysis",
    page_icon="⚽",
    layout="wide"
)

# Initialize session state
if 'radar_players' not in st.session_state:
    st.session_state.radar_players = []
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

# --- 3. CORE & POSITIONAL CONFIGURATIONS ---
# [LEAGUE_NAMES, COMPETITION_SEASONS, and ARCHETYPE definitions remain UNCHANGED]

# --- 4. DATA HANDLING & ANALYSIS FUNCTIONS (UPDATED) ---

@st.cache_resource(ttl=3600)
def get_all_leagues_data(_auth_credentials):
    """Downloads player statistics from all leagues with improved error handling"""
    # [Implementation remains UNCHANGED]

@st.cache_data(ttl=3600)
def process_data(_raw_data):
    """Processes raw data to calculate ages, position groups, and normalized metrics"""
    if _raw_data is None:
        return None

    df_processed = _raw_data.copy()
    df_processed.columns = [c.replace('player_season_', '') for c in df_processed.columns]
    
    # Clean string columns
    for col in ['player_name', 'team_name', 'league_name', 'season_name', 'primary_position']:
        if col in df_processed.columns and df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].str.strip()

    # --- Age Calculation ---
    def calculate_age(birth_date_str):
        if pd.isna(birth_date_str): return None
        try:
            birth_date = pd.to_datetime(birth_date_str).date()
            today = date.today()
            return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        except (ValueError, TypeError): return None
    df_processed['age'] = df_processed['birth_date'].apply(calculate_age)
    
    # --- Position Group Mapping ---
    def get_position_group(primary_position):
        for group, config in POSITIONAL_CONFIGS.items():
            if primary_position in config['positions']:
                return group
        return None
    df_processed['position_group'] = df_processed['primary_position'].apply(get_position_group)
    
    # --- Metric Calculation ---
    if 'padj_tackles_90' in df_processed.columns and 'padj_interceptions_90' in df_processed.columns:
        df_processed['padj_tackles_and_interceptions_90'] = (
            df_processed['padj_tackles_90'] + df_processed['padj_interceptions_90']
        )
    
    # --- Position-Specific Percentiles & Z-Scores ---
    negative_stats = ['turnovers_90', 'dispossessions_90', 'dribbled_past_90', 'fouls_90']
    
    for metric in ALL_METRICS_TO_PERCENTILE:
        if metric not in df_processed.columns:
            continue
            
        # Initialize columns
        df_processed[f'{metric}_pct'] = 0
        df_processed[f'{metric}_z'] = 0.0
        
        # Process by position group
        for group, group_df in df_processed.groupby('position_group', dropna=False):
            if group is None or len(group_df) < 5:  # Skip small groups
                continue
                
            metric_series = group_df[metric]
            
            # Percentiles
            if metric in negative_stats:
                # Invert percentiles for negative stats
                ranks = metric_series.rank(pct=True, ascending=True)
                df_processed.loc[group_df.index, f'{metric}_pct'] = (1 - ranks) * 100
            else:
                df_processed.loc[group_df.index, f'{metric}_pct'] = metric_series.rank(pct=True) * 100
            
            # Z-scores
            scaler = StandardScaler()
            z_scores = scaler.fit_transform(metric_series.values.reshape(-1, 1)).flatten()
            df_processed.loc[group_df.index, f'{metric}_z'] = z_scores

    # Final cleaning
    metric_cols = [col for col in df_processed.columns if '_90' in col or '_ratio' in col or 'length' in col]
    pct_cols = [col for col in df_processed.columns if '_pct' in col]
    z_cols = [col for col in df_processed.columns if '_z' in col]
    cols_to_clean = list(set(metric_cols + pct_cols + z_cols))
    df_processed[cols_to_clean] = df_processed[cols_to_clean].fillna(0)
    
    return df_processed

# --- 5. ANALYSIS & REPORTING FUNCTIONS (UPDATED) ---

def find_player_by_name(df, player_name):
    # [Implementation remains UNCHANGED]

def detect_player_archetype(target_player, archetypes):
    # [Implementation remains UNCHANGED]

def find_matches(target_player, pool_df, archetype_config, search_mode='similar', min_minutes=600):
    """Finds similar players using z-scores and cosine similarity"""
    key_identity_metrics = archetype_config['identity_metrics']
    key_weight = archetype_config['key_weight']
    
    # Get z-score metrics
    z_metrics = [f'{m}_z' for m in key_identity_metrics]
    target_group = target_player['position_group']
    
    # Filter pool
    pool_df = pool_df[
        (pool_df['minutes'] >= min_minutes) & 
        (pool_df['player_id'] != target_player['player_id']) & 
        (pool_df['position_group'] == target_group)
    ].copy()
    
    if pool_df.empty:
        return pd.DataFrame()

    # Prepare vectors
    target_vector = target_player[z_metrics].fillna(0).values.reshape(1, -1)
    pool_matrix = pool_df[z_metrics].fillna(0).values
    
    # Apply weights
    weights = np.full(len(z_metrics), key_weight)
    target_vector_w = target_vector * weights
    pool_matrix_w = pool_matrix * weights
    
    # Calculate similarity
    similarities = cosine_similarity(target_vector_w, pool_matrix_w)
    pool_df['similarity_score'] = similarities[0] * 100
    
    if search_mode == 'upgrade':
        # For upgrades, use average of percentiles
        pct_metrics = [f'{m}_pct' for m in key_identity_metrics]
        pool_df['upgrade_score'] = pool_df[pct_metrics].mean(axis=1)
        return pool_df.sort_values('upgrade_score', ascending=False)
    else:
        return pool_df.sort_values('similarity_score', ascending=False)

# --- 6. RADAR CHART FUNCTIONS (UNCHANGED) ---
# [_radar_angles_labels, _player_percentiles_for_metrics, 
#  _build_scatterpolar_trace, create_plotly_radar, 
#  render_plotly_with_legend_hover remain UNCHANGED]

# --- 7. STREAMLIT APP LAYOUT (UPDATED) ---
st.title("⚽ Advanced Multi-Position Player Analysis v10.1")

# Main data loading
processed_data = None
with st.spinner("Loading and processing data for all leagues..."):
    raw_data = get_all_leagues_data((USERNAME, PASSWORD))
    if raw_data is not None:
        processed_data = process_data(raw_data)
    else:
        st.error("Failed to load data. Please check credentials and connection.")

scouting_tab, comparison_tab = st.tabs(["Scouting Analysis", "Direct Comparison"])

# Player filter UI component
def create_player_filter_ui(data, key_prefix, pos_filter=None):
    # [Implementation remains UNCHANGED]

with scouting_tab:
    if processed_data is not None:
        st.sidebar.header("🔍 Scouting Controls")
        pos_options = list(POSITIONAL_CONFIGS.keys())
        selected_pos = st.sidebar.selectbox("1. Select Position", pos_options, key="scout_pos")
        filter_by_pos = st.sidebar.checkbox("Filter by position", value=True, key="pos_filter_toggle")
        
        st.sidebar.subheader("Select Target Player")
        min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 3000, 600, 100)
        pos_filter_arg = selected_pos if filter_by_pos else None
        target_player = create_player_filter_ui(processed_data, key_prefix="scout", pos_filter=pos_filter_arg)
        
        search_mode = st.sidebar.radio("Search Mode", ('Find Similar Players', 'Find Potential Upgrades'), key='scout_mode')
        search_mode_logic = 'upgrade' if search_mode == 'Find Potential Upgrades' else 'similar'

        if st.sidebar.button("Analyze Player", type="primary", key="scout_analyze") and target_player is not None:
            st.session_state.analysis_run = True
            st.session_state.target_player = target_player
            
            config = POSITIONAL_CONFIGS[selected_pos]
            archetypes = config["archetypes"]
            position_pool = processed_data[processed_data['primary_position'].isin(config['positions'])]

            detected_archetype, dna_df = detect_player_archetype(target_player, archetypes)
            st.session_state.detected_archetype = detected_archetype
            st.session_state.dna_df = dna_df
            
            if detected_archetype:
                archetype_config = archetypes[detected_archetype]
                matches = find_matches(
                    target_player, 
                    position_pool, 
                    archetype_config, 
                    search_mode_logic,
                    min_minutes
                )
                st.session_state.matches = matches
            else:
                st.session_state.matches = pd.DataFrame()

        if st.session_state.analysis_run and 'target_player' in st.session_state:
            tp = st.session_state.target_player
            st.header(f"Analysis: {tp['player_name']} ({tp['season_name']})")
            
            if st.session_state.detected_archetype:
                st.subheader(f"Detected Archetype: {st.session_state.detected_archetype}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(st.session_state.dna_df.reset_index(drop=True), hide_index=True)
                with col2:
                    st.write(f"**Description**: {POSITIONAL_CONFIGS[selected_pos]['archetypes'][st.session_state.detected_archetype]['description']}")

                st.subheader(f"Top 10 Matches ({search_mode})")
                if not st.session_state.matches.empty:
                    display_cols = ['player_name', 'age', 'team_name', 'league_name', 'season_name']
                    score_col = 'upgrade_score' if search_mode_logic == 'upgrade' else 'similarity_score'
                    display_cols.insert(2, score_col)
                    
                    matches_display = st.session_state.matches.head(10)[display_cols].copy()
                    matches_display[score_col] = matches_display[score_col].round(1)
                    st.dataframe(matches_display.rename(columns=lambda c: c.replace('_', ' ').title()), hide_index=True)
                    
                    # Add to radar buttons
                    st.subheader("Add Players to Radar")
                    for i, row in st.session_state.matches.head(10).iterrows():
                        btn_key = f"add_{row['player_id']}_{row['season_id']}"
                        if st.button(f"Add {row['player_name']} to Radar", key=btn_key):
                            # Check if already added
                            if not any(p['player_id'] == row['player_id'] and 
                                      p['season_id'] == row['season_id'] 
                                      for p in st.session_state.radar_players):
                                st.session_state.radar_players.append(row)
                                st.rerun()
                else:
                    st.warning("No matching players found")

            # Display radar players
            if st.session_state.radar_players:
                st.subheader("Players on Radar")
                radar_cols = st.columns(len(st.session_state.radar_players) or 1)
                for i, player_data in enumerate(st.session_state.radar_players):
                    with radar_cols[i]:
                        st.markdown(f"**{player_data['player_name']}**")
                        st.markdown(f"{player_data['team_name']} | {player_data['league_name']}")
                        st.markdown(f"`{player_data['season_name']}`")
                        if st.button("❌ Remove", key=f"remove_{i}"):
                            st.session_state.radar_players.pop(i)
                            st.rerun()

            # Display radar charts for selected players
            if st.session_state.radar_players or st.session_state.analysis_run:
                st.subheader("Player Radars")
                players_to_show = [st.session_state.target_player] + st.session_state.radar_players
                radars_to_show = POSITIONAL_CONFIGS[selected_pos]['radars']
                
                # Layout: 3 columns per row
                num_radars = len(radars_to_show)
                cols = st.columns(3) 
                radar_items = list(radars_to_show.items())

                for i in range(num_radars):
                    with cols[i % 3]:
                        radar_key, radar_config = radar_items[i]
                        fig, metrics = create_plotly_radar(players_to_show, radar_config)
                        render_plotly_with_legend_hover(fig, metrics, height=520)
        else:
            st.info("Select a position and target player to begin analysis")
    else:
        st.error("Data could not be loaded. Please check your credentials.")

with comparison_tab:
    # [Implementation remains UNCHANGED as per original]
