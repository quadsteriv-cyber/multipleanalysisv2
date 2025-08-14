# ----------------------------------------------------------------------
# ⚽ Advanced Multi-Position Player Analysis App v10.1 (Enhanced Stats & Interactive Radars) ⚽
#
# Changes in this version:
# - Added statistical soundness: z-score normalization, position-specific comparisons
# - Integrated radar chart display for similar players via "Add to Radar"
# - Maintained all physical profile radars and hover interactions
# - Added minimum minutes filter (600 minutes)
# - Fixed session state initialization issues
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

# Initialize session state variables
if 'comp_selections' not in st.session_state:
    st.session_state.comp_selections = {"league": None, "season": None, "team": None, "player": None}
if 'comparison_players' not in st.session_state:
    st.session_state.comparison_players = []
if 'radar_players' not in st.session_state:
    st.session_state.radar_players = []
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'target_player' not in st.session_state:
    st.session_state.target_player = None
if 'detected_archetype' not in st.session_state:
    st.session_state.detected_archetype = None
if 'dna_df' not in st.session_state:
    st.session_state.dna_df = None
if 'matches' not in st.session_state:
    st.session_state.matches = None

# --- 3. CORE & POSITIONAL CONFIGURATIONS ---

USERNAME = "quadsteriv@gmail.com"
PASSWORD = "SfORY1xR"

LEAGUE_NAMES = {
    4: "League One", 5: "League Two", 51: "Premiership", 65: "National League",
    76: "Liga", 78: "1. HNL", 89: "USL Championship", 106: "Veikkausliiga",
    107: "Premier Division", 129: "Championnat National", 166: "Premier League 2 Division One",
    179: "3. Liga", 260: "1st Division", 1035: "First Division B", 1385: "Championship",
    1442: "1. Division", 1581: "2. Liga", 1607: "Úrvalsdeild", 1778: "First Division",
    1848: "I Liga", 1865: "First League"
}

COMPETITION_SEASONS = {
    4: [235, 281, 317, 318],
    5: [235, 281, 317, 318],
    51: [235, 281, 317, 318],
    65: [281, 318],
    76: [317, 318],
    78: [317, 318],
    89: [106, 107, 282, 315],
    106: [315],
    107: [106, 107, 282, 315],
    129: [317, 318],
    166: [318],
    179: [317, 318],
    260: [317, 318],
    1035: [317, 318],
    1385: [235, 281, 317, 318],
    1442: [107, 282, 315],
    1581: [317, 318],
    1607: [315],
    1778: [282, 315],
    1848: [281, 317, 318],
    1865: [318]
}

# Archetype definitions
STRIKER_ARCHETYPES = {
    "Poacher (Fox in the Box)": {
        "description": "Clinical finisher, thrives in the penalty area, instinctive movement, minimal involvement in build-up.",
        "identity_metrics": ['npg_90', 'np_xg_90', 'conversion_ratio', 'touches_inside_box_90', 'shot_touch_ratio', 'np_xg_per_shot'],
        "key_weight": 1.7
    },
    "Target Man": {
        "description": "Strong aerial presence, holds up the ball, physical dominance.",
        "identity_metrics": ['aerial_wins_90', 'aerial_ratio', 'fouls_won_90', 'op_xgbuildup_90', 'touches_inside_box_90', 'passes_into_box_90'],
        "key_weight": 1.6
    },
    "Complete Forward": {
        "description": "Well-rounded—good finishing, dribbling, link-up, and movement.",
        "identity_metrics": ['npg_90', 'key_passes_90', 'dribbles_90', 'deep_progressions_90', 'op_xgbuildup_90', 'aerial_wins_90'],
        "key_weight": 1.6
    },
    "False 9": {
        "description": "Drops deep into midfield, playmaker-like vision, technical excellence.",
        "identity_metrics": ['op_xgbuildup_90', 'key_passes_90', 'through_balls_90', 'dribbles_90', 'carries_90', 'xa_90'],
        "key_weight": 1.5
    },
    "Advanced Forward": {
        "description": "Prioritizes runs in behind, thrives on through balls, pace-driven.",
        "identity_metrics": ['deep_progressions_90', 'through_balls_90', 'np_shots_90', 'touches_inside_box_90', 'npg_90', 'np_xg_90'],
        "key_weight": 1.6
    },
    "Pressing Forward": {
        "description": "Defensive work rate, triggers press, harasses defenders.",
        "identity_metrics": ['pressures_90', 'pressure_regains_90', 'counterpressures_90', 'aggressive_actions_90', 'padj_tackles_90', 'fouls_90'],
        "key_weight": 1.5
    },
    "Second Striker (Support Striker)": {
        "description": "Operates just behind main striker, creative link, dribbler.",
        "identity_metrics": ['dribbles_90', 'key_passes_90', 'xa_90', 'touches_inside_box_90', 'npg_90', 'carries_90'],
        "key_weight": 1.5
    },
    "Deep-Lying Forward": {
        "description": "Drops into midfield to orchestrate play, but still a striker.",
        "identity_metrics": ['op_xgbuildup_90', 'key_passes_90', 'long_balls_90', 'through_balls_90', 'carries_90', 'passing_ratio'],
        "key_weight": 1.5
    },
    "Wide Forward": {
        "description": "Starts wide but cuts inside; often part of a front two or fluid front three.",
        "identity_metrics": ['dribbles_90', 'crosses_90', 'deep_progressions_90', 'touches_inside_box_90', 'npg_90', 'np_shots_90'],
        "key_weight": 1.6
    }
}

# Radar metrics
STRIKER_RADAR_METRICS = {
    'finishing': {
        'name': 'Finishing', 'color': '#D32F2F',
        'metrics': {
            'npg_90': 'Non-Penalty Goals', 'np_xg_90': 'Non-Penalty xG',
            'np_shots_90': 'Shots p90', 'conversion_ratio': 'Shot Conversion %',
            'np_xg_per_shot': 'Avg. Shot Quality', 'touches_inside_box_90': 'Touches in Box p90'
        }
    },
    'box_presence': {
        'name': 'Box Presence', 'color': '#AF1D1D',
        'metrics': {
            'touches_inside_box_90': 'Touches in Box p90',
            'passes_inside_box_90': 'Passes in Box p90',
            'positive_outcome_90': 'Positive Outcomes p90',
            'shot_touch_ratio': 'Shot/Touch %',
            'op_passes_into_box_90': 'Passes into Box p90',
            'np_xg_per_shot': 'Avg. Shot Quality'
        }
    },
    'creation': {
        'name': 'Creation & Link-Up', 'color': '#FF6B35',
        'metrics': {
            'key_passes_90': 'Key Passes p90', 'xa_90': 'xA p90',
            'op_passes_into_box_90': 'Passes into Box p90', 'through_balls_90': 'Through Balls p90',
            'op_xgbuildup_90': 'xG Buildup p90', 'passing_ratio': 'Pass Completion %'
        }
    },
    'dribbling': {
        'name': 'Dribbling & Carrying', 'color': '#9C27B0',
        'metrics': {
            'dribbles_90': 'Successful Dribbles p90', 'dribble_ratio': 'Dribble Success %',
            'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
            'turnovers_90': 'Ball Security (Inv)', 'deep_progressions_90': 'Deep Progressions p90'
        }
    },
    # PHYSICAL profile
    'aerial': {
        'name': 'Aerial Prowess', 'color': '#607D8B',
        'metrics': {
            'aerial_wins_90': 'Aerial Duels Won p90', 'aerial_ratio': 'Aerial Win %',
            'aggressive_actions_90': 'Aggressive Actions p90', 'challenge_ratio': 'Defensive Duel Win %',
            'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
            'fouls_won_90': 'Fouls Won p90'
        }
    },
    'defensive': {
        'name': 'Defensive Contribution', 'color': '#4CAF50',
        'metrics': {
            'pressures_90': 'Pressures p90', 'pressure_regains_90': 'Pressure Regains p90',
            'counterpressures_90': 'Counterpressures p90', 'aggressive_actions_90': 'Aggressive Actions',
            'padj_tackles_90': 'P.Adj Tackles p90', 'dribbled_past_90': 'Times Dribbled Past p90'
        }
    }
}

WINGER_ARCHETYPES = {
    "Goal-Scoring Winger": {
        "description": "A winger focused on cutting inside to shoot and score goals.",
        "identity_metrics": ['npg_90', 'np_xg_90', 'np_shots_90', 'touches_inside_box_90', 'np_xg_per_shot', 'dribbles_90'],
        "key_weight": 1.6
    },
    "Creative Playmaker": {
        "description": "A winger who creates chances for others through key passes and assists.",
        "identity_metrics": ['xa_90', 'key_passes_90', 'op_passes_into_box_90', 'through_balls_90', 'op_xgbuildup_90', 'deep_progressions_90'],
        "key_weight": 1.5
    },
    "Traditional Winger": {
        "description": "Focuses on providing width, dribbling down the line, and delivering crosses.",
        "identity_metrics": ['crosses_90', 'crossing_ratio', 'dribbles_90', 'carry_length', 'deep_progressions_90', 'fouls_won_90'],
        "key_weight": 1.5
    }
}

WINGER_RADAR_METRICS = {
    'goal_threat': {
        'name': 'Goal Threat', 'color': '#D32F2F',
        'metrics': {
            'npg_90': 'Non-Penalty Goals', 'np_xg_90': 'Non-Penalty xG',
            'np_shots_90': 'Shots p90', 'touches_inside_box_90': 'Touches in Box p90',
            'conversion_ratio': 'Shot Conversion %', 'np_xg_per_shot': 'Avg. Shot Quality'
        }
    },
    'creation': {
        'name': 'Chance Creation', 'color': '#FF6B35',
        'metrics': {
            'key_passes_90': 'Key Passes p90', 'xa_90': 'xA p90',
            'op_passes_into_box_90': 'Passes into Box p90', 'through_balls_90': 'Through Balls p90',
            'op_xgbuildup_90': 'xG Buildup p90', 'passing_ratio': 'Pass Completion %'
        }
    },
    'progression': {
        'name': 'Dribbling & Progression', 'color': '#9C27B0',
        'metrics': {
            'dribbles_90': 'Successful Dribbles p90', 'dribble_ratio': 'Dribble Success %',
            'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
            'deep_progressions_90': 'Deep Progressions p90', 'fouls_won_90': 'Fouls Won p90'
        }
    },
    'crossing': {
        'name': 'Crossing Profile', 'color': '#00BCD4',
        'metrics': {
            'crosses_90': 'Completed Crosses p90', 'crossing_ratio': 'Cross Completion %',
            'box_cross_ratio': '% of Box Passes that are Crosses', 'op_passes_into_box_90': 'Passes into Box p90',
            'key_passes_90': 'Key Passes p90', 'xa_90': 'xA p90'
        }
    },
    'defensive': {
        'name': 'Defensive Work Rate', 'color': '#4CAF50',
        'metrics': {
            'pressures_90': 'Pressures p90', 'pressure_regains_90': 'Pressure Regains p90',
            'padj_tackles_90': 'P.Adj Tackles p90', 'padj_interceptions_90': 'P.Adj Interceptions p90',
            'dribbled_past_90': 'Times Dribbled Past p90', 'aggressive_actions_90': 'Aggressive Actions'
        }
    },
    # PHYSICAL profile
    'duels': {
        'name': 'Duels & Security', 'color': '#607D8B',
        'metrics': {
            'aerial_wins_90': 'Aerial Duels Won p90', 'aerial_ratio': 'Aerial Win %',
            'challenge_ratio': 'Defensive Duel Win %', 'fouls_won_90': 'Fouls Won p90',
            'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
            'turnovers_90': 'Ball Security (Inv)'
        }
    }
}

CM_ARCHETYPES = {
    "Deep-Lying Playmaker (Regista)": {
        "description": "Dictates tempo from deep, excels in progressive passing.",
        "identity_metrics": ['op_xgbuildup_90', 'long_balls_90', 'long_ball_ratio', 'forward_pass_proportion', 'passing_ratio', 'through_balls_90'],
        "key_weight": 1.6
    },
    "Box-to-Box Midfielder (B2B)": {
        "description": "Covers large vertical space, contributes in both boxes.",
        "identity_metrics": ['deep_progressions_90', 'carries_90', 'padj_tackles_and_interceptions_90', 'pressures_90', 'npg_90', 'touches_inside_box_90'],
        "key_weight": 1.6
    },
    "Ball-Winning Midfielder (Destroyer)": {
        "description": "Breaks up play, screens defense.",
        "identity_metrics": ['padj_tackles_90', 'padj_interceptions_90', 'pressure_regains_90', 'challenge_ratio', 'aggressive_actions_90', 'fouls_90'],
        "key_weight": 1.6
    },
    "Advanced Playmaker (Mezzala)": {
        "description": "Operates in half-spaces, creates in advanced zones.",
        "identity_metrics": ['xa_90', 'key_passes_90', 'op_passes_into_box_90', 'through_balls_90', 'dribbles_90', 'np_shots_90'],
        "key_weight": 1.5
    },
    "Transition Midfielder (Tempo Carrier)": {
        "description": "Drives forward in transition, breaks lines with carries.",
        "identity_metrics": ['carries_90', 'carry_length', 'dribbles_90', 'dribble_ratio', 'deep_progressions_90', 'fouls_won_90'],
        "key_weight": 1.5
    },
    "Holding Midfielder (Anchor)": {
        "description": "Protects the backline, distributes safely.",
        "identity_metrics": ['padj_interceptions_90', 'passing_ratio', 'op_xgbuildup_90', 'pressures_90', 'challenge_ratio', 'turnovers_90'],
        "key_weight": 1.5
    },
    "Attacking Midfielder (8.5 Role)": {
        "description": "Focused on final-third involvement.",
        "identity_metrics": ['npg_90', 'np_xg_90', 'xa_90', 'key_passes_90', 'touches_inside_box_90', 'np_shots_90'],
        "key_weight": 1.6
    }
}

CM_RADAR_METRICS = {
    'defending': {
        'name': 'Defensive Actions', 'color': '#D32F2F',
        'metrics': {
            'padj_tackles_and_interceptions_90': 'P.Adj Tackles+Ints',
            'challenge_ratio': 'Defensive Duel Win %',
            'dribbled_past_90': 'Times Dribbled Past p90',
            'aggressive_actions_90': 'Aggressive Actions',
            'pressures_90': 'Pressures p90'
        }
    },
    # PHYSICAL profile
    'duels': {
        'name': 'Duels & Physicality', 'color': '#AF1D1D',
        'metrics': {
            'aerial_wins_90': 'Aerial Duels Won', 'aerial_ratio': 'Aerial Win %',
            'fouls_won_90': 'Fouls Won', 'challenge_ratio': 'Defensive Duel Win %',
            'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
            'aggressive_actions_90': 'Aggressive Actions'
        }
    },
    'passing': {
        'name': 'Passing & Distribution', 'color': '#0066CC',
        'metrics': {
            'passing_ratio': 'Pass Completion %', 'forward_pass_proportion': 'Forward Pass %',
            'long_balls_90': 'Long Balls p90', 'long_ball_ratio': 'Long Ball Accuracy %',
            'op_xgbuildup_90': 'xG Buildup p90'
        }
    },
    'creation': {
        'name': 'Creativity & Creation', 'color': '#FF6B35',
        'metrics': {
            'key_passes_90': 'Key Passes p90', 'xa_90': 'xA p90',
            'through_balls_90': 'Through Balls p90', 'op_xgbuildup_90': 'xG Buildup p90',
            'op_passes_into_box_90': 'Passes into Box p90'
        }
    },
    'progression': {
        'name': 'Ball Progression', 'color': '#4CAF50',
        'metrics': {
            'deep_progressions_90': 'Deep Progressions', 'carries_90': 'Ball Carries p90',
            'carry_length': 'Avg. Carry Length', 'dribbles_90': 'Successful Dribbles',
            'dribble_ratio': 'Dribble Success %'
        }
    },
    'attacking': {
        'name': 'Attacking Output', 'color': '#9C27B0',
        'metrics': {
            'npg_90': 'Non-Penalty Goals', 'np_xg_90': 'Non-Penalty xG',
            'np_shots_90': 'Shots p90', 'touches_inside_box_90': 'Touches in Box',
            'np_xg_per_shot': 'Avg. Shot Quality'
        }
    }
}

FULLBACK_ARCHETYPES = {
    "Attacking Fullback": {
        "description": "High attacking output with crosses, key passes, and forward runs into the final third.",
        "identity_metrics": ['xa_90', 'crosses_90', 'op_passes_into_box_90', 'deep_progressions_90', 'key_passes_90', 'op_xgbuildup_90'],
        "key_weight": 1.5
    },
    "Defensive Fullback": {
        "description": "Solid defensive foundation with tackles, interceptions, and aerial duels.",
        "identity_metrics": ['padj_tackles_and_interceptions_90', 'challenge_ratio', 'aggressive_actions_90', 'pressures_90', 'aerial_wins_90', 'aerial_ratio'],
        "key_weight": 1.5
    },
    "Modern Wingback": {
        "description": "High energy player who covers huge distances, contributing in all phases of play.",
        "identity_metrics": ['deep_progressions_90', 'crosses_90', 'dribbles_90', 'padj_tackles_and_interceptions_90', 'pressures_90', 'xa_90'],
        "key_weight": 1.6
    }
}

FULLBACK_RADAR_METRICS = {
    'defensive_actions': {
        'name': 'Defensive Actions', 'color': '#00BCD4',
        'metrics': {
            'padj_tackles_and_interceptions_90': 'P.Adj Tackles+Ints p90',
            'challenge_ratio': 'Defensive Duel Win %',
            'dribbled_past_90': 'Times Dribbled Past p90',
            'pressures_90': 'Pressures p90',
            'aggressive_actions_90': 'Aggressive Actions p90'
        }
    },
    # PHYSICAL profile
    'duels': {
        'name': 'Duels', 'color': '#008294',
        'metrics': {
            'aerial_wins_90': 'Aerial Duels Won p90', 'aerial_ratio': 'Aerial Win %',
            'aggressive_actions_90': 'Aggressive Actions p90', 'fouls_won_90': 'Fouls Won p90',
            'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length'
        }
    },
    'progression_creation': {
        'name': 'Progression & Creation', 'color': '#FF6B35',
        'metrics': {
            'deep_progressions_90': 'Deep Progressions p90', 'carries_90': 'Ball Carries p90',
            'dribbles_90': 'Successful Dribbles p90', 'xa_90': 'xA p90',
            'op_passes_into_box_90': 'Passes into Box p90'
        }
    },
    'crossing': {
        'name': 'Crossing', 'color': '#FFA735',
        'metrics': {
            'crosses_90': 'Completed Crosses p90', 'crossing_ratio': 'Cross Completion %',
            'box_cross_ratio': '% of Box Passes that are Crosses', 'key_passes_90': 'Key Passes p90'
        }
    },
    'passing': {
        'name': 'Passing & Buildup', 'color': '#9C27B0',
        'metrics': {
            'passing_ratio': 'Pass Completion %', 'op_xgbuildup_90': 'xG Buildup p90',
            'key_passes_90': 'Key Passes p90', 'forward_pass_proportion': 'Forward Pass %'
        }
    },
    'work_rate': {
        'name': 'Work Rate & Security', 'color': '#4CAF50',
        'metrics': {
            'pressures_90': 'Pressures p90', 'pressure_regains_90': 'Pressure Regains p90',
            'turnovers_90': 'Ball Security (Inv)', 'dribbled_past_90': 'Times Dribbled Past p90'
        }
    }
}

CB_ARCHETYPES = {
    "Ball-Playing Defender": {
        "description": "Comfortable in possession, initiates attacks from the back with progressive passing.",
        "identity_metrics": ['op_xgbuildup_90', 'passing_ratio', 'long_balls_90', 'long_ball_ratio', 'forward_pass_proportion', 'carries_90'],
        "key_weight": 1.5
    },
    "Stopper": {
        "description": "Aggressive defender who steps out to challenge attackers and win the ball high up the pitch.",
        "identity_metrics": ['aggressive_actions_90', 'padj_tackles_90', 'challenge_ratio', 'pressures_90', 'aerial_wins_90', 'fouls_90'],
        "key_weight": 1.6
    },
    "Covering Defender": {
        "description": "Reads the game well, relying on positioning and interceptions to sweep up behind the defensive line.",
        "identity_metrics": ['padj_interceptions_90', 'padj_clearances_90', 'dribbled_past_90', 'pressure_regains_90', 'aerial_ratio', 'passing_ratio'],
        "key_weight": 1.5
    }
}

CB_RADAR_METRICS = {
    'ground_defending': {
        'name': 'Ground Duels', 'color': '#D32F2F',
        'metrics': {
            'padj_tackles_90': 'PAdj Tackles', 'challenge_ratio': 'Challenge Success %',
            'aggressive_actions_90': 'Aggressive Actions', 'pressures_90': 'Pressures p90'
        }
    },
    # PHYSICAL profile
    'aerial_duels': {
        'name': 'Aerial Duels & Clearances', 'color': '#4CAF50',
        'metrics': {
            'aerial_wins_90': 'Aerial Duels Won', 'aerial_ratio': 'Aerial Win %',
            'padj_clearances_90': 'PAdj Clearances', 'fouls_won_90': 'Fouls Won',
            'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length'
        }
    },
    'passing_distribution': {
        'name': 'Passing & Distribution', 'color': '#0066CC',
        'metrics': {
            'passing_ratio': 'Pass Completion %', 'pass_length': 'Avg. Pass Length',
            'long_balls_90': 'Long Balls p90', 'long_ball_ratio': 'Long Ball Accuracy %',
            'forward_pass_proportion': 'Forward Pass %'
        }
    },
    'ball_progression': {
        'name': 'Ball Progression', 'color': '#FFC107',
        'metrics': {
            'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
            'deep_progressions_90': 'Deep Progressions', 'op_xgbuildup_90': 'xG Buildup p90'
        }
    },
    'defensive_positioning': {
        'name': 'Defensive Positioning', 'color': '#00BCD4',
        'metrics': {
            'padj_interceptions_90': 'PAdj Interceptions', 'dribbled_past_90': 'Times Dribbled Past p90',
            'pressure_regains_90': 'Pressure Regains', 'turnovers_90': 'Ball Security (Inv)'
        }
    },
    'on_ball_security': {
        'name': 'On-Ball Security', 'color': '#607D8B',
        'metrics': {
            'turnovers_90': 'Ball Security (Inv)', 'op_xgbuildup_90': 'xG Buildup p90',
            'fouls_90': 'Fouls Committed', 'passing_ratio': 'Pass Completion %'
        }
    }
}

# Positional groupings
POSITIONAL_CONFIGS = {
    "Fullback": {"archetypes": FULLBACK_ARCHETYPES, "radars": FULLBACK_RADAR_METRICS, "positions": 
                 ['Left Back', 'Left Wing Back', 'Right Back', 'Right Wing Back']},
    "Center Back": {"archetypes": CB_ARCHETYPES, "radars": CB_RADAR_METRICS, "positions": 
                    ['Centre Back', 'Left Centre Back', 'Right Centre Back']},
    "Center Midfielder": {"archetypes": CM_ARCHETYPES, "radars": CM_RADAR_METRICS, "positions": [
        'Centre Attacking Midfielder', 'Centre Defensive Midfielder', 'Left Centre Midfielder', 
        'Left Defensive Midfielder', 'Right Centre Midfielder', 'Right Defensive Midfielder'
    ]},
    "Winger": {"archetypes": WINGER_ARCHETYPES, "radars": WINGER_RADAR_METRICS, "positions": [
        'Left Attacking Midfielder', 'Left Midfielder', 'Left Wing',
        'Right Attacking Midfielder', 'Right Midfielder', 'Right Wing'
    ]},
    "Striker": {"archetypes": STRIKER_ARCHETYPES, "radars": STRIKER_RADAR_METRICS, "positions": [
        'Centre Forward', 'Left Centre Forward', 'Right Centre Forward', 'Secondary Striker'
    ]}
}

ALL_METRICS_TO_PERCENTILE = sorted(list(set(
    metric for pos_config in POSITIONAL_CONFIGS.values()
    for archetype in pos_config['archetypes'].values() for metric in archetype['identity_metrics']
) | set(
    metric for pos_config in POSITIONAL_CONFIGS.values()
    for radar in pos_config['radars'].values() for metric in radar['metrics'].keys()
)))

# --- 4. DATA HANDLING & ANALYSIS FUNCTIONS (UPDATED) ---

@st.cache_resource(ttl=3600)
def get_all_leagues_data(_auth_credentials):
    """Downloads player statistics from all leagues with improved error handling."""
    all_dfs = []
    successful_loads = 0
    failed_loads = 0
    
    try:
        # Test authentication first
        test_url = "https://data.statsbombservices.com/api/v4/competitions"
        test_response = requests.get(test_url, auth=_auth_credentials, timeout=30)
        test_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Authentication failed. Please check your username and password. Error: {e}")
        return None
    
    # Progress tracking
    total_requests = sum(len(season_ids) for season_ids in COMPETITION_SEASONS.values())
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    current_request = 0
    
    for league_id, season_ids in COMPETITION_SEASONS.items():
        league_name = LEAGUE_NAMES.get(league_id, f"League {league_id}")
        
        for season_id in season_ids:
            current_request += 1
            progress = current_request / total_requests
            progress_bar.progress(progress)
            status_text.text(f"Loading {league_name} (Season {season_id})... {current_request}/{total_requests}")
            
            try:
                url = f"https://data.statsbombservices.com/api/v1/competitions/{league_id}/seasons/{season_id}/player-stats"
                response = requests.get(url, auth=_auth_credentials, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                if not data:
                    failed_loads += 1
                    continue
                    
                df_league = pd.json_normalize(data)
                if df_league.empty:
                    failed_loads += 1
                    continue
                
                df_league['league_name'] = league_name
                df_league['competition_id'] = league_id
                df_league['season_id'] = season_id
                all_dfs.append(df_league)
                successful_loads += 1
                
            except Exception:
                failed_loads += 1
                continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if not all_dfs:
        st.error("Could not load any data from the API. Please check your internet connection and API credentials.")
        return None
    
    st.success(f"Successfully loaded data from {successful_loads} league/season combinations.")
    
    try:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    except Exception as e:
        st.error(f"Error combining datasets: {e}")
        return None

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
    if not player_name: return None, None
    exact_matches = df[df['player_name'].str.lower() == player_name.lower()]
    if not exact_matches.empty: return exact_matches.iloc[0].copy(), None
    
    partial_matches = df[df['player_name'].str.lower().str.contains(player_name.lower(), na=False)]
    if not partial_matches.empty:
        suggestions = partial_matches[['player_name', 'team_name']].head(5).to_dict('records')
        return None, suggestions
    return None, None

def detect_player_archetype(target_player, archetypes):
    archetype_scores = {}
    for name, config in archetypes.items():
        metrics = [f"{m}_pct" for m in config['identity_metrics']]
        valid_metrics = [m for m in metrics if m in target_player.index and pd.notna(target_player[m])]
        score = target_player[valid_metrics].mean() if valid_metrics else 0
        archetype_scores[name] = score
    
    best_archetype = max(archetype_scores, key=archetype_scores.get) if archetype_scores else None
    return best_archetype, pd.DataFrame(archetype_scores.items(), columns=['Archetype', 'Affinity Score']).sort_values(by='Affinity Score', ascending=False)

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



# --- 6. RADAR CHART FUNCTIONS (UPDATED) ---

# This function remains the same as your original code, as it's not a part of the requested changes.
def _radar_angles_labels(metrics_dict):
    labels = list(metrics_dict.values())
    metrics = list(metrics_dict.keys())
    return metrics, labels

# This function remains the same as your original code, as it's not a part of the requested changes.
def _player_percentiles_for_metrics(player_series, metrics):
    return [float(player_series.get(f"{m}_pct", 0.0)) for m in metrics]


def _build_scatterpolar_trace(player_series, metrics, player_label, color, show_text=False):
    """
    Builds a single Scatterpolar trace for a player.
    Adjusted for increased transparency and a more distinct hover effect.
    """
    values = _player_percentiles_for_metrics(player_series, metrics)
    # Close the radar by repeating the first point
    values += values[:1]
    theta = metrics + [metrics[0]]

    text_vals = [f"{int(round(v))}" for v in values]
    text_vals[-1] = text_vals[0]

    trace = go.Scatterpolar(
        r=values,
        theta=theta,
        mode="lines+markers+text",
        name=player_label,
        line=dict(width=2, color=color),
        marker=dict(size=5, color=color),
        text=text_vals if show_text else text_vals,
        textfont=dict(size=11, color="rgba(0,0,0,0)"),
        textposition="top center",
        hovertemplate="%{theta}<br>%{r:.0f}th percentile<extra>" + player_label + "</extra>",
        fill="toself",
        # Lower opacity for a more transparent look
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)",
        opacity=0.8,
        legendgroup=player_label,
        hoveron="points+fills",
    )
    return trace

def create_plotly_radar(players_data, radar_config, bg_color="#111111"):
    """
    Generates a Plotly Figure for a radar chart with multiple players.
    Uses the specified color palette for improved readability.
    """
    metrics_dict = radar_config['metrics']
    group_name = radar_config['name']

    # This function is correct and returns a list of metrics and labels
    metrics, labels = _radar_angles_labels(metrics_dict)

    # Updated color palette as requested
    palette = [
        '#FF0000',  # Red
        '#0000FF',  # Blue
        '#00FF00',  # Green
        '#FFA500',  # Orange
        '#FFC0CB',  # Pink
    ]
    # Fallback colors if more than 5 players are compared
    fallback_palette = [
        "#FFFF00", "#00FFFF", "#800080", "#FFD700"
    ]
    full_palette = palette + fallback_palette

    fig = go.Figure()

    for i, player_series in enumerate(players_data):
        player_name = player_series.get('player_name', 'Unknown')
        season_name = player_series.get('season_name', 'Unknown')
        label = f"{player_name} ({season_name})"
        color = full_palette[i % len(full_palette)]

        # Convert hex to RGB for a slight transparency
        rgb_color = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
        rgba_fillcolor = f'rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, 0.2)'
        
        # Get the percentile values for the player's metrics
        percentile_values = _player_percentiles_for_metrics(player_series, metrics)
        
        # Build the trace with the specified color
        trace = go.Scatterpolar(
            r=percentile_values + [percentile_values[0]],
            theta=metrics + [metrics[0]],
            mode="lines+markers+text",
            name=label,
            line=dict(width=2, color=color),
            marker=dict(size=5, color=color),
            text=[f"{int(round(v))}" for v in percentile_values] + [f"{int(round(percentile_values[0]))}"],
            textfont=dict(size=11, color="rgba(0,0,0,0)"),
            textposition="top center",
            hovertemplate="%{theta}<br>%{r:.0f}th percentile<extra>" + label + "</extra>",
            fill="toself",
            fillcolor=rgba_fillcolor,
            opacity=0.8,
            legendgroup=label,
            hoveron="points+fills",
        )
        fig.add_trace(trace)

    fig.update_layout(
        title=dict(
            text=group_name,
            x=0.5, xanchor='center',
            y=0.95, yanchor='top',
            font=dict(size=18, color="white"),
            pad=dict(t=24, b=4, l=4, r=4)
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.15, yanchor="top",
            font=dict(size=11, color="white"),
            itemsizing="trace"
        ),
        polar=dict(
            bgcolor=bg_color,
            radialaxis=dict(range=[0, 100], showline=False, showticklabels=True, tickfont=dict(color="white", size=10),
                             gridcolor="rgba(255,255,255,0.15)", tickangle=0),
            angularaxis=dict(
                tickvals=metrics,
                ticktext=labels,
                tickfont=dict(size=11, color="white"),
                gridcolor="rgba(255,255,255,0.1)"
            )
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(t=80, b=90, l=40, r=40),
        hovermode="closest"
    )

    fig.update_layout(height=520)

    return fig, metrics

def render_plotly_with_legend_hover(fig, metrics, height=520):
    """
    Renders Plotly fig inside Streamlit with custom JS to:
    - On legend hover: highlight target trace (opacity 1.0, text visible), fade others to grey (opacity ~0.2, color grey, text hidden).
    - On legend unhover: reset opacities and hide text labels again.
    - Percentile labels appear on-chart only for hovered player.
    """
    div_id = f"plotly-radar-{uuid.uuid4().hex}"
    html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False, div_id=div_id)
    # Inject custom JS handlers
    custom_js = f"""
    <script>
      (function() {{
        const el = document.getElementById('{div_id}');
        if (!el) return;
        const resetStyles = () => {{
          const gd = el;
          const data = gd.data || [];
          for (let i = 0; i < data.length; i++) {{
            Plotly.restyle(gd, {{
              'opacity': 0.8,
              'textfont.color': 'rgba(0,0,0,0)',
              'line.width': 2,
              'line.color': gd.data[i].line.color
            }}, [i]);
          }}
        }};
        el.addEventListener('plotly_afterplot', function() {{
          resetStyles();
        }});
        el.on('plotly_legendhover', function(evt) {{
          const gd = el;
          const idx = evt.curveNumber;
          if (idx == null) return;
          const n = (gd.data || []).length;
          for (let i = 0; i < n; i++) {{
            if (i === idx) {{
              Plotly.restyle(gd, {{
                'opacity': 1.0,
                'textfont.color': '#ffffff',
                'line.width': 3.5,
                'line.color': gd.data[i].line.color
              }}, [i]);
            }} else {{
              Plotly.restyle(gd, {{
                'opacity': 0.2,
                'textfont.color': 'rgba(0,0,0,0)',
                'line.width': 2,
                'line.color': 'grey'
              }}, [i]);
            }}
          }}
        }});
        el.on('plotly_legendunhover', function(evt) {{
          resetStyles();
        }});
        el.on('plotly_legendclick', function(evt) {{
          // Let Plotly handle visibility toggling; keep our styling consistent
          setTimeout(() => {{
            const gd = el;
            const data = gd.data || [];
            const visibleIdx = [];
            data.forEach((tr, i) => {{
              if (tr.visible === true || tr.visible === undefined) visibleIdx.push(i);
            }});
            if (visibleIdx.length === 1) {{
              for (let i = 0; i < data.length; i++) {{
                Plotly.restyle(gd, {{
                  'textfont.color': (i === visibleIdx[0]) ? '#ffffff' : 'rgba(0,0,0,0)',
                  'opacity': (i === visibleIdx[0]) ? 1.0 : 0.2,
                  'line.width': (i === visibleIdx[0]) ? 3.5 : 2,
                  'line.color': (i === visibleIdx[0]) ? gd.data[i].line.color : 'grey'
                }}, [i]);
              }}
            }} else {{
              const n = data.length;
              for (let i = 0; i < n; i++) {{
                Plotly.restyle(gd, {{
                  'textfont.color': 'rgba(0,0,0,0)',
                  'opacity': 0.8,
                  'line.width': 2,
                  'line.color': gd.data[i].line.color
                }}, [i]);
              }}
            }}
          }}, 0);
          return false;
        }});
      }})();
    </script>
    """
    # Create a full-screen button
    if st.button("Pop-out Radar Chart", key=f"fullscreen_{div_id}", help="Click to expand this chart vertically."):
        components.html(wrapped, height=height + 60, scrolling=False)
        st.session_state[f'expanded_{div_id}'] = not st.session_state.get(f'expanded_{div_id}', False)

    height_to_use = 1000 if st.session_state.get(f'expanded_{div_id}') else height

    wrapped = f"""
    <div style="margin-bottom: 28px;">
      {html}
      {custom_js}
    </div>
    """
    components.html(wrapped, height=height_to_use + 60, scrolling=False)
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
    leagues = sorted(data['league_name'].dropna().unique())
    
    selected_league = st.selectbox("League", leagues, key=f"{key_prefix}_league", index=None, placeholder="Choose a league")
    
    if selected_league:
        league_df = data[data['league_name'] == selected_league]
        seasons = sorted(league_df['season_name'].unique())
        selected_season = st.selectbox("Season", seasons, key=f"{key_prefix}_season", index=None, placeholder="Choose a season")
        
        if selected_season:
            season_df = league_df[league_df['season_name'] == selected_season]
            
            if pos_filter:
                valid_positions = POSITIONAL_CONFIGS[pos_filter]['positions']
                season_df_filtered = season_df[season_df['primary_position'].isin(valid_positions)]
                
                # Diagnostic message
                if season_df_filtered.empty and not season_df.empty:
                    available_pos = sorted(season_df['primary_position'].unique())
                    st.warning(f"No players found for '{pos_filter}'. Available positions in this selection: {available_pos}")
                    return None
                season_df = season_df_filtered

            teams = ["All Teams"] + sorted(season_df['team_name'].unique())
            selected_team = st.selectbox("Team", teams, key=f"{key_prefix}_team")
            
            if selected_team:
                if selected_team != "All Teams":
                    player_pool = season_df[season_df['team_name'] == selected_team]
                else:
                    player_pool = season_df
                
                if player_pool.empty:
                    st.warning(f"No players found for the selected filters.")
                    return None

                player_pool_display = player_pool.copy()
                player_pool_display['age_str'] = player_pool_display['age'].apply(lambda x: str(int(x)) if pd.notna(x) else 'N/A')
                player_pool_display['display_name'] = player_pool_display['player_name'] + " (" + player_pool_display['age_str'] + ", " + player_pool_display['primary_position'].fillna('N/A') + ")"
                
                players = sorted(player_pool_display['display_name'].unique())
                selected_display_name = st.selectbox("Player", players, key=f"{key_prefix}_player", index=None, placeholder="Choose a player")
                
                if selected_display_name:
                    player_instance_df = player_pool_display[player_pool_display['display_name'] == selected_display_name]
                    if not player_instance_df.empty:
                        original_index = player_instance_df.index[0]
                        return data.loc[original_index]
    return None

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

        if st.session_state.analysis_run and 'target_player' in st.session_state and st.session_state.target_player is not None:
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
                if st.session_state.matches is not None and not st.session_state.matches.empty:
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
                            if not any(
                                p['player_id'] == row['player_id'] and 
                                p['season_id'] == row['season_id']
                                for p in st.session_state.radar_players
                            ):
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
    st.header("Multi-Player Direct Comparison")

    if processed_data is not None:
        # Ensure state is initialized
        if 'comp_selections' not in st.session_state:
            st.session_state.comp_selections = {"league": None, "season": None, "team": None, "player": None}
        if 'comparison_players' not in st.session_state:
            st.session_state.comparison_players = []
            
        def player_filter_ui_comp(data, key_prefix):
            state = st.session_state.comp_selections
            
            leagues = sorted(data['league_name'].dropna().unique())
            
            league_idx = leagues.index(state['league']) if state['league'] in leagues else None
            selected_league = st.selectbox("League", leagues, key=f"{key_prefix}_league", index=league_idx, placeholder="Choose a league")
            
            if selected_league != state['league']:
                st.session_state.comp_selections['league'] = selected_league
                st.session_state.comp_selections['season'] = None
                st.session_state.comp_selections['team'] = None
                st.session_state.comp_selections['player'] = None
                st.rerun()

            if st.session_state.comp_selections['league']:
                league_df = data[data['league_name'] == st.session_state.comp_selections['league']]
                seasons = sorted(league_df['season_name'].unique())
                season_idx = seasons.index(state['season']) if state['season'] in seasons else None
                selected_season = st.selectbox("Season", seasons, key=f"{key_prefix}_season", index=season_idx, placeholder="Choose a season")
                
                if selected_season != state['season']:
                    st.session_state.comp_selections['season'] = selected_season
                    st.session_state.comp_selections['team'] = None
                    st.session_state.comp_selections['player'] = None
                    st.rerun()

            if st.session_state.comp_selections['season']:
                season_df = data[
                    (data['league_name'] == st.session_state.comp_selections['league']) & 
                    (data['season_name'] == st.session_state.comp_selections['season'])
                ]
                teams = ["All Teams"] + sorted(season_df['team_name'].unique())
                team_idx = teams.index(state['team']) if state['team'] in teams else 0
                selected_team = st.selectbox("Team", teams, key=f"{key_prefix}_team", index=team_idx)
                
                if selected_team != state['team']:
                    st.session_state.comp_selections['team'] = selected_team
                    st.session_state.comp_selections['player'] = None
                    st.rerun()

            if st.session_state.comp_selections['team']:
                if st.session_state.comp_selections['team'] != "All Teams":
                    player_pool = data[
                        (data['league_name'] == st.session_state.comp_selections['league']) & 
                        (data['season_name'] == st.session_state.comp_selections['season']) & 
                        (data['team_name'] == st.session_state.comp_selections['team'])
                    ]
                else:
                    player_pool = data[
                        (data['league_name'] == st.session_state.comp_selections['league']) & 
                        (data['season_name'] == st.session_state.comp_selections['season'])
                    ]
                
                players = sorted(player_pool['player_name'].unique())
                player_idx = players.index(state['player']) if state['player'] in players else None
                selected_player_name = st.selectbox("Player", players, key=f"{key_prefix}_player", index=player_idx, placeholder="Choose a player")
                st.session_state.comp_selections['player'] = selected_player_name
            
            if st.session_state.comp_selections['player']:
                player_instance = processed_data[
                    (processed_data['player_name'] == st.session_state.comp_selections['player']) & 
                    (processed_data['season_name'] == st.session_state.comp_selections['season']) &
                    (processed_data['league_name'] == st.session_state.comp_selections['league'])
                ]
                if not player_instance.empty:
                    return player_instance.iloc[0]
            return None

        with st.container(border=True):
            st.subheader("Add a Player to Comparison")
            player_instance = player_filter_ui_comp(processed_data, key_prefix="comp")

            if st.button("Add Player", type="primary"):
                if player_instance is not None:
                    # Create a unique identifier for the player+season
                    player_id = f"{player_instance['player_id']}_{player_instance['season_id']}"
                    
                    # Check if already added
                    if not any(
                        f"{p['player_id']}_{p['season_id']}" == player_id 
                        for p in st.session_state.comparison_players
                    ):
                        st.session_state.comparison_players.append(player_instance)
                        st.rerun()
                    else:
                        st.warning("This player and season is already in the comparison.")
                else:
                    st.warning("Please select a valid player.")

        st.divider()

        st.subheader("Current Comparison")
        if not st.session_state.comparison_players:
            st.info("Add one or more players using the selection box above to start a comparison.")
        else:
            player_cols = st.columns(len(st.session_state.comparison_players) or 1)
            for i, player_data in enumerate(st.session_state.comparison_players):
                with player_cols[i]:
                    st.markdown(f"**{player_data['player_name']}**")
                    st.markdown(f"*{player_data['team_name']}*")
                    st.markdown(f"`{player_data['season_name']}`")
                    if st.button("Remove", key=f"remove_comp_{i}"):
                        st.session_state.comparison_players.pop(i)
                        st.rerun()

        st.divider()
        
        if st.session_state.comparison_players:
            st.subheader("Radar Chart Comparison")
            
            radar_pos_options = list(POSITIONAL_CONFIGS.keys())
            selected_radar_pos = st.selectbox("Select Radar Set to Use for Comparison", radar_pos_options)
            
            radars_to_show = POSITIONAL_CONFIGS[selected_radar_pos]['radars']
            
            # Layout: 3 columns per row
            num_radars = len(radars_to_show)
            cols = st.columns(3) 
            radar_items = list(radars_to_show.items())

            for i in range(num_radars):
                with cols[i % 3]:
                    radar_key, radar_config = radar_items[i]
                    fig, metrics = create_plotly_radar(st.session_state.comparison_players, radar_config)
                    render_plotly_with_legend_hover(fig, metrics, height=520)
    else:
        st.error("Data could not be loaded. Please check your credentials.")
