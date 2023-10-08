import pandas as pd

# Load the cleaned dataframes
anime_df = pd.read_csv("clean_anime.csv")
scores_df = pd.read_csv("clean_scores.csv")

# Global variables
num_users = scores_df['encoded_user_id'].nunique()
num_animes = scores_df['encoded_anime_id'].nunique()
