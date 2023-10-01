
# Import Statements
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import torch
import pandas as pd
import numpy as np
import re
from model import NCF

app = Flask(__name__)

# Load datasets
anime_df = pd.read_csv('clean_anime.csv', index_col=0)
scores_df = pd.read_csv('clean_scores.csv', index_col=0)

# Global Settings / variables
app.secret_key = '65489dfg4s654654df' # Set to use dot_env before going live
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_users = scores_df['user_id'].nunique()
num_animes = anime_df['anime_id'].nunique()

# Load model and set to eval mode
model = NCF(num_users, num_animes)
model.load_state_dict(torch.load('recommender_model.pth'))
model.eval()

# Set user IDs as all users are new users
new_user_id = scores_df['user_id'].max()
new_user_id += 1

def get_top_n_recommendations(processed_ratings, N=5):
    # Convert the dictionary keys and values to lists
    anime_ids = list(map(int, processed_ratings.keys()))
    ratings = list(map(int, processed_ratings.values()))
    
    # Convert lists to numpy arrays
    anime_ids_array = np.array(anime_ids)
    ratings_array = np.array(ratings)
    
    # Stack arrays to create a 2D array of shape (len(processed_ratings), 2)
    user_vector = np.column_stack((anime_ids_array, ratings_array))
    
    # Convert the numpy array to a PyTorch tensor
    user_vector_tensor = torch.tensor(user_vector).float().to(device)
    
    # Prepare the input data - pair the user vector with all possible animes
    all_anime_ids = anime_df['anime_id'].unique().tolist()
    user_vector = user_vector.repeat(len(anime_df), 1)
    anime_tensor = torch.tensor(all_anime_ids, dtype=torch.long).to(device)
    
    # Get the predictions
    with torch.no_grad():
        predictions = model(user_vector, anime_tensor).squeeze(-1)
    
    # Pair predictions with anime ids
    paired_predictions = list(zip(all_anime_ids, predictions.cpu().numpy()))
    
    # Sort by the prediction values
    paired_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top N anime ids
    top_n_anime_ids = [x[0] for x in paired_predictions[:N]]
    
    # Return the top N anime
    return anime_df.iloc[top_n_anime_ids]

def get_top10_anime():
    # Filter out anime with Popularity of 0
    top_anime = anime_df[(anime_df['Popularity'] > 0)]
    
    # Sort the filtered anime by rank and get the top 10
    top_30_anime = top_anime.sort_values(by='Popularity').head(30)
    
    # Randomly select 10 out of the top 30
    random_10_anime = top_30_anime.sample(10)
   
    return random_10_anime.to_dict(orient='records')

# Get top 10 anime as JSON
@app.route('/top10anime')
def top10anime():
    return jsonify(get_top10_anime())
 
# Display ratings page
@app.route('/rate_anime')
def rate_anime():
    # Get the top 10 anime list
    anime_list = get_top10_anime()
    return render_template('ratings.html', anime_list=anime_list)

# Submit user generated ratings to recommend 
@app.route('/submit_ratings', methods=['POST'])
def submit_ratings():
    # Get ratings from the form
    ratings = request.form.to_dict()

    # Process the ratings (e.g., remove any anime rated 0)
    processed_ratings = {k: v for k, v in ratings.items() if v != "0"}

    # Store ratings in a session variable
    session['user_ratings'] = processed_ratings

    # Redirect to the recommendation route to display the recommendations
    return redirect(url_for('recommend'))
 
# Make recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    # Retrieve user ratings from the session
    user_ratings = session.get('user_ratings', {})

    # Process the ratings to extract numeric anime ID, remove any anime rated 0, and convert string ratings to integers
    processed_ratings = {}
    for k, v in user_ratings.items():
        match = re.search(r'(\d+)$', k)
        if match and v != "0":
            anime_id = int(match.group())
            rating = int(v)
            processed_ratings[anime_id] = rating

    # Get the top 5 recommendations for the user using the processed_ratings
    recommendations = get_top_n_recommendations(processed_ratings, N=5)

    # Render the recommendations on the watchnext.html page
    return render_template('watchnext.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)