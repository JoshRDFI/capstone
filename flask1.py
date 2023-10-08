# Import Statements
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import torch
import pandas as pd
import numpy as np
import re
from model import NCF

app = Flask(__name__)

# Load datasets
from data_loader import anime_df, scores_df

# Global Settings / variables
app.secret_key = '65489dfg4s654654df' # Set to use dot_env before going live
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from data_loader import num_users, num_animes

# Load model and set to eval mode
model = NCF(num_users, num_animes)
model.load_state_dict(torch.load('recommender_model.pth'))
model = model.to(device)
model.eval()

# Set user IDs as all users are new users
new_user_id = scores_df['encoded_user_id'].max()
new_user_id += 1

def get_top_n_recommendations(user_ratings, N=10):
    # Ensure the model is set to evaluation mode
    model.eval()

    # Convert user ratings to a user vector
    user_vector = torch.zeros(num_animes).to(device)
    for anime_id, rating in user_ratings.items():
        user_vector[anime_id] = rating

    # Convert the user vector to a tensor
    user_vector_tensor = user_vector.unsqueeze(0)

    # Create a tensor of all anime for which we want to predict the user's ratings
    anime_tensor = torch.tensor(list(range(num_animes)), dtype=torch.int64).to(device)

    # Get the model's predictions for these animes
    with torch.no_grad():
        predictions = model(user_vector_tensor, anime_tensor)

    # Get the top N anime recommendations
    _, indices = torch.topk(predictions.squeeze(), N, dim=0)
    recommended_anime_ids = indices.squeeze().tolist()
    recommended_anime_ids = [anime_df.iloc[id]['anime_id'] for id in recommended_anime_ids]

    # Convert list of IDs to a DataFrame
    recommendations_df = anime_df[anime_df['anime_id'].isin(recommended_anime_ids)]
    return recommendations_df

# Make recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    # Retrieve user ratings from the session
    user_ratings = session.get('user_ratings', {})
    anime_id_to_encoded = dict(zip(anime_df['anime_id'], anime_df['encoded_anime_id']))

    processed_ratings = {}
    for k, v in user_ratings.items():
        match = re.search(r'(\d+)$', k)
        if match and v != "0":
            anime_id = int(match.group())
            rating = int(v)
            encoded_anime_id = anime_id_to_encoded.get(anime_id, None)
            if encoded_anime_id is not None:
                processed_ratings[encoded_anime_id] = rating

    # Get the top 5 recommendations for the user using the processed_ratings
    recommendations = get_top_n_recommendations(processed_ratings, N=5)

    # Render the recommendations on the watchnext.html page
    return render_template('watchnext.html', recommendations=recommendations.to_dict(orient='records'))


def get_top10_anime():
    # Filter out anime with Popularity of 0
    top_anime = anime_df[(anime_df['Popularity'] > 0)]
    
    # Sort the filtered anime by rank
    top_30_anime = top_anime.sort_values(by='Popularity').head(30)
    
    # Randomly select 10 out of the top 30
    random_10_anime = top_30_anime.sample(10)
   
    return random_10_anime.to_dict(orient='records')

# Get top 10 anime as JSON
@app.route('/top10anime')
def top10anime():
    return jsonify(get_top10_anime())
 
# Display user ratings page
@app.route('/rate_anime')
def rate_anime():
    anime_list = get_top10_anime()
    return render_template('ratings.html', anime_list=anime_list)

# Submit user generated ratings to recommend 
@app.route('/submit_ratings', methods=['POST'])
def submit_ratings():
    # Get ratings from the form
    ratings = request.form.to_dict()

    # Process the ratings and remove any anime rated 0
    processed_ratings = {k: v for k, v in ratings.items() if v != "0"}

    # Store ratings
    session['user_ratings'] = processed_ratings

    # Redirect to the recommendation route to display the recommendations
    return redirect(url_for('recommend'))


if __name__ == '__main__':
    app.run(debug=True)